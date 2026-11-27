/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * What the hand-written WebGPU network computes: the Twig Capture local model
 * (twigcapture scripts/model_utils.py LocalOnlyClassifier, a torchvision
 * r3d_18 with a (3, 7, 7) stem) as a fixed list of steps, every weight read
 * from the ONNX file by position and checked against the architecture.
 *
 * The structure follows Philipp Schlegel's segclrPlan.ts in Coda
 * (https://github.com/navis-org/coda, src/packs/seam/gpuAssessor, MIT): a
 * fixed architecture rather than a general ONNX interpreter, so a file that
 * is not this network is refused when it is opened, with the reason, instead
 * of computing something else.
 *
 * Two exports are read, with the same weights:
 * - the plain export (export_local_onnx.py): 5-D Conv nodes, weights
 *   (Cout, Cin, kd, kh, kw);
 * - the Conv2d rewrite that ONNX Runtime Web is served today
 *   (conv3d_as_conv2d.py, `concat`): each 3D Conv as one Conv2d over the kd
 *   depth taps stacked along channels, weights (Cout, kd * Cin, kh, kw) with
 *   the tap outermost. The weights are put back in (Cout, Cin, kd, kh, kw)
 *   order here; depth stride and padding come from the architecture.
 * In both, batch norm is already folded into each Conv's bias by the export.
 *
 * Layout everywhere: channels outermost, then depth, height, width
 * ([C][D][H][W], width fastest), which is the model input's own
 * [3, 18, 128, 128].
 */

import type {
  OnnxGraph,
  OnnxNode,
} from "#src/layer/segmentation/twig_inference/gpu/onnx_proto.js";

export type Vec3 = readonly [number, number, number];

export interface ConvStep {
  readonly name: string;
  readonly cin: number;
  readonly cout: number;
  /** Input and output extents (depth, height, width). */
  readonly inShape: Vec3;
  readonly outShape: Vec3;
  readonly kernel: Vec3;
  readonly stride: Vec3;
  readonly pad: Vec3;
  /** Buffer names: the patch is "input". */
  readonly input: string;
  readonly output: string;
  /** ReLU after the bias (a block's first conv, and the stem). */
  readonly relu: boolean;
  /** Added after the bias, then ReLU (a block's second conv). */
  readonly residual?: string;
  /** (Cout, Cin, kd, kh, kw), a fresh array per call. */
  readonly weight: () => Float32Array;
  readonly bias: Float32Array;
}

export interface HeadStep {
  readonly input: string;
  readonly channels: number;
  readonly voxels: number;
  /** fc.0: (hidden, channels) and (hidden); fc.3: (2, hidden) and (2). */
  readonly hidden: number;
  readonly w1: Float32Array;
  readonly b1: Float32Array;
  readonly w2: Float32Array;
  readonly b2: Float32Array;
}

export interface R3dPlan {
  /** In the order they run: a block's downsample before its second conv. */
  readonly convs: readonly ConvStep[];
  readonly head: HeadStep;
  /** The patch: (channels, depth, height, width). */
  readonly input: readonly [number, number, number, number];
  /** Multiply-adds x 2 per patch. */
  readonly flops: number;
}

export class PlanError extends Error {}

interface Spec {
  name: string;
  cin: number;
  cout: number;
  kernel: Vec3;
  stride: Vec3;
  pad: Vec3;
}

const K3: Vec3 = [3, 3, 3];
const S1: Vec3 = [1, 1, 1];
const S2: Vec3 = [2, 2, 2];
const P1: Vec3 = [1, 1, 1];

/** The convolutions in file order (torchvision's: conv1, conv2, downsample). */
export function r3dSpecs(): Spec[] {
  const out: Spec[] = [
    {
      name: "stem",
      cin: 3,
      cout: 64,
      kernel: [3, 7, 7],
      stride: [1, 2, 2],
      pad: [1, 3, 3],
    },
  ];
  const widths = [64, 128, 256, 512];
  widths.forEach((w, li) => {
    const prev = li ? widths[li - 1] : 64;
    const first = li ? S2 : S1;
    const L = `layer${li + 1}`;
    out.push({
      name: `${L}.0.conv1`,
      cin: prev,
      cout: w,
      kernel: K3,
      stride: first,
      pad: P1,
    });
    out.push({
      name: `${L}.0.conv2`,
      cin: w,
      cout: w,
      kernel: K3,
      stride: S1,
      pad: P1,
    });
    if (li)
      out.push({
        name: `${L}.0.downsample`,
        cin: prev,
        cout: w,
        kernel: [1, 1, 1],
        stride: S2,
        pad: [0, 0, 0],
      });
    out.push({
      name: `${L}.1.conv1`,
      cin: w,
      cout: w,
      kernel: K3,
      stride: S1,
      pad: P1,
    });
    out.push({
      name: `${L}.1.conv2`,
      cin: w,
      cout: w,
      kernel: K3,
      stride: S1,
      pad: P1,
    });
  });
  return out;
}

const outExtent = (i: number, k: number, s: number, p: number) =>
  Math.floor((i + 2 * p - k) / s) + 1;

function ints(node: OnnxNode, key: string): number[] | undefined {
  const v = node.attributes.get(key);
  return Array.isArray(v) ? [...v] : undefined;
}

const same = (a: readonly number[] | undefined, b: readonly number[]) =>
  !!a && a.length === b.length && a.every((v, i) => v === b[i]);

/** (Cout, kd * Cin, kh, kw), tap outermost -> (Cout, Cin, kd, kh, kw). */
export function unstackDepthTaps(
  w: Float32Array,
  cout: number,
  cin: number,
  kernel: Vec3,
): Float32Array {
  const [kd, kh, kw] = kernel;
  const plane = kh * kw;
  const out = new Float32Array(w.length);
  for (let co = 0; co < cout; ++co)
    for (let td = 0; td < kd; ++td)
      for (let ci = 0; ci < cin; ++ci) {
        const src = ((co * kd + td) * cin + ci) * plane;
        const dst = ((co * cin + ci) * kd + td) * plane;
        out.set(w.subarray(src, src + plane), dst);
      }
  return out;
}

/** The model input the plan is built for. */
export const R3D_INPUT: readonly [number, number, number, number] = [
  3, 18, 128, 128,
];

/** Read the network from an ONNX graph, or throw PlanError saying why not. */
export function r3dPlan(graph: OnnxGraph): R3dPlan {
  const specs = r3dSpecs();
  const convNodes = graph.nodes.filter((n) => n.opType === "Conv");
  const gemms = graph.nodes.filter((n) => n.opType === "Gemm");
  if (convNodes.length !== specs.length)
    throw new PlanError(
      `expected ${specs.length} convolutions (r3d_18), found ${convNodes.length}`,
    );
  if (gemms.length !== 2)
    throw new PlanError(
      `expected 2 fully connected layers, found ${gemms.length}`,
    );
  const tensor = (name: string | undefined, dims: number[], what: string) => {
    const t = name === undefined ? undefined : graph.initializers.get(name);
    if (!t) throw new PlanError(`${what}: no weight ${name ?? "(none)"}`);
    if (!t.data)
      throw new PlanError(`${what}: ${t.name} unreadable: ${t.unread}`);
    if (!same(t.dims, dims))
      throw new PlanError(
        `${what}: ${t.name} is ${t.dims.join("x")}, expected ${dims.join("x")}`,
      );
    return t;
  };

  const shapes = new Map<string, Vec3>([
    ["input", [R3D_INPUT[1], R3D_INPUT[2], R3D_INPUT[3]]],
  ]);
  let flops = 0;
  const steps = new Map<string, ConvStep>();
  specs.forEach((sp, i) => {
    const node = convNodes[i];
    const [kd, kh, kw] = sp.kernel;
    const w4 = graph.initializers.get(node.inputs[1])?.dims.length === 4;
    let weight: () => Float32Array;
    const what = `${sp.name} (Conv ${i})`;
    if (w4) {
      // Conv2d rewrite: the depth taps are stacked along channels.
      const wt = tensor(node.inputs[1], [sp.cout, kd * sp.cin, kh, kw], what);
      if (!same(ints(node, "strides"), [sp.stride[1], sp.stride[2]]))
        throw new PlanError(`${what}: strides ${ints(node, "strides")}`);
      if (
        !same(ints(node, "pads"), [sp.pad[1], sp.pad[2], sp.pad[1], sp.pad[2]])
      )
        throw new PlanError(`${what}: pads ${ints(node, "pads")}`);
      weight = stacked(wt.data!, sp.cout, sp.cin, sp.kernel);
    } else {
      const wt = tensor(node.inputs[1], [sp.cout, sp.cin, kd, kh, kw], what);
      if (!same(ints(node, "strides"), [...sp.stride]))
        throw new PlanError(`${what}: strides ${ints(node, "strides")}`);
      const pads = ints(node, "pads") ?? [0, 0, 0, 0, 0, 0];
      if (!same(pads, [...sp.pad, ...sp.pad]))
        throw new PlanError(`${what}: pads ${pads}`);
      weight = plain(wt.data!);
    }
    const groups = node.attributes.get("group");
    if (groups !== undefined && groups !== 1)
      throw new PlanError(`${what}: grouped convolution`);
    const bias = tensor(node.inputs[2], [sp.cout], `${what} bias`)
      .data!().slice();
    const isDown = sp.name.endsWith("downsample");
    const isConv2 = sp.name.endsWith("conv2");
    const block = sp.name.replace(/\.conv[12]$|\.downsample$/, "");
    // The patch, the previous block's output, or this block's first conv.
    const input =
      sp.name === "stem"
        ? "input"
        : isConv2
          ? `${block}.conv1`
          : `${prevBlock(sp.name)}.out`;
    const inShape = shapes.get(input);
    if (!inShape) throw new PlanError(`${what}: no input ${input}`);
    const outShape = [0, 1, 2].map((d) =>
      outExtent(inShape[d], sp.kernel[d], sp.stride[d], sp.pad[d]),
    ) as unknown as Vec3;
    const hasDown = specs.some((o) => o.name === `${block}.downsample`);
    const residual = isConv2
      ? sp.name.endsWith(".0.conv2") && hasDown
        ? `${block}.downsample`
        : `${prevBlock(sp.name)}.out`
      : undefined;
    const output =
      sp.name === "stem" ? "stem.out" : isConv2 ? `${block}.out` : sp.name;
    shapes.set(output, outShape);
    const step: ConvStep = {
      name: sp.name,
      cin: sp.cin,
      cout: sp.cout,
      inShape,
      outShape,
      kernel: sp.kernel,
      stride: sp.stride,
      pad: sp.pad,
      input,
      output,
      relu: !isDown,
      residual,
      weight,
      bias,
    };
    steps.set(sp.name, step);
    flops +=
      2 *
      sp.cout *
      sp.cin *
      kd *
      kh *
      kw *
      outShape[0] *
      outShape[1] *
      outShape[2];
  });

  // Run order: a block's downsample must exist before its second conv adds it.
  const convs: ConvStep[] = [];
  for (const sp of specs) {
    if (sp.name.endsWith(".0.conv2")) {
      const ds = steps.get(sp.name.replace(".conv2", ".downsample"));
      if (ds) convs.push(ds);
    }
    if (!sp.name.endsWith("downsample")) convs.push(steps.get(sp.name)!);
  }
  // Every residual must have the shape of the conv it is added to.
  for (const c of convs) {
    if (!c.residual) continue;
    const r = convs.find((o) => o.output === c.residual);
    if (!r || r.cout !== c.cout || !same(r.outShape, c.outShape))
      throw new PlanError(`${c.name}: residual ${c.residual} does not match`);
  }

  const last = convs[convs.length - 1];
  const [g1, g2] = gemms;
  const channels = last.cout;
  const w1t = graph.initializers.get(g1.inputs[1]);
  const hidden = w1t?.dims[0] ?? 0;
  const fc1 = tensor(g1.inputs[1], [hidden, channels], "fc.0");
  const fc1b = tensor(g1.inputs[2], [hidden], "fc.0 bias");
  const fc2 = tensor(g2.inputs[1], [2, hidden], "fc.3");
  const fc2b = tensor(g2.inputs[2], [2], "fc.3 bias");
  for (const g of gemms)
    if (g.attributes.get("transB") !== 1)
      throw new PlanError(
        "fully connected layers must be stored transposed (transB)",
      );
  const voxels = last.outShape[0] * last.outShape[1] * last.outShape[2];
  return {
    convs,
    head: {
      input: last.output,
      channels,
      voxels,
      hidden,
      w1: fc1.data!().slice(),
      b1: fc1b.data!().slice(),
      w2: fc2.data!().slice(),
      b2: fc2b.data!().slice(),
    },
    input: R3D_INPUT,
    flops,
  };
}

/** "layer2.0.conv1" -> "layer1.1"; "layer1.0.*" -> "stem"; "layerN.1.*" -> "layerN.0". */
function prevBlock(name: string): string {
  const m = /^layer(\d)\.(\d)\./.exec(name)!;
  const L = +m[1];
  const b = +m[2];
  if (b === 1) return `layer${L}.0`;
  return L === 1 ? "stem" : `layer${L - 1}.1`;
}

/*
 * Weight readers made outside r3dPlan, holding only what they read: a closure
 * made inside it would keep the whole plan's scope alive with it.
 */
function stacked(
  read: () => Float32Array,
  cout: number,
  cin: number,
  kernel: Vec3,
): () => Float32Array {
  return () => unstackDepthTaps(read(), cout, cin, kernel);
}

function plain(read: () => Float32Array): () => Float32Array {
  return () => read().slice();
}
