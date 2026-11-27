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

// The hand-written WebGPU network's plan and shaders, without a GPU: the
// r3d_18 wiring read from a graph shaped like our exports, the weight
// rearrangements, and the refusals. Parity with PyTorch needs a GPU and is
// checked in the browser (see BRIEF.md, "The model").

import { readFileSync, existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type {
  OnnxGraph,
  OnnxNode,
  OnnxTensor,
} from "#src/layer/segmentation/twig_inference/gpu/onnx_proto.js";
import { readOnnx } from "#src/layer/segmentation/twig_inference/gpu/onnx_proto.js";
import {
  PlanError,
  r3dPlan,
  r3dSpecs,
  unstackDepthTaps,
} from "#src/layer/segmentation/twig_inference/gpu/r3d_plan.js";
import {
  convLayout,
  convShader,
  convWeightLayout,
  frugalShaders,
  headShader,
  reduceShader,
} from "#src/layer/segmentation/twig_inference/gpu/r3d_shaders.js";

/** A graph with our exports' Conv/Gemm nodes, weights made on demand. */
function fakeGraph(form: "c2d" | "plain", tweak?: (i: number) => number[]) {
  const initializers = new Map<string, OnnxTensor>();
  const nodes: OnnxNode[] = [];
  const add = (name: string, dims: number[]) => {
    const n = dims.reduce((a, b) => a * b, 1);
    initializers.set(name, {
      name,
      dims,
      data: () => Float32Array.from({ length: n }, (_, i) => (i % 7) * 0.01),
    });
  };
  r3dSpecs().forEach((sp, i) => {
    const [kd, kh, kw] = sp.kernel;
    let dims =
      form === "c2d"
        ? [sp.cout, kd * sp.cin, kh, kw]
        : [sp.cout, sp.cin, kd, kh, kw];
    if (tweak) dims = tweak(i) ?? dims;
    add(`w${i}`, dims);
    add(`b${i}`, [sp.cout]);
    const attributes = new Map<string, number | string | readonly number[]>(
      form === "c2d"
        ? [
            ["kernel_shape", [kh, kw]],
            ["strides", [sp.stride[1], sp.stride[2]]],
            ["pads", [sp.pad[1], sp.pad[2], sp.pad[1], sp.pad[2]]],
          ]
        : [
            ["kernel_shape", [kd, kh, kw]],
            ["strides", [...sp.stride]],
            ["pads", [...sp.pad, ...sp.pad]],
          ],
    );
    nodes.push({
      opType: "Conv",
      inputs: [`x${i}`, `w${i}`, `b${i}`],
      outputs: [`y${i}`],
      attributes,
    });
  });
  add("fc.0.weight", [256, 512]);
  add("fc.0.bias", [256]);
  add("fc.3.weight", [2, 256]);
  add("fc.3.bias", [2]);
  for (const [w, b] of [
    ["fc.0.weight", "fc.0.bias"],
    ["fc.3.weight", "fc.3.bias"],
  ])
    nodes.push({
      opType: "Gemm",
      inputs: ["f", w, b],
      outputs: ["g"],
      attributes: new Map([["transB", 1]]),
    });
  return { initializers, nodes } as OnnxGraph;
}

describe("r3dPlan", () => {
  for (const form of ["c2d", "plain"] as const) {
    it(`reads r3d_18 from the ${form} export`, () => {
      const plan = r3dPlan(fakeGraph(form));
      expect(plan.convs).toHaveLength(20);
      expect(plan.convs[0].outShape).toEqual([18, 64, 64]);
      const names = plan.convs.map((c) => c.name);
      // A block's downsample runs before the conv that adds it.
      expect(names.indexOf("layer2.0.downsample")).toBeLessThan(
        names.indexOf("layer2.0.conv2"),
      );
      const by = (n: string) => plan.convs.find((c) => c.name === n)!;
      expect(by("layer1.0.conv1").input).toBe("stem.out");
      expect(by("layer1.0.conv2").residual).toBe("stem.out");
      expect(by("layer2.0.conv2").residual).toBe("layer2.0.downsample");
      expect(by("layer2.0.downsample").input).toBe("layer1.1.out");
      expect(by("layer2.1.conv2").residual).toBe("layer2.0.out");
      expect(by("layer2.0.downsample").relu).toBe(false);
      expect(by("layer2.0.conv1").outShape).toEqual([9, 32, 32]);
      expect(by("layer3.0.conv1").outShape).toEqual([5, 16, 16]);
      expect(by("layer4.1.conv2").outShape).toEqual([3, 8, 8]);
      expect(plan.head).toMatchObject({
        channels: 512,
        voxels: 192,
        hidden: 256,
      });
      // ~124 GFLOP a patch.
      expect(plan.flops / 1e9).toBeCloseTo(123.7, 0);
    });
  }

  it("refuses a weight of the wrong shape, naming it", () => {
    expect(() =>
      r3dPlan(
        fakeGraph("plain", (i) =>
          i === 3 ? [64, 64, 3, 3, 1] : (undefined as any),
        ),
      ),
    ).toThrow(PlanError);
    expect(() =>
      r3dPlan(
        fakeGraph("plain", (i) =>
          i === 3 ? [64, 64, 3, 3, 1] : (undefined as any),
        ),
      ),
    ).toThrow(/layer1.1.conv1/);
  });

  it("refuses a graph that is not r3d_18", () => {
    const g = fakeGraph("plain");
    expect(() => r3dPlan({ ...g, nodes: g.nodes.slice(1) })).toThrow(
      /expected 20 convolutions/,
    );
  });
});

describe("weights", () => {
  it("puts stacked depth taps back in (Cout, Cin, kd, kh, kw) order", () => {
    const cout = 2;
    const cin = 3;
    const k = [2, 1, 2] as const;
    // Stacked index ((co * kd + td) * cin + ci) * kh*kw + p holds a code.
    const w = new Float32Array(cout * 2 * cin * 2);
    for (let co = 0; co < cout; ++co)
      for (let td = 0; td < 2; ++td)
        for (let ci = 0; ci < cin; ++ci)
          for (let p = 0; p < 2; ++p)
            w[((co * 2 + td) * cin + ci) * 2 + p] =
              co * 1000 + ci * 100 + td * 10 + p;
    const out = unstackDepthTaps(w, cout, cin, k);
    for (let co = 0; co < cout; ++co)
      for (let ci = 0; ci < cin; ++ci)
        for (let td = 0; td < 2; ++td)
          for (let p = 0; p < 2; ++p)
            expect(out[((co * cin + ci) * 2 + td) * 2 + p]).toBe(
              co * 1000 + ci * 100 + td * 10 + p,
            );
  });

  it("lays weights out tap-major for the shader, zero-padded", () => {
    const plan = r3dPlan(fakeGraph("plain"));
    const stem = plan.convs[0];
    const L = convLayout(stem);
    expect(L.k).toBe(3 * 147);
    const w = stem.weight();
    const out = convWeightLayout(stem, L, w);
    expect(out.length).toBe(L.chunks * 16 * 64);
    // k = tap * Cin + ci, row of Cout.
    for (const [co, ci, tap] of [
      [0, 0, 0],
      [5, 2, 100],
      [63, 1, 146],
    ])
      expect(out[(tap * 3 + ci) * 64 + co]).toBe(w[(co * 3 + ci) * 147 + tap]);
    expect(out.subarray(L.k * 64).every((v) => v === 0)).toBe(true);
  });
});

describe("shaders", () => {
  const plan = r3dPlan(fakeGraph("plain"));
  it("splits the long, small late stages only", () => {
    const splits = plan.convs.map((c) => convLayout(c).splits);
    expect(splits.slice(0, 5)).toEqual([1, 1, 1, 1, 1]);
    expect(Math.max(...splits.slice(-5))).toBeGreaterThan(1);
  });
  it("generates WGSL with every size a constant", () => {
    for (const c of plan.convs) {
      const L = convLayout(c);
      const code = convShader(c, L);
      expect(code).toContain("@compute");
      expect(code).not.toMatch(/undefined|NaN/);
      if (L.splits > 1)
        expect(reduceShader(c, L.splits)).not.toMatch(/undefined|NaN/);
    }
    expect(headShader(plan.head)).not.toMatch(/undefined|NaN/);
  });
  it("writes the inner loop out in the Firefox form, the same terms in order", () => {
    const c = plan.convs[1];
    const L = convLayout(c);
    const plain = convShader(c, L);
    const unrolled = convShader(c, L, { unrolled: true });
    expect(plain).toContain("for (var r = 0u; r < 16u; r++)");
    expect(unrolled).not.toContain("for (var r = 0u");
    // Every fma of the plain loop body appears once per step, in order.
    const fmas = (code: string) => code.match(/acc\d_\d = fma/g)!.length;
    expect(fmas(unrolled)).toBe(16 * fmas(plain));
    expect(unrolled.indexOf("As[0u *")).toBeLessThan(
      unrolled.indexOf("As[15u *"),
    );
  });
  it("detects Firefox on a Mac only", () => {
    const mac = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:160.0)";
    expect(frugalShaders(`${mac} Gecko/20100101 Firefox/160.0`)).toBe(true);
    expect(
      frugalShaders(`${mac} AppleWebKit/537.36 Chrome/157.0 Safari/537.36`),
    ).toBe(false);
    expect(
      frugalShaders(
        "Mozilla/5.0 (Windows NT 10.0; rv:160.0) Gecko/20100101 Firefox/160.0",
      ),
    ).toBe(false);
  });
});

// The real model, when its path is given (not in the repo: 133 MB), e.g.
// TWIG_MODEL=.../local_fp32c2d.onnx npx vitest --run r3d_plan
const real = process.env.TWIG_MODEL;
describe.skipIf(!real || !existsSync(real))("the published model", () => {
  it("plans", () => {
    const plan = r3dPlan(readOnnx(new Uint8Array(readFileSync(real!))));
    expect(plan.convs).toHaveLength(20);
    expect(plan.convs.every((c) => c.bias.length === c.cout)).toBe(true);
  });
});
