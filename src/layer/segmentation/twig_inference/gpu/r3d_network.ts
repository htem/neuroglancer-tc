/**
 * @license
 * Ported from Coda by Philipp Schlegel (https://github.com/navis-org/coda,
 * src/packs/seam/gpuAssessor), with changes for the Twig Capture model.
 *
 * MIT License
 *
 * Copyright (c) 2026 NAVis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * The Twig Capture model on WebGPU without ONNX Runtime: one compute pass a
 * patch. `r3d_plan.ts` says what to compute and `r3d_shaders.ts` how; this
 * owns the buffers, the pipelines and the submits.
 *
 * Coda's network.ts (Philipp Schlegel), cut down: everything is allocated
 * when the network is built and reused for every patch, so scoring a patch is
 * one 3.5 MB upload, ~30 dispatches and an 8-byte read back.
 *
 * **Submitted in turn, read back together.** The buffers are shared between
 * patches, so patches are submitted one after another, but a patch's answer
 * is not awaited before the next is submitted: the queue runs submitted work
 * (and `writeBuffer`) in order, so patch n+1's upload cannot land before
 * patch n has been computed, and each patch copies its logits to a read-back
 * buffer of its own. Up to `IN_FLIGHT` patches are outstanding, which spares
 * every patch the round trip of a wait (Firefox's is ~100 ms, Coda measured).
 * The answers are the same to the bit either way.
 */

import type { R3dPlan } from "#src/layer/segmentation/twig_inference/gpu/r3d_plan.js";
import type {
  Precision,
  Tile,
} from "#src/layer/segmentation/twig_inference/gpu/r3d_shaders.js";
import {
  convLayout,
  convShader,
  convWeightFloats,
  convWeightLayout,
  FLAT,
  DEFAULT_TILE,
  F32,
  frugalShaders,
  headShader,
  toHalfBits,
  headWeights,
  reduceShader,
  voxelsOf,
} from "#src/layer/segmentation/twig_inference/gpu/r3d_shaders.js";

// GPUBufferUsage / GPUMapMode, as numbers (no @webgpu/types in this repo).
const MAP_READ = 0x1;
const COPY_SRC = 0x4;
const COPY_DST = 0x8;
const STORAGE_USE = 0x80;
const MAP_MODE_READ = 0x1;
const STORAGE = STORAGE_USE | COPY_DST | COPY_SRC;

/** Patches outstanding at once (Coda: nothing more was bought past 12). */
export const IN_FLIGHT = 12;

interface Dispatch {
  readonly name: string;
  readonly pipeline: any;
  readonly group: any;
  readonly groups: readonly [number, number, number];
}

export interface StepTiming {
  readonly name: string;
  readonly ms: number;
}

/** A high-performance WebGPU device with the adapter's buffer limits. */
export async function r3dDevice(
  features: string[] = [],
): Promise<{ device: any; adapter: string }> {
  const gpu = (navigator as any).gpu;
  if (!gpu) throw new Error("this browser offers no WebGPU");
  const adapter = await gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("WebGPU offered no adapter");
  const missing = features.filter((f) => !adapter.features.has(f));
  if (missing.length)
    throw new Error(`this GPU lacks WebGPU feature(s) ${missing.join(", ")}`);
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  const info = adapter.info ?? {};
  const name =
    [info.vendor, info.architecture, info.description]
      .filter(Boolean)
      .join(" ") || "unknown adapter";
  return { device, adapter: name };
}

/** A built network: GPU objects only, so nothing of the plan or the file stays reachable. */
export class R3dNetwork {
  private queue: Promise<unknown> = Promise.resolve();
  private destroyed = false;
  private lost: string | undefined;
  private readonly spare: any[] = [];

  constructor(
    private readonly device: any,
    private readonly dispatches: readonly Dispatch[],
    private readonly owned: readonly any[],
    private readonly input: any,
    private readonly logits: any,
    /** Floats a patch holds. */
    readonly patchFloats: number,
    /** Multiply-adds x 2 per patch. */
    readonly flops: number,
    readonly adapter: string,
  ) {
    this.bytes = owned.reduce((n, b) => n + b.size, 0);
    void device.lost.then((info: any) => {
      this.lost = info.message || info.reason || "device lost";
    });
  }

  /** GPU memory held, in bytes: weights and every intermediate. */
  readonly bytes: number;
  /** Built in the Firefox form (`frugalShaders`). */
  frugal = false;
  precision: Precision = F32;
  tile: Tile = DEFAULT_TILE;

  /**
   * `frugal`: the Firefox form of the shaders (unrolled inner loop), by
   * default where `frugalShaders` says so. The logits are the same either way.
   */
  static async build(
    plan: R3dPlan,
    opts: {
      splitTarget?: number;
      frugal?: boolean;
      precision?: Precision;
      tile?: Tile;
    } = {},
  ) {
    const precision = opts.precision ?? F32;
    const f16 = precision.store === "f16" || precision.math === "f16";
    const { device, adapter } = await r3dDevice(f16 ? ["shader-f16"] : []);
    const owned: any[] = [];
    const frugal = opts.frugal ?? frugalShaders();
    try {
      const net = await assemble(
        device,
        adapter,
        plan,
        owned,
        opts.splitTarget,
        frugal,
        precision,
        opts.tile ?? DEFAULT_TILE,
      );
      net.frugal = frugal;
      net.precision = precision;
      net.tile = opts.tile ?? DEFAULT_TILE;
      return net;
    } catch (e) {
      for (const b of owned) b.destroy();
      device.destroy();
      throw e;
    }
  }

  /**
   * Logits (2 per patch) for these patches, each [3, 18, 128, 128] float32,
   * with up to IN_FLIGHT outstanding.
   */
  async logitsFor(patches: Float32Array[]): Promise<Float32Array> {
    const out = new Float32Array(2 * patches.length);
    const pending: Promise<void>[] = [];
    for (let i = 0; i < patches.length; ++i) {
      if (pending.length >= IN_FLIGHT) await pending.shift();
      const rb = await this.serial(async () => this.submit(patches[i]));
      pending.push(
        this.read(rb).then((v) => {
          out[2 * i] = v[0];
          out[2 * i + 1] = v[1];
        }),
      );
    }
    await Promise.all(pending);
    return out;
  }

  private serial<T>(f: () => Promise<T>): Promise<T> {
    const run = this.queue.then(f, f);
    this.queue = run.catch(() => {});
    return run;
  }

  private submit(patch: Float32Array): any {
    if (this.destroyed) throw new Error("the WebGPU network has been closed");
    if (this.lost) throw new Error(`the GPU went away: ${this.lost}`);
    if (patch.length !== this.patchFloats)
      throw new Error(
        `a patch is ${this.patchFloats} floats, got ${patch.length}`,
      );
    this.device.queue.writeBuffer(
      this.input,
      0,
      patch.buffer,
      patch.byteOffset,
      patch.byteLength,
    );
    const rb =
      this.spare.pop() ??
      this.device.createBuffer({ size: 8, usage: MAP_READ | COPY_DST });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const d of this.dispatches) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.group);
      pass.dispatchWorkgroups(...d.groups);
    }
    pass.end();
    enc.copyBufferToBuffer(this.logits, 0, rb, 0, 8);
    this.device.queue.submit([enc.finish()]);
    return rb;
  }

  private async read(rb: any): Promise<[number, number]> {
    // Unmapped whatever happens: a buffer left mapped refuses later mapAsync.
    try {
      await rb.mapAsync(MAP_MODE_READ);
      const v = new Float32Array(rb.getMappedRange());
      return [v[0], v[1]];
    } finally {
      rb.unmap();
      if (this.destroyed) rb.destroy();
      else this.spare.push(rb);
    }
  }

  /**
   * Each step timed alone, recorded `repeats` times in one submit and
   * divided, so the round trip of a wait is spread thin. For measuring only.
   */
  profile(patch: Float32Array, repeats = 30): Promise<StepTiming[]> {
    return this.serial(async () => {
      this.device.queue.writeBuffer(
        this.input,
        0,
        patch.buffer,
        patch.byteOffset,
        patch.byteLength,
      );
      await this.device.queue.onSubmittedWorkDone();
      const out: StepTiming[] = [];
      for (const d of this.dispatches) {
        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let r = 0; r < repeats; r++) {
          pass.setPipeline(d.pipeline);
          pass.setBindGroup(0, d.group);
          pass.dispatchWorkgroups(...d.groups);
        }
        pass.end();
        const t = performance.now();
        this.device.queue.submit([enc.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        out.push({ name: d.name, ms: (performance.now() - t) / repeats });
      }
      return out;
    });
  }

  /** Refuse new work; free the GPU once what was submitted has finished. */
  destroy(): void {
    this.destroyed = true;
    void this.queue
      .then(() => this.device.queue.onSubmittedWorkDone())
      .catch(() => {})
      .then(() => {
        for (const b of this.owned) b.destroy();
        for (const b of this.spare.splice(0)) b.destroy();
        this.device.destroy();
      });
  }
}

async function assemble(
  device: any,
  adapter: string,
  plan: R3dPlan,
  owned: any[],
  splitTarget?: number,
  frugal = false,
  precision: Precision = F32,
  tile: Tile = DEFAULT_TILE,
): Promise<R3dNetwork> {
  /** Bytes per stored activation or weight. */
  const SB = precision.store === "f16" ? 2 : 4;
  const limit = device.limits.maxStorageBufferBindingSize;
  const buffer = (bytes: number, mapped = false) => {
    if (bytes > limit)
      throw new Error(
        `a ${bytes}-byte buffer is past this GPU's ${limit}-byte binding limit`,
      );
    const b = device.createBuffer({
      size: Math.max(4, Math.ceil(bytes / 4) * 4),
      usage: STORAGE,
      mappedAtCreation: mapped,
    });
    owned.push(b);
    return b;
  };
  /** A new buffer filled through its mapped range: no staging copy. */
  const filled = (floats: number, fill: (into: Float32Array) => void) => {
    const b = buffer(floats * 4, true);
    fill(new Float32Array(b.getMappedRange(), 0, floats));
    b.unmap();
    return b;
  };
  /** A new buffer of these values as f16, through its mapped range. */
  const halves = (a: Float32Array<ArrayBufferLike>) => {
    const b = buffer(a.length * 2, true);
    const into = new Uint16Array(b.getMappedRange(), 0, a.length);
    for (let i = 0; i < a.length; ++i) into[i] = toHalfBits(a[i]);
    b.unmap();
    return b;
  };
  const upload = (a: Float32Array<ArrayBufferLike>) =>
    filled(a.length, (into) => into.set(a));

  const patchFloats = plan.input.reduce((a, b) => a * b, 1);
  const input = buffer(patchFloats * 4);
  const tensors = new Map<string, any>([["input", input]]);

  // Liveness: an activation's buffer is reused once its last reader has run.
  const lastUse = new Map<string, number>();
  plan.convs.forEach((c, i) => {
    lastUse.set(c.input, i);
    if (c.residual) lastUse.set(c.residual, i);
  });
  lastUse.set(plan.head.input, plan.convs.length);
  const free = new Map<number, any[]>();
  const take = (bytes: number) => free.get(bytes)?.pop() ?? buffer(bytes);
  const give = (name: string, b: any) => {
    if (name === "input") return;
    const list = free.get(b.size) ?? [];
    list.push(b);
    free.set(b.size, list);
  };

  const layouts = plan.convs.map((c) => convLayout(c, splitTarget, tile));
  const partialBytes = Math.max(
    4,
    ...plan.convs.map((c, i) =>
      layouts[i].splits > 1
        ? layouts[i].splits * c.cout * voxelsOf(c.outShape) * 4
        : 0,
    ),
  );
  const partial = buffer(partialBytes);

  const pipelines = new Map<string, Promise<any>>();
  const pipeline = (code: string) => {
    const had = pipelines.get(code);
    if (had) return had;
    const made: Promise<any> = device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code }),
        entryPoint: "main",
      },
    });
    pipelines.set(code, made);
    return made;
  };
  const bind = (pipe: any, buffers: any[]) =>
    device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });

  const dispatches: Dispatch[] = [];
  for (let i = 0; i < plan.convs.length; ++i) {
    const c = plan.convs[i];
    const L = layouts[i];
    const wbuf =
      SB === 4
        ? filled(convWeightFloats(c, L), (into) =>
            convWeightLayout(c, L, c.weight(), into),
          )
        : halves(convWeightLayout(c, L, c.weight()));
    const bias = upload(c.bias);
    const inp = tensors.get(c.input);
    const res = c.residual ? tensors.get(c.residual) : undefined;
    if (!inp || (c.residual && !res))
      throw new Error(`${c.name}: an input is not computed before it`);
    const out = take(c.cout * voxelsOf(c.outShape) * SB);
    tensors.set(c.output, out);
    const conv = await pipeline(
      convShader(c, L, { unrolled: frugal, precision }),
    );
    if (L.splits === 1) {
      dispatches.push({
        name: c.name,
        pipeline: conv,
        group: bind(
          conv,
          res ? [inp, wbuf, out, bias, res] : [inp, wbuf, out, bias],
        ),
        groups: L.groups,
      });
    } else {
      dispatches.push({
        name: `${c.name} (${L.splits} splits)`,
        pipeline: conv,
        group: bind(conv, [inp, wbuf, partial]),
        groups: L.groups,
      });
      const red = await pipeline(reduceShader(c, L.splits, precision));
      dispatches.push({
        name: `${c.name} sum`,
        pipeline: red,
        group: bind(
          red,
          res ? [partial, out, bias, res] : [partial, out, bias],
        ),
        groups: [Math.ceil((c.cout * voxelsOf(c.outShape)) / FLAT), 1, 1],
      });
    }
    // Free what this step read for the last time (never its own output).
    for (const name of [c.input, c.residual]) {
      if (name && lastUse.get(name) === i && name !== c.output) {
        give(name, tensors.get(name));
        tensors.delete(name);
      }
    }
  }
  const head = plan.head;
  const headIn = tensors.get(head.input);
  const logits = buffer(8);
  const hp = await pipeline(headShader(head, precision));
  dispatches.push({
    name: "head",
    pipeline: hp,
    group: bind(hp, [headIn, upload(headWeights(head)), logits]),
    groups: [1, 1, 1],
  });
  return new R3dNetwork(
    device,
    dispatches,
    owned,
    input,
    logits,
    patchFloats,
    plan.flops,
    adapter,
  );
}
