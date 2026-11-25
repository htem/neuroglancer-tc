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
 * The merge classifier, run on the reviewer's machine.
 *
 * Three runtimes, chosen by `Backend`:
 * - **webgpu** (and auto, where the browser offers WebGPU): our own WebGPU
 *   network (gpu/, ported from Philipp Schlegel's Coda), which runs the r3d_18
 *   straight from the ONNX file's weights about 2-3x faster than ONNX
 *   Runtime's WebGPU path (BRIEF.md, "The model"). The first time a model is
 *   opened on a GPU it is checked against ONNX Runtime on two patches; if
 *   they disagree by more than PARITY_MAX_DP (or the network cannot be
 *   built), ONNX Runtime on WebGPU is used instead, and the verdict is kept
 *   per model and adapter so later opens skip the check.
 * - **webgpu-ort**: ONNX Runtime Web on WebGPU, the previous path.
 * - **wasm**: ONNX Runtime Web on the CPU, in a worker where the browser
 *   allows it (see cpu_worker.ts), so the viewer stays responsive.
 *
 * ONNX Runtime is loaded on first use from jsDelivr (pinned version) rather
 * than bundled, so the viewer build does not carry ~20 MB of wasm for people
 * who never run inference. The model bytes are cached in IndexedDB by URL,
 * so they are downloaded once per browser.
 */

import { CpuWorker } from "#src/layer/segmentation/twig_inference/cpu_worker.js";
import { logEvent } from "#src/layer/segmentation/twig_inference/diagnostics.js";
import { readOnnx } from "#src/layer/segmentation/twig_inference/gpu/onnx_proto.js";
import { R3dNetwork } from "#src/layer/segmentation/twig_inference/gpu/r3d_network.js";
import { r3dPlan } from "#src/layer/segmentation/twig_inference/gpu/r3d_plan.js";
import { ORT_VERSION } from "#src/layer/segmentation/twig_inference/ort_version.js";
import { PATCH_VOXELS } from "#src/layer/segmentation/twig_inference/volume_ops.js";
import {
  CACHE,
  idbGet,
  idbPut,
} from "#src/layer/segmentation/twig_local_results.js";

const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.js`;

export type Backend = "auto" | "webgpu" | "webgpu-ort" | "wasm";

/** Largest |p| difference from ONNX Runtime the WebGPU network may show. */
export const PARITY_MAX_DP = 1e-4;

let ortPromise: Promise<any> | undefined;

function loadOrt(): Promise<any> {
  const w = window as any;
  if (w.ort) return Promise.resolve(w.ort);
  if (!ortPromise) {
    ortPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = ORT_URL;
      s.crossOrigin = "anonymous";
      s.onload = () => resolve(w.ort);
      s.onerror = () => {
        ortPromise = undefined;
        reject(new Error(`could not load ONNX Runtime Web from ${ORT_URL}`));
      };
      document.head.appendChild(s);
    });
  }
  return ortPromise;
}

export async function webgpuAvailable(): Promise<boolean> {
  try {
    const gpu = (navigator as any).gpu;
    if (!gpu) return false;
    return !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

/** Model bytes from the IndexedDB cache, else downloaded and cached. */
export async function getModelBytes(
  url: string,
  onProgress?: (msg: string) => void,
): Promise<ArrayBuffer> {
  const key = `model:${url}`;
  const cached = await idbGet<ArrayBuffer>(CACHE, key);
  if (cached) return cached;
  onProgress?.("downloading model (once per browser)…");
  const r = await fetch(url);
  if (!r.ok) throw new Error(`model ${url}: HTTP ${r.status}`);
  const buf = await r.arrayBuffer();
  await idbPut(CACHE, key, buf);
  return buf;
}

/** Store model bytes chosen from a local file, under a `file:` key. */
export async function putLocalModel(file: File): Promise<string> {
  const url = `file:${file.name}:${file.size}`;
  await idbPut(CACHE, `model:${url}`, await file.arrayBuffer());
  return url;
}

/** n patches packed in x, as views. */
function split(x: Float32Array, n: number): Float32Array[] {
  const per = 3 * PATCH_VOXELS;
  return Array.from({ length: n }, (_, i) =>
    x.subarray(i * per, (i + 1) * per),
  );
}

/** softmax(logits)[1] for n patches' logit pairs. */
function probs(logits: Float32Array, n: number, into = new Float32Array(n)) {
  for (let j = 0; j < n; ++j) {
    const a = logits[2 * j];
    const b = logits[2 * j + 1];
    const mx = Math.max(a, b);
    const ea = Math.exp(a - mx);
    const eb = Math.exp(b - mx);
    into[j] = eb / (ea + eb);
  }
  return into;
}

/**
 * Two deterministic test patches, like make_bench_assets.py's reference
 * patches: noisy EM and two masks (a straight boundary, then a disc in a
 * ring), so the check exercises both masks, the EM and every padded edge.
 */
export function parityPatches(): Float32Array {
  const [D, H, W] = [18, 128, 128];
  const v = D * H * W;
  const x = new Float32Array(2 * 3 * v);
  let seed = 12345;
  const rnd = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let p = 0; p < 2; ++p) {
    const o = p * 3 * v;
    for (let i = 0; i < v; ++i) {
      // ~N(0, 0.4) clipped to [-1, 1], from the sum of four uniforms.
      const g = (rnd() + rnd() + rnd() + rnd() - 2) * 0.4 * Math.sqrt(3);
      x[o + i] = Math.max(-1, Math.min(1, g));
      const yy = Math.floor(i / W) % H;
      const xx = i % W;
      const r = Math.hypot(xx - 64, yy - 64);
      x[o + v + i] = p === 0 ? +(xx < 64) : +(r < 30);
      x[o + 2 * v + i] = p === 0 ? +(xx >= 66) : +(r > 33);
    }
  }
  return x;
}

export class MergeModel {
  private constructor(
    private ort: any,
    private session: any,
    readonly backend: "webgpu" | "wasm",
    /** The WebGPU network, when it runs the model instead of ONNX Runtime. */
    readonly gpu?: R3dNetwork,
    /** The CPU worker, when ONNX Runtime runs there. */
    private worker?: CpuWorker,
  ) {}

  /** What scores the patches, for the result's benchmark block. */
  get runtime() {
    if (this.gpu)
      return `twig-capture webgpu r3d (Coda-derived${this.gpu.frugal ? ", unrolled" : ""}) on ${this.gpu.adapter}`;
    return `onnxruntime-web ${ORT_VERSION} (${this.backend}${this.worker ? ", worker" : ""})`;
  }

  static async create(
    modelUrl: string,
    backend: Backend,
    onProgress?: (msg: string) => void,
    /** Force the Firefox shader form on or off (measuring only). */
    frugal?: boolean,
  ): Promise<MergeModel> {
    const bytes = await getModelBytes(modelUrl, onProgress);
    const useGpu =
      backend === "webgpu" ||
      backend === "webgpu-ort" ||
      (backend === "auto" && (await webgpuAvailable()));
    if (useGpu && backend !== "webgpu-ort") {
      const gpu = await openGpuNetwork(modelUrl, bytes, onProgress, frugal);
      if (gpu) return new MergeModel(undefined, undefined, "webgpu", gpu);
    }
    if (!useGpu) {
      const worker = await CpuWorker.open(bytes, onProgress);
      if (worker)
        return new MergeModel(undefined, undefined, "wasm", undefined, worker);
    }
    const ort = await loadOrt();
    const { session, ep } = await ortSession(
      ort,
      bytes,
      useGpu,
      backend,
      onProgress,
    );
    return new MergeModel(ort, session, ep);
  }

  /**
   * p(merge) for n patches packed in x ([n, 3, 18, 128, 128] float32).
   */
  async score(x: Float32Array, n: number): Promise<Float32Array> {
    return this.scorePatches(split(x, n));
  }

  /** p(merge) for each patch ([3, 18, 128, 128] float32 each). */
  async scorePatches(patches: Float32Array[]): Promise<Float32Array> {
    const n = patches.length;
    if (this.gpu) return probs(await this.gpu.logitsFor(patches), n);
    if (this.worker) return probs(await this.worker.logits(patches), n);
    return probs(await ortLogits(this.ort, this.session, patches), n);
  }

  async release() {
    this.gpu?.destroy();
    this.worker?.close();
    await this.session?.release?.();
  }
}

async function ortSession(
  ort: any,
  bytes: ArrayBuffer,
  useGpu: boolean,
  backend: Backend,
  onProgress?: (msg: string) => void,
): Promise<{ session: any; ep: "webgpu" | "wasm" }> {
  const start = async (ep: "webgpu" | "wasm") => {
    if (ep === "wasm") {
      // Threads need crossOriginIsolated (COOP/COEP); without it ORT runs
      // single-threaded.
      ort.env.wasm.numThreads = self.crossOriginIsolated
        ? Math.min(navigator.hardwareConcurrency || 4, 8)
        : 1;
    }
    onProgress?.(
      `starting model on ${ep === "webgpu" ? "GPU (WebGPU)" : "CPU (WASM)"}…`,
    );
    logEvent("info", `starting ONNX Runtime ${ORT_VERSION} on ${ep}`);
    // Graph optimisation is safe for the Conv2d model (no Pad feeds a Conv).
    return ort.InferenceSession.create(bytes, {
      executionProviders: [ep],
      graphOptimizationLevel: "all",
    });
  };
  let ep: "webgpu" | "wasm" = useGpu ? "webgpu" : "wasm";
  let session;
  try {
    session = await start(ep);
  } catch (e) {
    // A GPU that advertises WebGPU but cannot run the model (old drivers,
    // missing features, some Windows/Intel setups) must not end the run
    // when the user asked for "auto": fall back to the CPU and say so.
    if (ep !== "webgpu" || backend !== "auto") throw e;
    logEvent("warn", "WebGPU session failed; falling back to CPU (WASM)", e);
    ep = "wasm";
    session = await start(ep);
  }
  logEvent(
    "info",
    `model ready on ${ep} (${(bytes.byteLength / 1e6).toFixed(1)} MB)`,
  );
  return { session, ep };
}

/**
 * Logits for each patch through an ONNX Runtime session, one at a time
 * (batch 1 measured fastest on Apple WebGPU; the CPU gains nothing).
 */
async function ortLogits(
  ort: any,
  session: any,
  patches: Float32Array[],
): Promise<Float32Array> {
  const out = new Float32Array(2 * patches.length);
  for (let i = 0; i < patches.length; ++i) {
    const t = new ort.Tensor("float32", patches[i], [1, 3, 18, 128, 128]);
    const res = await session.run({ x_local: t });
    out.set((res.logits.data as Float32Array).subarray(0, 2), 2 * i);
    t.dispose?.();
  }
  return out;
}

/**
 * The WebGPU network for this model, checked against ONNX Runtime the first
 * time on this adapter; undefined (with the reason logged) if it cannot be
 * built or does not match, so the caller falls back to ONNX Runtime.
 */
async function openGpuNetwork(
  modelUrl: string,
  bytes: ArrayBuffer,
  onProgress?: (msg: string) => void,
  frugal?: boolean,
): Promise<R3dNetwork | undefined> {
  let net: R3dNetwork | undefined;
  try {
    onProgress?.("starting model on GPU (WebGPU)…");
    net = await R3dNetwork.build(r3dPlan(readOnnx(new Uint8Array(bytes))), {
      frugal,
    });
    const key =
      `gpuParity:v1:${modelUrl}:${net.adapter}` + (net.frugal ? ":frugal" : "");
    let verdict = await idbGet<{ ok: boolean; maxDp: number }>(CACHE, key);
    if (!verdict) {
      onProgress?.("checking the GPU model against ONNX Runtime (once)…");
      const x = parityPatches();
      const ort = await loadOrt();
      const { session, ep } = await ortSession(ort, bytes, true, "webgpu");
      let ref: Float32Array;
      try {
        ref = probs(await ortLogits(ort, session, split(x, 2)), 2);
      } finally {
        await session.release?.();
      }
      const got = probs(await net.logitsFor(split(x, 2)), 2);
      const maxDp = Math.max(...[0, 1].map((i) => Math.abs(got[i] - ref[i])));
      verdict = { ok: maxDp <= PARITY_MAX_DP && Number.isFinite(maxDp), maxDp };
      logEvent(
        verdict.ok ? "info" : "warn",
        `WebGPU network vs ONNX Runtime (${ep}): max|dp| ${maxDp.toExponential(2)}` +
          ` (p ${[...got].map((v) => v.toFixed(6))} vs ${[...ref].map((v) => v.toFixed(6))})`,
      );
      await idbPut(CACHE, key, verdict);
    }
    if (!verdict.ok) {
      logEvent(
        "warn",
        `WebGPU network disagrees with ONNX Runtime on ${net.adapter} (max|dp| ${verdict.maxDp}); using ONNX Runtime`,
      );
      net.destroy();
      return undefined;
    }
    logEvent(
      "info",
      `model ready on WebGPU network (${net.adapter}, ${(net.bytes / 1e6).toFixed(0)} MB GPU memory)`,
    );
    return net;
  } catch (e) {
    net?.destroy();
    logEvent("warn", "WebGPU network unavailable; using ONNX Runtime", e);
    return undefined;
  }
}

/**
 * For measuring from the console (BRIEF.md, "Measuring"):
 * `await twigCaptureModel.open(url, "webgpu")`, then `.score(x, n)`, or for
 * the WebGPU network `.gpu.profile(patch)`.
 */
(globalThis as any).twigCaptureModel = {
  open: (url: string, backend: Backend, frugal?: boolean) =>
    MergeModel.create(url, backend, undefined, frugal),
  parityPatches,
};
