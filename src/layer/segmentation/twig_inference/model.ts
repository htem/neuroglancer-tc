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
 * The merge classifier, run with ONNX Runtime Web on the reviewer's machine.
 *
 * Backend: WebGPU where the browser offers an adapter, else WASM (CPU), so it
 * works without a GPU (slower: see BRIEF.md for measured times). The runtime
 * is loaded on first use from jsDelivr (pinned version) rather than bundled,
 * so the viewer build does not carry ~20 MB of wasm for people who never run
 * inference. The model bytes are cached in IndexedDB by URL, so they are
 * downloaded once per browser.
 */

import { logEvent } from "#src/layer/segmentation/twig_inference/diagnostics.js";
import { ORT_VERSION } from "#src/layer/segmentation/twig_inference/ort_version.js";
import { PATCH_VOXELS } from "#src/layer/segmentation/twig_inference/volume_ops.js";
import {
  CACHE,
  idbGet,
  idbPut,
} from "#src/layer/segmentation/twig_local_results.js";

const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.js`;

export type Backend = "auto" | "webgpu" | "wasm";

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

export class MergeModel {
  private constructor(
    private ort: any,
    private session: any,
    readonly backend: "webgpu" | "wasm",
    readonly batchSize: number,
  ) {}

  static async create(
    modelUrl: string,
    backend: Backend,
    onProgress?: (msg: string) => void,
  ): Promise<MergeModel> {
    const ort = await loadOrt();
    const bytes = await getModelBytes(modelUrl, onProgress);
    const useGpu =
      backend === "webgpu" || (backend === "auto" && (await webgpuAvailable()));
    const start = async (ep: "webgpu" | "wasm") => {
      if (ep === "wasm") {
        // Threads need crossOriginIsolated (COOP/COEP); without it ORT runs
        // single-threaded.
        ort.env.wasm.numThreads = self.crossOriginIsolated
          ? Math.min(navigator.hardwareConcurrency || 4, 8)
          : 1;
        // ~3 s per patch on one core, on the main thread. ORT's worker proxy
        // (env.wasm.proxy) would keep the viewer responsive, but it cannot
        // start a worker from the cross-origin CDN script; bundling ORT would
        // fix that.
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
    // Batch 1 measured fastest on Apple WebGPU; CPU gains nothing from batching.
    return new MergeModel(ort, session, ep, 1);
  }

  /**
   * p(merge) for n patches packed in x ([n, 3, 18, 128, 128] float32).
   */
  async score(x: Float32Array, n: number): Promise<Float32Array> {
    const per = 3 * PATCH_VOXELS;
    return this.scorePatches(
      Array.from({ length: n }, (_, i) => x.subarray(i * per, (i + 1) * per)),
    );
  }

  /** p(merge) for each patch ([3, 18, 128, 128] float32 each), one at a time. */
  async scorePatches(patches: Float32Array[]): Promise<Float32Array> {
    const p = new Float32Array(patches.length);
    for (let i = 0; i < patches.length; ++i) {
      const t = new this.ort.Tensor(
        "float32",
        patches[i],
        [1, 3, 18, 128, 128],
      );
      const out = await this.session.run({ x_local: t });
      const logits = out.logits.data as Float32Array;
      const a = logits[0];
      const b = logits[1];
      const mx = Math.max(a, b);
      const ea = Math.exp(a - mx);
      const eb = Math.exp(b - mx);
      p[i] = eb / (ea + eb);
      t.dispose?.();
    }
    return p;
  }

  /** What scores the patches, for the result's benchmark block. */
  get runtime() {
    return `onnxruntime-web ${ORT_VERSION} (${this.backend})`;
  }

  async release() {
    await this.session.release?.();
  }
}
