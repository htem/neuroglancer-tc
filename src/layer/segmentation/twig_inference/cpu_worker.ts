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
 * ONNX Runtime on the CPU (WASM) in a worker, so the viewer stays responsive
 * while a patch takes ~3 s.
 *
 * ORT's own worker proxy (env.wasm.proxy) cannot start from the cross-origin
 * CDN script it is loaded from. This worker is made from a Blob instead, which
 * has the page's origin, and loads the same pinned ORT build with
 * importScripts (allowed cross-origin in a classic worker), pointing ORT at
 * the CDN for its wasm. Still single-threaded: threads need the page to be
 * cross-origin isolated (COOP/COEP), which would break CAVE's login popup.
 *
 * If the worker cannot be made (a Content-Security-Policy that forbids blob
 * workers, say), `open` returns undefined and ORT runs on the main thread as
 * before.
 */

import { logEvent } from "#src/layer/segmentation/twig_inference/diagnostics.js";
import { ORT_VERSION } from "#src/layer/segmentation/twig_inference/ort_version.js";

const DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const WORKER_SOURCE = `
importScripts(${JSON.stringify(DIST + "ort.webgpu.min.js")});
ort.env.wasm.wasmPaths = ${JSON.stringify(DIST)};
ort.env.wasm.numThreads = 1;
let session;
onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "open") {
      session = await ort.InferenceSession.create(m.bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      postMessage({ id: m.id, ok: true });
    } else if (m.type === "run") {
      const per = 3 * 18 * 128 * 128;
      const out = new Float32Array(2 * m.n);
      for (let i = 0; i < m.n; ++i) {
        const t = new ort.Tensor("float32", m.x.subarray(i * per, (i + 1) * per), [1, 3, 18, 128, 128]);
        const r = await session.run({ x_local: t });
        out.set(r.logits.data.subarray(0, 2), 2 * i);
      }
      postMessage({ id: m.id, ok: true, logits: out }, [out.buffer]);
    }
  } catch (err) {
    postMessage({ id: m.id, ok: false, error: String((err && err.message) || err) });
  }
};
`;

export class CpuWorker {
  private next = 0;
  private waiting = new Map<
    number,
    { resolve: (v: any) => void; reject: (e: Error) => void }
  >();

  private constructor(private worker: Worker) {
    worker.onmessage = (e) => {
      const w = this.waiting.get(e.data.id);
      if (!w) return;
      this.waiting.delete(e.data.id);
      if (e.data.ok) w.resolve(e.data);
      else w.reject(new Error(e.data.error));
    };
    worker.onerror = (e) => {
      const err = new Error(`CPU worker: ${e.message || "failed"}`);
      for (const w of this.waiting.values()) w.reject(err);
      this.waiting.clear();
    };
  }

  private call(msg: any, transfer: Transferable[] = []): Promise<any> {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.worker.postMessage({ ...msg, id }, transfer);
    });
  }

  /** A worker with the model loaded, or undefined if one cannot be made. */
  static async open(
    bytes: ArrayBuffer,
    onProgress?: (msg: string) => void,
  ): Promise<CpuWorker | undefined> {
    let w: CpuWorker | undefined;
    try {
      onProgress?.("starting model on CPU (WASM, in a worker)…");
      const url = URL.createObjectURL(
        new Blob([WORKER_SOURCE], { type: "text/javascript" }),
      );
      w = new CpuWorker(new Worker(url));
      URL.revokeObjectURL(url);
      // A copy: the bytes stay cached for the next open.
      const copy = bytes.slice(0);
      await w.call({ type: "open", bytes: copy }, [copy]);
      logEvent(
        "info",
        `model ready on CPU (WASM) in a worker, ORT ${ORT_VERSION}`,
      );
      return w;
    } catch (e) {
      w?.close();
      logEvent(
        "warn",
        "CPU worker unavailable; running ONNX Runtime on the page",
        e,
      );
      return undefined;
    }
  }

  /** Logits (2 per patch) for these patches. */
  async logits(patches: Float32Array[]): Promise<Float32Array> {
    const per = 3 * 18 * 128 * 128;
    // One packed copy, transferred: the caller keeps its patches.
    const x = new Float32Array(patches.length * per);
    patches.forEach((p, i) => x.set(p, i * per));
    const r = await this.call({ type: "run", x, n: patches.length }, [
      x.buffer,
    ]);
    return r.logits as Float32Array;
  }

  close() {
    this.worker.terminate();
    for (const w of this.waiting.values()) w.reject(new Error("closed"));
    this.waiting.clear();
  }
}
