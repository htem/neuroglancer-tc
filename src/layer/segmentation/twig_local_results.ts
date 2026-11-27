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
 * Twig Capture results computed in this browser (Twig Selector -> in-browser
 * inference), kept on the layer so they outlive the Twig Capture tab, which
 * is disposed whenever the side panel hides.
 *
 * Results are plain JSON in the same shape run_twigcapture_v2.py writes, so
 * the Twig Capture tab opens them exactly as it opens a fetched file. They are
 * persisted in IndexedDB (per browser, never uploaded), keyed by name; names
 * start with LOCAL_RESULT_PREFIX so a shared link that names one is not
 * fetched from the catalog.
 */

import { NullarySignal } from "#src/util/signal.js";

export const LOCAL_RESULT_PREFIX = "browser_";

export interface LocalResultJson {
  name: string;
  root_id: string;
  dataset?: string;
  timestamp?: string;
  candidates: any[];
  [key: string]: unknown;
}

const DB_NAME = "neuroglancer-twig-local";
const DB_VERSION = 1;
const RESULTS = "results";
// Shared with the inference code: an object store for anything else worth
// caching between sessions (the model bytes, per-root work).
export const CACHE = "cache";

let dbPromise: Promise<IDBDatabase | undefined> | undefined;

/** The shared IndexedDB handle, or undefined where IndexedDB is unavailable. */
export function openTwigDb(): Promise<IDBDatabase | undefined> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(RESULTS))
          db.createObjectStore(RESULTS);
        if (!db.objectStoreNames.contains(CACHE)) db.createObjectStore(CACHE);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(undefined);
    } catch {
      resolve(undefined);
    }
  });
  return dbPromise;
}

function wrap<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function idbGet<T>(
  store: string,
  key: string,
): Promise<T | undefined> {
  const db = await openTwigDb();
  if (!db) return undefined;
  try {
    return await wrap(db.transaction(store).objectStore(store).get(key));
  } catch {
    return undefined;
  }
}

export async function idbPut(store: string, key: string, value: unknown) {
  const db = await openTwigDb();
  if (!db) return;
  try {
    await wrap(
      db.transaction(store, "readwrite").objectStore(store).put(value, key),
    );
  } catch {
    // quota or private mode: the result still lives in memory
  }
}

export async function idbDelete(store: string, key: string) {
  const db = await openTwigDb();
  if (!db) return;
  try {
    await wrap(
      db.transaction(store, "readwrite").objectStore(store).delete(key),
    );
  } catch {
    // ignore
  }
}

async function idbAll<T>(store: string): Promise<T[]> {
  const db = await openTwigDb();
  if (!db) return [];
  try {
    return await wrap(db.transaction(store).objectStore(store).getAll());
  } catch {
    return [];
  }
}

export class LocalTwigResults {
  /** Every result, oldest first. */
  results: LocalResultJson[] = [];
  /** Fires when a result is added, replaced or removed. */
  changed = new NullarySignal();
  /** The name the Twig Capture tab should show next, set by add(). */
  pendingOpen: string | undefined;
  /**
   * Several result names the Twig Capture tab should open together as one
   * combined list (stage and merge across neurons), set by the Twig Selector.
   */
  pendingCombined: string[] | undefined;
  ready: Promise<void>;

  constructor() {
    this.ready = idbAll<LocalResultJson>(RESULTS).then((all) => {
      for (const r of all) {
        if (!this.results.some((x) => x.name === r.name)) this.results.push(r);
      }
      this.results.sort((a, b) =>
        (a.timestamp ?? "").localeCompare(b.timestamp ?? ""),
      );
      if (all.length) this.changed.dispatch();
    });
  }

  get(name: string) {
    return this.results.find((r) => r.name === name);
  }

  /** Latest result for a root in a dataset, if any. */
  forRoot(rootId: string, dataset?: string) {
    for (let i = this.results.length - 1; i >= 0; --i) {
      const r = this.results[i];
      if (
        r.root_id === rootId &&
        (dataset === undefined || r.dataset === dataset)
      )
        return r;
    }
    return undefined;
  }

  add(result: LocalResultJson, open = true) {
    const i = this.results.findIndex((r) => r.name === result.name);
    if (i >= 0) this.results[i] = result;
    else this.results.push(result);
    if (open) this.pendingOpen = result.name;
    void idbPut(RESULTS, result.name, result);
    this.changed.dispatch();
  }

  remove(name: string) {
    this.results = this.results.filter((r) => r.name !== name);
    void idbDelete(RESULTS, name);
    this.changed.dispatch();
  }
}
