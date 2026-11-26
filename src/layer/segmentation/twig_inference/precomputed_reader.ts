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
 * A small, self-contained reader for neuroglancer-precomputed volumes, used by
 * in-browser Twig Capture to cut out exact voxel boxes (neuroglancer's own
 * chunk machinery serves the renderer, not arbitrary cutouts).
 *
 * Supports what BANC needs: sharded (neuroglancer_uint64_sharded_v1, identity
 * hash) and unsharded scales; compressed_segmentation uint64 (returned as
 * small label indices into a shared id table), and jpeg / raw uint8.
 * Decoded chunks are kept in an LRU so neighbouring regions share fetches.
 */

import { encodeZIndexCompressed } from "#src/util/zorder.js";

export interface ScaleInfo {
  key: string;
  resolution: [number, number, number];
  size: [number, number, number];
  voxel_offset: [number, number, number];
  chunk_sizes: [number, number, number][];
  encoding: string;
  compressed_segmentation_block_size?: [number, number, number];
  sharding?: {
    "@type": string;
    preshift_bits: number;
    hash: string;
    minishard_bits: number;
    shard_bits: number;
    minishard_index_encoding?: string;
    data_encoding?: string;
  };
}

export interface VolumeInfo {
  data_type: string;
  num_channels: number;
  scales: ScaleInfo[];
}

/**
 * Maps a path under a volume URL (precomputed://gs://b/p, gs://, s3:// or
 * https://) to a fetchable https URL. gs:// goes through the GCS JSON API,
 * as neuroglancer's own gs:// store does: unlike storage.googleapis.com/<b>/
 * it always answers CORS for public objects (Aedes' EM bucket has no CORS
 * policy of its own), and it honours Range.
 */
export function objectUrlFn(url: string): (path: string) => string {
  const u = url.replace(/^precomputed:\/\//, "").replace(/\/+$/, "");
  if (u.startsWith("gs://")) {
    const [bucket, ...rest] = u.slice(5).split("/");
    const prefix = rest.join("/");
    return (path) =>
      `https://storage.googleapis.com/storage/v1/b/${bucket}/o/${encodeURIComponent(
        prefix ? `${prefix}/${path}` : path,
      )}?alt=media`;
  }
  if (u.startsWith("s3://")) {
    const [bucket, ...rest] = u.slice(5).split("/");
    const base = `https://${bucket}.s3.amazonaws.com/${rest.join("/")}`;
    return (path) => `${base}/${path}`;
  }
  return (path) => `${u}/${path}`;
}

async function gunzip(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Resolves when the browser is back online (or at once if it is online). */
function whenOnline(signal?: AbortSignal): Promise<void> {
  if (typeof navigator === "undefined" || navigator.onLine !== false) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const done = () => {
      window.removeEventListener("online", done);
      signal?.removeEventListener("abort", stop);
      resolve();
    };
    const stop = () => {
      window.removeEventListener("online", done);
      reject(signal!.reason ?? new DOMException("Aborted", "AbortError"));
    };
    window.addEventListener("online", done);
    signal?.addEventListener("abort", stop, { once: true });
  });
}

/**
 * fetch, retried through network blips. A dropped connection makes fetch
 * throw "Failed to fetch" at once, so without this one blip failed every
 * region queued behind it (a beta tester lost 60 regions in 0.3 s). Network
 * errors, 429 and 5xx are retried with backoff (1, 2, 4, 8, 16 s); while the
 * browser reports itself offline it waits for the connection instead.
 * Other HTTP statuses are returned to the caller as they are.
 */
/**
 * Equivalent hostnames for public S3 buckets served over HTTP/1.1, where
 * Chrome opens at most 6 connections per host. Requests are spread over the
 * bucket's virtual-hosted, path-style and regional endpoints (verified to
 * serve identical bytes with `Access-Control-Allow-Origin: *`), tripling
 * the connections. GCS is HTTP/2 and needs none.
 */
const MIRRORS: Record<string, string[]> = {
  "https://bossdb-open-data.s3.amazonaws.com/": [
    "https://bossdb-open-data.s3.amazonaws.com/",
    "https://s3.amazonaws.com/bossdb-open-data/",
    "https://s3.us-east-1.amazonaws.com/bossdb-open-data/",
  ],
};
let mirrorTurn = 0;
let mirrorsOn = true;
/** One of `url`'s equivalent hosts, in turn; `url` itself if it has none. */
export function mirrored(url: string): string {
  if (!mirrorsOn) return url;
  for (const [base, alts] of Object.entries(MIRRORS))
    if (url.startsWith(base))
      return alts[mirrorTurn++ % alts.length] + url.slice(base.length);
  return url;
}

export async function fetchRetrying(
  url: string,
  init: RequestInit = {},
  signal?: AbortSignal,
  attempts = 6,
): Promise<Response> {
  for (let attempt = 0; ; ++attempt) {
    await whenOnline(signal);
    try {
      const r = await fetch(mirrored(url), { ...init, signal });
      if ((r.status === 429 || r.status >= 500) && attempt < attempts - 1) {
        throw new Error(`HTTP ${r.status} for ${url}`);
      }
      return r;
    } catch (e) {
      if (signal?.aborted || attempt >= attempts - 1) throw e;
      await new Promise((res) => setTimeout(res, 1000 * 2 ** attempt));
    }
  }
}

/**
 * Per-host fetch statistics for the run timeline (pipeline.ts): requests,
 * bytes, and requests in flight (fetch through the body), with their time
 * integral for a mean.
 */
export interface HostStats {
  requests: number;
  bytes: number;
  inflight: number;
  maxInflight: number;
  /** Integral of `inflight` over time, in request-ms. */
  weighted: number;
  last: number;
}
export const hostStats = new Map<string, HostStats>();

async function tracked(
  url: string,
  read: () => Promise<Uint8Array>,
): Promise<Uint8Array> {
  const host = new URL(url, globalThis.location?.href).host;
  let s = hostStats.get(host);
  if (!s) {
    s = {
      requests: 0,
      bytes: 0,
      inflight: 0,
      maxInflight: 0,
      weighted: 0,
      last: performance.now(),
    };
    hostStats.set(host, s);
  }
  const step = () => {
    const t = performance.now();
    s!.weighted += s!.inflight * (t - s!.last);
    s!.last = t;
  };
  step();
  s.inflight++;
  s.requests++;
  if (s.inflight > s.maxInflight) s.maxInflight = s.inflight;
  try {
    const b = await read();
    s.bytes += b.byteLength;
    return b;
  } finally {
    step();
    s.inflight--;
  }
}

async function fetchRange(
  url: string,
  start: number,
  end: number,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  if (end <= start) return new Uint8Array(0);
  return tracked(url, () => fetchRangeUntracked(url, start, end, signal));
}

async function fetchRangeUntracked(
  url: string,
  start: number,
  end: number,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const r = await fetchRetrying(
    url,
    { headers: { Range: `bytes=${start}-${end - 1}` } },
    signal,
  );
  if (r.status === 404) return new Uint8Array(0);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
  const buf = new Uint8Array(await r.arrayBuffer());
  // A server that ignores Range returns the whole object.
  return r.status === 200 && buf.length > end - start
    ? buf.subarray(start, end)
    : buf;
}

class LRU<V> {
  private map = new Map<string, V>();
  constructor(private capacity: number) {}
  get(k: string) {
    const v = this.map.get(k);
    if (v !== undefined) {
      this.map.delete(k);
      this.map.set(k, v);
    }
    return v;
  }
  clear() {
    this.map.clear();
  }
  set(k: string, v: V) {
    this.map.delete(k);
    this.map.set(k, v);
    while (this.map.size > this.capacity) {
      this.map.delete(this.map.keys().next().value!);
    }
  }
}

/**
 * A Map without the engine's size limit. A JS Map holds at most 2^24
 * (16,777,216) entries and then throws "RangeError: Map maximum size
 * exceeded"; one long run (a 242-L2 Aedes neuron at level 2) passes that in
 * supervoxels seen. Entries go into the newest of several Maps, a new one
 * started every LARGE_MAP_SHARD entries, so the common small case is one
 * Map and one lookup.
 */
export const LARGE_MAP_SHARD = 1 << 23;
export class LargeMap<K, V> {
  private maps: Map<K, V>[] = [new Map()];
  constructor(private shard = LARGE_MAP_SHARD) {}
  get size() {
    let n = 0;
    for (const m of this.maps) n += m.size;
    return n;
  }
  get(k: K): V | undefined {
    const ms = this.maps;
    for (let i = ms.length - 1; i >= 0; --i) {
      const v = ms[i].get(k);
      if (v !== undefined || ms[i].has(k)) return v;
    }
    return undefined;
  }
  has(k: K): boolean {
    for (const m of this.maps) if (m.has(k)) return true;
    return false;
  }
  set(k: K, v: V): this {
    for (const m of this.maps)
      if (m.has(k)) {
        m.set(k, v);
        return this;
      }
    let last = this.maps[this.maps.length - 1];
    if (last.size >= this.shard) this.maps.push((last = new Map()));
    last.set(k, v);
    return this;
  }
  clear() {
    this.maps = [new Map()];
  }
}

/** Shared table of uint64 ids for label-index volumes; index 0 is id 0. */
export class IdTable {
  ids: bigint[] = [0n];
  private index = new LargeMap<bigint, number>();
  constructor() {
    this.index.set(0n, 0);
  }
  get(id: bigint): number {
    let i = this.index.get(id);
    if (i === undefined) {
      i = this.ids.length;
      this.ids.push(id);
      this.index.set(id, i);
    }
    return i;
  }
}

type Chunk = Uint8Array | Uint32Array;

export class PrecomputedVolume {
  readonly objectUrl: (path: string) => string;
  info!: VolumeInfo;
  scale!: ScaleInfo;
  /** Replaced (with the chunk cache) by resetIds(); read it afresh each time. */
  idTable = new IdTable();
  private chunks: LRU<Promise<Chunk | null>>;
  private shardIndex = new Map<string, Promise<Uint8Array>>();
  private minishards = new Map<
    string,
    Promise<Map<bigint, [number, number]>>
  >();
  private ready: Promise<void> | undefined;

  constructor(
    readonly url: string,
    private resolution: [number, number, number],
    cacheChunks = 256,
  ) {
    this.objectUrl = objectUrlFn(url);
    this.chunks = new LRU(cacheChunks);
  }

  /**
   * Forget every label index: a new id table and an empty chunk cache (the
   * cached chunks hold indices into the old table). Only safe when no cutout
   * is in flight, i.e. between runs; runTwigCapture calls it first, so the
   * table no longer grows across a session's runs until it hits the Map
   * limit (every region then failed, "Map maximum size exceeded").
   */
  resetIds() {
    this.idTable = new IdTable();
    this.chunks.clear();
  }

  init(): Promise<void> {
    if (!this.ready) {
      this.ready = (async () => {
        const r = await fetchRetrying(this.objectUrl("info"));
        if (!r.ok) throw new Error(`${this.url}/info: HTTP ${r.status}`);
        this.info = await r.json();
        const s = this.info.scales.find((sc) =>
          sc.resolution.every(
            (v, d) => Math.abs(v - this.resolution[d]) < 1e-6,
          ),
        );
        if (!s) {
          throw new Error(
            `${this.url}: no scale at ${this.resolution.join("x")} nm`,
          );
        }
        this.scale = s;
      })();
      this.ready.catch(() => (this.ready = undefined));
    }
    return this.ready;
  }

  get isSegmentation() {
    return this.scale.encoding === "compressed_segmentation";
  }

  /**
   * Voxels [lo, lo+shape) in this scale's voxel grid, x fastest. Outside the
   * volume reads as 0. Segmentation returns label indices into idTable.
   */
  async cutout(
    lo: readonly number[],
    shape: readonly number[],
    signal?: AbortSignal,
  ): Promise<Chunk> {
    await this.init();
    const { chunk_sizes, voxel_offset, size } = this.scale;
    const cs = chunk_sizes[0];
    const out = this.isSegmentation
      ? new Uint32Array(shape[0] * shape[1] * shape[2])
      : new Uint8Array(shape[0] * shape[1] * shape[2]);
    const c0 = [0, 1, 2].map((d) =>
      Math.floor((Math.max(lo[d], voxel_offset[d]) - voxel_offset[d]) / cs[d]),
    );
    const c1 = [0, 1, 2].map((d) =>
      Math.floor(
        (Math.min(lo[d] + shape[d], voxel_offset[d] + size[d]) -
          1 -
          voxel_offset[d]) /
          cs[d],
      ),
    );
    const jobs: Promise<void>[] = [];
    for (let cz = c0[2]; cz <= c1[2]; ++cz)
      for (let cy = c0[1]; cy <= c1[1]; ++cy)
        for (let cx = c0[0]; cx <= c1[0]; ++cx) {
          jobs.push(
            this.getChunk([cx, cy, cz], signal).then((data) => {
              if (!data) return;
              const cl = [cx, cy, cz].map(
                (c, d) => voxel_offset[d] + c * cs[d],
              );
              const ce = cl.map((v, d) =>
                Math.min(v + cs[d], voxel_offset[d] + size[d]),
              );
              const csz = ce.map((v, d) => v - cl[d]);
              const a = cl.map((v, d) => Math.max(v, lo[d]));
              const b = ce.map((v, d) => Math.min(v, lo[d] + shape[d]));
              for (let z = a[2]; z < b[2]; ++z)
                for (let y = a[1]; y < b[1]; ++y) {
                  const src =
                    a[0] - cl[0] + csz[0] * (y - cl[1] + csz[1] * (z - cl[2]));
                  const dst =
                    a[0] -
                    lo[0] +
                    shape[0] * (y - lo[1] + shape[1] * (z - lo[2]));
                  out.set(data.subarray(src, src + b[0] - a[0]) as any, dst);
                }
            }),
          );
        }
    await Promise.all(jobs);
    return out;
  }

  /**
   * Start loading the chunks under [lo, lo+shape) into the chunk cache
   * without copying anything out, so a later cutout there is a cache hit.
   * Failures are left to that cutout.
   */
  prefetch(lo: readonly number[], shape: readonly number[]) {
    if (!this.scale) return;
    const { chunk_sizes, voxel_offset, size } = this.scale;
    const cs = chunk_sizes[0];
    const c0 = [0, 1, 2].map((d) =>
      Math.floor((Math.max(lo[d], voxel_offset[d]) - voxel_offset[d]) / cs[d]),
    );
    const c1 = [0, 1, 2].map((d) =>
      Math.floor(
        (Math.min(lo[d] + shape[d], voxel_offset[d] + size[d]) -
          1 -
          voxel_offset[d]) /
          cs[d],
      ),
    );
    for (let cz = c0[2]; cz <= c1[2]; ++cz)
      for (let cy = c0[1]; cy <= c1[1]; ++cy)
        for (let cx = c0[0]; cx <= c1[0]; ++cx)
          this.getChunk([cx, cy, cz]).catch(() => {});
  }

  private getChunk(c: number[], signal?: AbortSignal) {
    const key = c.join(",");
    let p = this.chunks.get(key);
    if (!p) {
      p = this.loadChunk(c, signal);
      p.catch(() => this.chunks.set(key, undefined as any));
      this.chunks.set(key, p);
    }
    return p;
  }

  private async loadChunk(
    c: number[],
    signal?: AbortSignal,
  ): Promise<Chunk | null> {
    const { chunk_sizes, voxel_offset, size, sharding, key } = this.scale;
    const cs = chunk_sizes[0];
    const lo = c.map((v, d) => voxel_offset[d] + v * cs[d]);
    const hi = lo.map((v, d) => Math.min(v + cs[d], voxel_offset[d] + size[d]));
    const csz = hi.map((v, d) => v - lo[d]);
    let bytes: Uint8Array;
    if (sharding) {
      const grid = [0, 1, 2].map((d) => Math.ceil(size[d] / cs[d]));
      const id = encodeZIndexCompressed(
        Uint32Array.from(c),
        Uint32Array.from(grid),
      );
      const got = await this.readSharded(id, signal);
      if (!got) return null;
      bytes = got;
    } else {
      const name = `${lo[0]}-${hi[0]}_${lo[1]}-${hi[1]}_${lo[2]}-${hi[2]}`;
      const url = this.objectUrl(`${key}/${name}`);
      let missing = false;
      bytes = await tracked(url, async () => {
        const r = await fetchRetrying(url, {}, signal);
        if (r.status === 404) {
          missing = true;
          return new Uint8Array(0);
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return new Uint8Array(await r.arrayBuffer());
      });
      if (missing) return null;
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) bytes = await gunzip(bytes);
    }
    const n = csz[0] * csz[1] * csz[2];
    switch (this.scale.encoding) {
      case "compressed_segmentation":
        return this.decodeCompressedSegmentation(bytes, csz);
      case "jpeg":
        return decodeJpegGray(bytes, csz[0], csz[1] * csz[2]);
      case "raw":
        return bytes.subarray(0, n);
      default:
        throw new Error(`unsupported encoding ${this.scale.encoding}`);
    }
  }

  private async readSharded(
    chunkId: bigint,
    signal?: AbortSignal,
  ): Promise<Uint8Array | null> {
    const sh = this.scale.sharding!;
    const hashed = chunkId >> BigInt(sh.preshift_bits);
    if (sh.hash !== "identity") throw new Error(`hash ${sh.hash} unsupported`);
    const minishard = Number(hashed & ((1n << BigInt(sh.minishard_bits)) - 1n));
    const shard = Number(
      (hashed >> BigInt(sh.minishard_bits)) &
        ((1n << BigInt(sh.shard_bits)) - 1n),
    );
    const name = shard.toString(16).padStart(Math.ceil(sh.shard_bits / 4), "0");
    const url = this.objectUrl(`${this.scale.key}/${name}.shard`);
    const indexBytes = 16 * (1 << sh.minishard_bits);
    let idx = this.shardIndex.get(url);
    if (!idx) {
      idx = fetchRange(url, 0, indexBytes, signal);
      idx.catch(() => this.shardIndex.delete(url));
      this.shardIndex.set(url, idx);
    }
    const index = await idx;
    if (index.length < indexBytes) return null;
    const mkey = `${url}#${minishard}`;
    let mp = this.minishards.get(mkey);
    if (!mp) {
      mp = (async () => {
        const dv = new DataView(
          index.buffer,
          index.byteOffset,
          index.byteLength,
        );
        const start = Number(dv.getBigUint64(16 * minishard, true));
        const end = Number(dv.getBigUint64(16 * minishard + 8, true));
        const map = new Map<bigint, [number, number]>();
        if (end <= start) return map;
        let raw = await fetchRange(
          url,
          indexBytes + start,
          indexBytes + end,
          signal,
        );
        if (sh.minishard_index_encoding === "gzip") raw = await gunzip(raw);
        const v = new BigUint64Array(
          raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
        );
        const m = v.length / 3;
        let id = 0n;
        let pos = 0n;
        for (let i = 0; i < m; ++i) {
          id += v[i];
          const s0 = pos + v[m + i];
          const len = v[2 * m + i];
          map.set(id, [indexBytes + Number(s0), Number(len)]);
          pos = s0 + len;
        }
        return map;
      })();
      mp.catch(() => this.minishards.delete(mkey));
      this.minishards.set(mkey, mp);
    }
    const entry = (await mp).get(chunkId);
    if (!entry) return null;
    let data = await fetchRange(url, entry[0], entry[0] + entry[1], signal);
    if (sh.data_encoding === "gzip") data = await gunzip(data);
    return data;
  }

  /** compressed_segmentation, uint64, 1 channel -> label indices. */
  private decodeCompressedSegmentation(bytes: Uint8Array, csz: number[]) {
    const data = new Uint32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    const bs = this.scale.compressed_segmentation_block_size ?? [8, 8, 8];
    const out = new Uint32Array(csz[0] * csz[1] * csz[2]);
    const base = data[0]; // channel 0 offset (uint32 words)
    const gb = [0, 1, 2].map((d) => Math.ceil(csz[d] / bs[d]));
    const table = this.idTable;
    for (let bz = 0; bz < gb[2]; ++bz)
      for (let by = 0; by < gb[1]; ++by)
        for (let bx = 0; bx < gb[0]; ++bx) {
          const h = base + 2 * (bx + gb[0] * (by + gb[1] * bz));
          const tableOffset = base + (data[h] & 0xffffff);
          const bits = data[h] >>> 24;
          const valuesOffset = base + data[h + 1];
          // Table entries are mapped to global label indices lazily: the
          // table's length is not stored, so reading 2^bits entries could
          // pull in bytes that belong to the next block.
          const lut = new Map<number, number>();
          const lookup = (e: number) => {
            let g = lut.get(e);
            if (g === undefined) {
              const lo = data[tableOffset + 2 * e];
              const hi = data[tableOffset + 2 * e + 1];
              g =
                lo === 0 && hi === 0
                  ? 0
                  : table.get((BigInt(hi) << 32n) | BigInt(lo));
              lut.set(e, g);
            }
            return g;
          };
          const mask =
            bits === 0 ? 0 : bits >= 32 ? 0xffffffff : (1 << bits) - 1;
          const x0 = bx * bs[0];
          const y0 = by * bs[1];
          const z0 = bz * bs[2];
          for (let z = 0; z < bs[2]; ++z) {
            const zz = z0 + z;
            if (zz >= csz[2]) break;
            for (let y = 0; y < bs[1]; ++y) {
              const yy = y0 + y;
              if (yy >= csz[1]) break;
              for (let x = 0; x < bs[0]; ++x) {
                const xx = x0 + x;
                if (xx >= csz[0]) break;
                let v = 0;
                if (bits) {
                  const k = x + bs[0] * (y + bs[1] * z);
                  const bit = k * bits;
                  v =
                    ((data[valuesOffset + (bit >>> 5)] >>> (bit & 31)) &
                      mask) >>>
                    0;
                }
                out[xx + csz[0] * (yy + csz[1] * zz)] = lookup(v);
              }
            }
          }
        }
    return out;
  }
}

/** Decode a grayscale JPEG of width x height into one byte per pixel. */
async function decodeJpegGray(
  bytes: Uint8Array,
  width: number,
  height: number,
): Promise<Uint8Array> {
  const bitmap = await createImageBitmap(
    new Blob([bytes as BlobPart], { type: "image/jpeg" }),
    { colorSpaceConversion: "none", premultiplyAlpha: "none" },
  );
  if (bitmap.width !== width || bitmap.height !== height) {
    throw new Error(
      `jpeg chunk is ${bitmap.width}x${bitmap.height}, expected ${width}x${height}`,
    );
  }
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; ++i) out[i] = rgba[4 * i];
  return out;
}

/** For measuring: `twigCaptureFetch.setMirrors(false)` in the console. */
(globalThis as any).twigCaptureFetch = {
  hostStats,
  setMirrors: (on: boolean) => (mirrorsOn = on),
};
