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
 * In-browser port of run_twigcapture_v2.py (neuron-first, models.use: local):
 * root id -> L2 nodes -> binned regions -> supervoxel cutouts -> touching
 * candidates -> contact sites -> patches -> ONNX model -> a result JSON in the
 * runner's own schema, which the Twig Capture tab opens like any result file.
 *
 * Differences from the cluster runner, all deliberate:
 * - no synapse-density gate (needs a cluster-only grid), so every non-backbone
 *   L2 bin is searched; `maxRegions` caps the work;
 * - EM is the dataset's public 8 nm mip (FANC 8.6 nm, see datasets.ts);
 * - synapse counts are not computed (no synapse tables in the browser).
 */

import { requestAsyncComputation } from "#src/async_computation/request.js";
import { twigContactSites } from "#src/layer/segmentation/twig_inference/contact_sites_request.js";
import type { TwigDataset } from "#src/layer/segmentation/twig_inference/datasets.js";
import {
  TWIG_PARAMS,
  gridOf,
} from "#src/layer/segmentation/twig_inference/datasets.js";
import { logEvent } from "#src/layer/segmentation/twig_inference/diagnostics.js";
import type { MergeModel } from "#src/layer/segmentation/twig_inference/model.js";
import {
  IdTable,
  PrecomputedVolume,
} from "#src/layer/segmentation/twig_inference/precomputed_reader.js";
import type {
  Shape,
  findContactSites,
} from "#src/layer/segmentation/twig_inference/volume_ops.js";
import {
  PATCH_VOXELS,
  labelBoxes,
  organelleFeatures,
  touchingLabelsCenter,
  voxelNear,
  writeMasks,
} from "#src/layer/segmentation/twig_inference/volume_ops.js";
import type { LocalResultJson } from "#src/layer/segmentation/twig_local_results.js";

/** What the pipeline needs from the graphene graph source. */
export interface GraphAccess {
  fetchGraph(path: string, init?: RequestInit): Promise<Response>;
  info: { app?: { l2CacheUrl: string; table: string } };
}

export interface RunOptions {
  /** Region cap per level. */
  maxRegions: number;
  /**
   * 1: merges onto the neuron. 2+: suggestions with p >= chainThreshold are
   * treated as merged and the search repeats around them (as the orphan-first
   * runner simulates merges), so level 2 also proposes merges onto the
   * level-1 suggestions.
   */
  level: number;
  chainThreshold: number;
  /** Contact sites scored per pair (8 in the runner; 1 is ~3x less work). */
  maxSites: number;
  /** Largest candidate proposed, in L2 nodes (default 20, as the cluster). */
  maxCandidateL2?: number;
  /** contact_gap_voxels (default TWIG_PARAMS.contactGap). */
  contactGap?: number;
  /** How it applies (default TWIG_PARAMS.contactGapMode). */
  contactGapMode?: "all" | "nontouching";
  modelUrl: string;
}

export interface Progress {
  phase: string;
  done: number;
  total: number;
  patches: number;
  found: number;
  /** Model time per patch so far (ms), once a patch has been scored. */
  msPerPatch?: number;
}

/**
 * Time spent per phase of a run, in ms. Phases can overlap once regions are
 * pipelined (see runTwigCapture), so they need not sum to the wall time:
 * each is the wall time that phase was busy, summed over its calls.
 */
export class PhaseTimer {
  readonly ms: Record<string, number> = {};
  readonly calls: Record<string, number> = {};
  add(phase: string, ms: number) {
    this.ms[phase] = (this.ms[phase] ?? 0) + ms;
    this.calls[phase] = (this.calls[phase] ?? 0) + 1;
  }
  async time<T>(phase: string, f: () => Promise<T>): Promise<T> {
    const t = performance.now();
    try {
      return await f();
    } finally {
      this.add(phase, performance.now() - t);
    }
  }
  sync<T>(phase: string, f: () => T): T {
    const t = performance.now();
    try {
      return f();
    } finally {
      this.add(phase, performance.now() - t);
    }
  }
  /** Seconds per phase, rounded, for the result's benchmark block. */
  seconds(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(this.ms))
      out[k] = +(v / 1000).toFixed(2);
    return out;
  }
}

interface L2Attr {
  rep_coord_nm?: [number, number, number];
  max_dt_nm?: number;
  size_nm3?: number;
}

/** Shared per dataset: volumes, caches that stay valid across runs. */
export class DatasetContext {
  readonly sv: PrecomputedVolume;
  readonly em: PrecomputedVolume;
  readonly svRoot = new Map<bigint, bigint>();
  readonly l2Count = new Map<string, number>();
  constructor(readonly dataset: TwigDataset) {
    gridOf(dataset); // throws if the EM is not 4x finer than the supervoxels
    this.sv = new PrecomputedVolume(
      dataset.supervoxelUrl,
      dataset.supervoxelResolution,
      48,
    );
    this.em = new PrecomputedVolume(dataset.emUrl, dataset.emResolution, 512);
  }
}

async function json(r: Promise<Response>) {
  return (await r).json();
}

export async function getLeavesL2(
  g: GraphAccess,
  root: string,
): Promise<string[]> {
  const j = await json(
    g.fetchGraph(`/node/${root}/leaves?stop_layer=2&int64_as_str=1`),
  );
  return (j.leaf_ids ?? []).map(String);
}

async function getL2Attributes(
  g: GraphAccess,
  ids: string[],
): Promise<Record<string, L2Attr>> {
  const app = g.info.app;
  if (!app) throw new Error("graph source has no L2 cache");
  const out: Record<string, L2Attr> = {};
  for (let i = 0; i < ids.length; i += 2000) {
    const part = ids.slice(i, i + 2000);
    const j = await json(
      g.fetchGraph(
        `${app.l2CacheUrl}/table/${app.table}/attributes?int64_as_str=1&attribute_names=rep_coord_nm,max_dt_nm,size_nm3`,
        { method: "POST", body: JSON.stringify({ l2_ids: part }) },
      ),
    );
    Object.assign(out, j);
  }
  return out;
}

/**
 * Roots of many supervoxels (chunkedgraph roots_binary, as caveclient).
 * With `stopLayer` 2, their L2 nodes instead.
 */
async function getRoots(
  g: GraphAccess,
  svs: bigint[],
  stopLayer?: number,
): Promise<bigint[]> {
  const out: bigint[] = [];
  const path = "/roots_binary" + (stopLayer ? `?stop_layer=${stopLayer}` : "");
  for (let i = 0; i < svs.length; i += 20000) {
    const part = BigUint64Array.from(svs.slice(i, i + 20000));
    let got: BigUint64Array | undefined;
    for (let attempt = 0; attempt < 2 && !got; ++attempt) {
      try {
        const r = await g.fetchGraph(path, {
          method: "POST",
          body: part.buffer as ArrayBuffer,
          headers: { "Content-Type": "application/octet-stream" },
        });
        got = new BigUint64Array(await r.arrayBuffer());
      } catch (e) {
        if (attempt) throw e;
        await new Promise((res) => setTimeout(res, 1000));
      }
    }
    if (got!.length !== part.length)
      throw new Error("roots_binary: length mismatch");
    out.push(...got!);
  }
  return out;
}

/** EM cutouts fetched at once per region (was 6). */
const EM_CONCURRENCY = 12;
/** Regions whose supervoxel cutouts are fetched ahead of the one processed. */
const PREFETCH_REGIONS = 2;
/**
 * Patches that may wait for the model behind the region being processed
 * (~3.5 MB each, so ~450 MB). At least one region always waits, however
 * large, so the model is never idle while a region is cut out.
 */
const MAX_QUEUED_PATCHES = 128;

async function pool<T>(items: T[], n: number, f: (t: T) => Promise<void>) {
  let i = 0;
  await Promise.all(
    Array.from({ length: Math.min(n, items.length) }, async () => {
      while (i < items.length) await f(items[i++]);
    }),
  );
}

function stamp(d: Date) {
  const p = (v: number) => String(v).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

interface SiteBuilt {
  pair: string;
  rank: number;
  voxels: number;
  /** The claimed site position, released again if the site cannot be scored. */
  claim: [number, number, number];
  row: any;
}

export async function runTwigCapture(
  root: string,
  graph: GraphAccess,
  ctx: DatasetContext,
  model: MergeModel,
  opts: RunOptions,
  onProgress: (p: Progress) => void,
  signal: AbortSignal,
): Promise<LocalResultJson> {
  const P = TWIG_PARAMS;
  const ds = ctx.dataset;
  const { svNm, binNm, zRes, emPerSv } = gridOf(ds);
  const t0 = performance.now();
  const prog: Progress = {
    phase: "L2 graph",
    done: 0,
    total: 0,
    patches: 0,
    found: 0,
  };
  const timer = new PhaseTimer();
  const gap = opts.contactGap ?? P.contactGap;
  const gapMode = opts.contactGapMode ?? P.contactGapMode;
  const report = () => onProgress({ ...prog });
  const check = () => {
    if (signal.aborted) throw new DOMException("cancelled", "AbortError");
  };
  report();
  await Promise.all([ctx.sv.init(), ctx.em.init()]);

  const s: Shape = P.cutout;
  const rootBig = BigInt(root);
  const rejected: Record<string, number> = {};
  const rejectedOnce = new Set<string>();
  const rejectLog: any[] = [];
  const claimed = new Map<string, [number, number, number][]>();
  const best = new Map<
    string,
    SiteBuilt & { p: number; found: number; scored: number; hop: number }
  >();
  const reject = (cand: string, why: string, extra: any = {}) => {
    if (rejectedOnce.has(cand)) return;
    rejectedOnce.add(cand);
    rejected[why] = (rejected[why] ?? 0) + 1;
    rejectLog.push({ target_segment_id: cand, reason: why, ...extra });
  };
  const emSize = ctx.em.scale.size;
  const emOff = ctx.em.scale.voxel_offset;
  const half = [16 * svNm, 16 * svNm, 9 * zRes];
  let regionErrors = 0;
  let regionsSearched = 0;
  let regionsTotal = 0;
  let totalL2 = 0;

  // Coverage (see coverageCentres): the L2 nodes this level searches, and
  // cells of their voxels that a region saw outside its search band.
  let keptL2 = new Set<bigint>();
  const svL2 = new Map<bigint, bigint>();
  let coverageCells = new Set<string>();
  let coverageRegions = 0;
  let coverageErrors = 0;

  /** L2 nodes of `roots` -> binned region centres (first node per bin). */
  const regionsFor = async (roots: string[], skipBackbone: boolean) => {
    const all: [number, number, number][] = [];
    const seen = new Set<string>();
    keptL2 = new Set<bigint>();
    for (const r of roots) {
      const leaves = await timer.time("l2_graph", () => getLeavesL2(graph, r));
      totalL2 += leaves.length;
      check();
      const attrs = await timer.time("l2_graph", () =>
        getL2Attributes(graph, leaves),
      );
      check();
      for (const id of keptL2Ids(leaves, attrs, skipBackbone))
        keptL2.add(BigInt(id));
      for (const c of binL2(
        leaves,
        attrs,
        ds.supervoxelResolution,
        skipBackbone,
      )) {
        const key = c.join(",");
        if (!seen.has(key)) {
          seen.add(key);
          all.push(c);
        }
      }
    }
    // One region per bin across all roots (first seen wins), as before.
    const bins = new Map<string, [number, number, number]>();
    for (const c of all) {
      const key = `${Math.floor(c[0] / P.binXY)},${Math.floor(c[1] / P.binXY)},${Math.floor(c[2] / P.binZ)}`;
      if (!bins.has(key)) bins.set(key, c);
    }
    return [...bins.values()];
  };

  /**
   * Coverage, part 1: store (as COVERAGE_GRID cells, in global supervoxel
   * voxels) the neuron's voxels that this cutout holds OUTSIDE its search
   * band and that belong to an L2 node this level searches. Backbone/soma
   * nodes skipped by binL2 stay skipped. A node that reaches past a band is
   * connected and has voxels in it, so it always crosses the border zone and
   * is seen here.
   */
  const recordOutOfBand = async (
    svLab: Uint32Array,
    rootLab: Uint32Array,
    target: number,
    origin: number[],
  ) => {
    const b = P.border;
    const [X, Y, Z] = s;
    const out: number[] = [];
    const labs = new Set<number>();
    for (let z = 0, i = 0; z < Z; ++z) {
      const zIn = z >= b && z < Z - b;
      for (let y = 0; y < Y; ++y) {
        const yzIn = zIn && y >= b && y < Y - b;
        for (let x = 0; x < X; ++x, ++i) {
          if (rootLab[i] !== target) continue;
          if (yzIn && x >= b && x < X - b) continue;
          out.push(i);
          labs.add(svLab[i]);
        }
      }
    }
    if (!out.length) return;
    const svIds = ctx.sv.idTable.ids;
    const need = [...labs].map((l) => svIds[l]).filter((sv) => !svL2.has(sv));
    if (need.length) {
      const l2 = await getRoots(graph, need, 2);
      need.forEach((sv, k) => svL2.set(sv, l2[k]));
    }
    const keepLab = new Set(
      [...labs].filter((l) => keptL2.has(svL2.get(svIds[l])!)),
    );
    if (!keepLab.size) return;
    const G = COVERAGE_GRID;
    for (const i of out) {
      if (!keepLab.has(svLab[i])) continue;
      const x = i % X;
      const y = Math.floor(i / X) % Y;
      const z = Math.floor(i / (X * Y));
      coverageCells.add(
        `${Math.floor((origin[0] + x) / G[0])},${Math.floor((origin[1] + y) / G[1])},${Math.floor((origin[2] + z) / G[2])}`,
      );
    }
  };

  /**
   * One region: every root in `object` counts as the neuron (merges already
   * accepted are simulated by relabelling them), and new neighbours are
   * proposed with hop = `hop`.
   */
  /**
   * A region's supervoxel cutout, 200 x 200 x 120 at [svNm, svNm, z] (bins
   * are half a supervoxel voxel), with the roots of its supervoxels in
   * ctx.svRoot. Neither depends on what earlier regions found, so the next
   * regions' are fetched while this one is processed and scored (see
   * `prefetch`); the result is the same as fetching each in turn.
   */
  const fetchRegion = async (origin: number[]) => {
    const svLab = (await timer.time("sv_cutout", () =>
      ctx.sv.cutout(origin, s, signal),
    )) as Uint32Array;
    check();
    const svIds = ctx.sv.idTable.ids;
    // Supervoxels -> roots (cached; sv roots do not change mid-run).
    const tCpu = performance.now();
    const present = new Set<number>();
    for (let i = 0; i < svLab.length; ++i) present.add(svLab[i]);
    present.delete(0);
    const need = [...present]
      .map((l) => svIds[l])
      .filter((sv) => !ctx.svRoot.has(sv));
    timer.add("sites", performance.now() - tCpu);
    if (need.length) {
      const roots = await timer.time("sv_roots", () => getRoots(graph, need));
      need.forEach((sv, i) => ctx.svRoot.set(sv, roots[i]));
    }
    return { svLab, present };
  };
  const originOf = (coord: [number, number, number]) => {
    const g = [Math.floor(coord[0] / 2), Math.floor(coord[1] / 2), coord[2]];
    return [g[0] - s[0] / 2, g[1] - s[1] / 2, g[2] - s[2] / 2];
  };
  const fetched = new Map<string, ReturnType<typeof fetchRegion>>();
  /** Start fetching these regions (at most PREFETCH_REGIONS outstanding). */
  const prefetch = (coords: [number, number, number][]) => {
    for (const c of coords) {
      if (fetched.size >= PREFETCH_REGIONS) break;
      const key = c.join(",");
      if (fetched.has(key)) continue;
      const f = fetchRegion(originOf(c));
      f.catch(() => {}); // surfaced when the region is processed
      fetched.set(key, f);
    }
  };
  const fetchedRegion = (coord: [number, number, number]) => {
    const key = coord.join(",");
    const f = fetched.get(key) ?? fetchRegion(originOf(coord));
    fetched.delete(key);
    return f;
  };

  // Scoring runs behind the regions: a region queues its patches and the
  // next region is cut out while the model scores them. Results are applied
  // in region order, exactly as if each region had waited for its scores.
  // `queuedPairs` stands in for `best` for pairs queued but not yet applied,
  // which is all a later region reads of `best` within a level.
  let scoring: Promise<void> = Promise.resolve();
  const scoringJobs: { done: Promise<void>; patches: number }[] = [];
  let queuedPatches = 0;
  const queuedPairs = new Map<string, number>();
  const drainScoring = async () => {
    await scoring;
    scoringJobs.length = 0;
    queuedPatches = 0;
    check();
  };

  /**
   * A region's work that reads nothing earlier regions write (claims, scores,
   * rejects): roots -> labels, coverage cells, the neighbour search, partner
   * sizes and every candidate's contact sites. It runs while the previous
   * region's EM is fetched (see `search`); `finishRegion` then does the rest
   * in region order, so the result is the same as one region at a time.
   */
  const prepareRegion = async (
    coord: [number, number, number],
    object: Set<bigint>,
  ) => {
    const origin = originOf(coord);
    const { svLab, present } = await timer.time("wait_cutout", () =>
      fetchedRegion(coord),
    );
    check();
    const svIds = ctx.sv.idTable.ids;
    let tCpu = performance.now();
    const rootTable = new IdTable();
    const target = rootTable.get(rootBig);
    const svToRoot = new Map<number, number>();
    for (const l of present) {
      const r = ctx.svRoot.get(svIds[l])!;
      svToRoot.set(l, object.has(r) ? target : rootTable.get(r));
    }
    const rootLab = new Uint32Array(svLab.length);
    for (let i = 0; i < svLab.length; ++i) {
      const l = svLab[i];
      if (l) rootLab[i] = svToRoot.get(l)!;
    }
    timer.add("labels", performance.now() - tCpu);
    // Coverage: record this level's neuron voxels outside the search band.
    try {
      await timer.time("coverage", () =>
        recordOutOfBand(svLab, rootLab, target, origin),
      );
    } catch (e) {
      if ((e as any)?.name === "AbortError") throw e;
      coverageErrors++;
    }
    // Neighbours touching the (simulated) neuron in the centre box.
    tCpu = performance.now();
    const { keep, rejects } = touchingLabelsCenter(rootLab, s, target, {
      border: P.border,
      minContact: P.minContactVoxels,
      maxContact: P.maxContactVoxels,
      minSegment: P.minSegmentVoxels,
      gap,
      gapMode,
    });
    timer.add("touching", performance.now() - tCpu);
    // Partner sizes (cached per root), in parallel.
    const keepIds = [...keep.keys()];
    await timer.time("cand_l2", () =>
      pool(keepIds, 8, async (l) => {
        const id = String(rootTable.ids[l]);
        if (!ctx.l2Count.has(id)) {
          try {
            ctx.l2Count.set(id, (await getLeavesL2(graph, id)).length);
          } catch {
            ctx.l2Count.set(id, 0);
          }
        }
      }),
    );
    check();
    // Each candidate's contact sites. The runner's create_contact_mask: in
    // "nontouching" mode a pair keeps its one-voxel interface (judged on the
    // candidate's whole bbox, not the centre band) and only a pair with none
    // is measured across the gap; "all" widens every pair.
    tCpu = performance.now();
    // On the async-computation workers, one candidate a task, each given
    // only its crop: its box +- (2 + gap), clipped to the cutout, which is
    // exactly what findContactSites crops to itself, so the sites are the
    // same (offset back by the crop's corner).
    const sitesFor = new Map<number, ReturnType<typeof findContactSites>>();
    const boxes = labelBoxes(rootLab, s, keepIds);
    const first = gapMode === "all" ? gap : 0;
    const fallback = gap > 0 && gapMode !== "all" ? gap : undefined;
    const margin = 2 + Math.max(first, fallback ?? 0);
    await Promise.all(
      keepIds.map(async (l) => {
        const [blo, bhi] = boxes.get(l)!;
        if (bhi[0] < 0) {
          sitesFor.set(l, []);
          return;
        }
        const lo = blo.map((v) => Math.max(0, v - margin));
        const hi = bhi.map((v, d) => Math.min(s[d], v + margin + 1));
        const cs: [number, number, number] = [
          hi[0] - lo[0],
          hi[1] - lo[1],
          hi[2] - lo[2],
        ];
        const crop = new Uint32Array(cs[0] * cs[1] * cs[2]);
        for (let z = 0; z < cs[2]; ++z)
          for (let y = 0; y < cs[1]; ++y) {
            const src = lo[0] + s[0] * (lo[1] + y + s[1] * (lo[2] + z));
            crop.set(
              rootLab.subarray(src, src + cs[0]),
              cs[0] * (y + cs[1] * z),
            );
          }
        const sites = await requestAsyncComputation(
          twigContactSites,
          signal,
          [crop.buffer],
          crop,
          cs,
          target,
          l,
          P.minContactBlob,
          opts.maxSites,
          first,
          fallback,
          lo as [number, number, number],
        );
        sitesFor.set(l, sites);
      }),
    );
    timer.add("contacts", performance.now() - tCpu);
    return {
      origin,
      svLab,
      rootTable,
      target,
      rootLab,
      keep,
      rejects,
      sitesFor,
    };
  };
  type Prepared = Awaited<ReturnType<typeof prepareRegion>>;

  const finishRegion = async (
    prep: Prepared,
    coord: [number, number, number],
    hop: number,
  ) => {
    const { origin, svLab, rootTable, target, rootLab, keep, rejects } = prep;
    const svIds = ctx.sv.idTable.ids;
    const tCpu = performance.now();
    for (const [l, [why, ct, sz]] of rejects) {
      const id = String(rootTable.ids[l]);
      if (!best.has(id) && !queuedPairs.has(id))
        reject(id, why, { contact_voxels: ct, cand_voxels: sz });
    }
    // A pair scored at an earlier level keeps that level's score.
    const cands = [...keep.keys()].filter((l) => {
      const b = best.get(String(rootTable.ids[l]));
      return !b || b.hop === hop;
    });
    // Contact sites -> patches.
    const built: SiteBuilt[] = [];
    const pending: Float32Array[] = [];
    const foundPerPair = new Map<string, number>();
    for (const l of cands) {
      const candId = String(rootTable.ids[l]);
      const stats = keep.get(l)!;
      const nL2 = ctx.l2Count.get(candId) ?? 0;
      if (nL2 > (opts.maxCandidateL2 ?? P.maxCandidateL2)) {
        reject(candId, "candidate_l2", { ...stats, num_l2_nodes: nL2 });
        continue;
      }
      const sites = prep.sitesFor.get(l)!;
      if (!sites.length) {
        reject(candId, "no_contact", stats);
        continue;
      }
      foundPerPair.set(candId, sites.length);
      const seen = claimed.get(candId) ?? [];
      claimed.set(candId, seen);
      let reason: string | undefined;
      sites.forEach(([voxels, centroid], i) => {
        const c = centroid.map((v) => Math.trunc(v)) as [
          number,
          number,
          number,
        ];
        // v1-style site position for the claim test (_site_nm).
        const q = [
          (coord[0] + (c[0] - 100) * 2) * binNm,
          (coord[1] + (c[1] - 100) * 2) * binNm,
          (coord[2] + (c[2] - 60)) * zRes,
        ];
        if (
          seen.some((r) =>
            [0, 1, 2].every((d) => Math.abs(q[d] - r[d]) < half[d]),
          )
        )
          return;
        const x = new Float32Array(3 * PATCH_VOXELS);
        if (!writeMasks(rootLab, s, c, target, l, x, PATCH_VOXELS)) {
          reason ??= "local_overflow";
          return;
        }
        seen.push(q as [number, number, number]);
        const gc = [origin[0] + c[0], origin[1] + c[1], origin[2] + c[2]];
        const ctrNm = [gc[0] * svNm, gc[1] * svNm, gc[2] * zRes];
        let sinkRoot: bigint | undefined;
        const side = (lab: number) => {
          const pt = voxelNear(rootLab, s, c, lab);
          if (!pt) return null;
          const sv = svIds[svLab[pt[0] + s[0] * (pt[1] + s[1] * pt[2])]];
          if (lab === target) sinkRoot = ctx.svRoot.get(sv);
          return {
            supervoxel_id: String(sv),
            coordinate_nm: [
              (origin[0] + pt[0]) * svNm,
              (origin[1] + pt[1]) * svNm,
              (origin[2] + pt[2]) * zRes,
            ],
          };
        };
        const merge_sink = side(target);
        const merge_source = side(l);
        built.push({
          pair: candId,
          rank: i + 1,
          voxels,
          claim: q as [number, number, number],
          row: {
            target_segment_id: candId,
            coordinate_nm: ctrNm,
            num_l2_nodes: nL2,
            contact_voxels: stats.contact_voxels,
            cand_voxels: stats.cand_voxels,
            // Found only because contact_gap_voxels widened the search: the
            // two do not touch at supervoxel resolution (as the runner).
            gap_contact: stats.gap_contact,
            cand_size_um3: +(
              (stats.cand_voxels * svNm * svNm * zRes) /
              1e9
            ).toFixed(4),
            merge_sink,
            merge_source,
            site_rank: i + 1,
            site_voxels: voxels,
            cutout_origin_sv: origin,
            centroid_sv: c,
            // Order of the merge, from what the sink side lies on: the neuron
            // itself is 1 even when found while searching a later level
            // (that level's windows overlap the neuron where level 1's region
            // cap stopped short); a suggestion from level k makes it k + 1.
            // best.hop below stays the level that found it, for chaining.
            hop: sinkRoot === rootBig ? 1 : hop,
            // The piece the sink side actually lies on: the neuron, or a
            // suggestion from an earlier level that this one would join.
            ...(sinkRoot !== undefined && sinkRoot !== rootBig
              ? { via_root: String(sinkRoot) }
              : {}),
            _emCentre: [gc[0] * emPerSv, gc[1] * emPerSv, gc[2]],
          },
        });
        pending.push(x);
      });
      if (reason && !built.some((b) => b.pair === candId))
        reject(candId, reason, stats);
    }
    timer.add("sites", performance.now() - tCpu);
    // EM for each built site; drop out-of-volume / empty windows.
    const scoreIdx: number[] = [];
    const tEm = performance.now();
    await pool(
      built.map((_, i) => i),
      EM_CONCURRENCY,
      async (i) => {
        const b = built[i];
        const e = b.row._emCentre as number[];
        const lo = [e[0] - 64, e[1] - 64, e[2] - 9];
        if (
          lo.some((v, d) => v < emOff[d]) ||
          lo.some((v, d) => v + [128, 128, 18][d] > emOff[d] + emSize[d])
        ) {
          b.row._reject = "em_overflow";
          return;
        }
        const em = (await ctx.em.cutout(
          lo,
          [128, 128, 18],
          signal,
        )) as Uint8Array;
        let zeros = 0;
        for (let k = 0; k < em.length; ++k) if (em[k] === 0) zeros++;
        const zf = +(zeros / em.length).toFixed(4);
        b.row.em_zero_frac = zf;
        if (zf > P.maxEmZeroFrac) {
          b.row._reject = "em_zero_frac";
          return;
        }
        const x = pending[i];
        Object.assign(
          b.row,
          organelleFeatures(
            em,
            x.subarray(PATCH_VOXELS, 2 * PATCH_VOXELS),
            x.subarray(2 * PATCH_VOXELS),
          ),
        );
        for (let k = 0; k < em.length; ++k) x[k] = (em[k] / 255) * 2 - 1;
        scoreIdx.push(i);
      },
    );
    timer.add("em_cutout", performance.now() - tEm);
    // In site order, not the order the EM cutouts happened to arrive in, so
    // a run's result (down to the order of its rejected_candidates log, and
    // ties between a pair's sites) does not depend on network timing.
    scoreIdx.sort((a, b) => a - b);
    check();
    for (const b of built) {
      // As the runner: a site that could not be scored releases its claim,
      // so another region where its EM window is usable can still try it.
      if (b.row._reject) {
        const seen = claimed.get(b.pair);
        const k = seen?.indexOf(b.claim) ?? -1;
        if (k >= 0) seen!.splice(k, 1);
      }
      if (b.row._reject && !scoreIdx.some((i) => built[i].pair === b.pair)) {
        reject(b.pair, b.row._reject, {
          contact_voxels: b.row.contact_voxels,
          cand_voxels: b.row.cand_voxels,
        });
      }
    }
    // Score, and keep the best site per pair (max over sites): queued
    // behind the regions, applied in order (see `scoring`).
    if (scoreIdx.length) {
      // The patches themselves, not a packed copy: a queued region holds
      // ~3.5 MB a patch until it is scored.
      const patches = scoreIdx.map((i) => pending[i]);
      const pairs = scoreIdx.map((i) => built[i].pair);
      for (const id of pairs)
        queuedPairs.set(id, (queuedPairs.get(id) ?? 0) + 1);
      // Bound what waits for the model (memory), oldest region first.
      while (
        scoringJobs.length &&
        queuedPatches + patches.length > MAX_QUEUED_PATCHES
      ) {
        const job = scoringJobs.shift()!;
        await job.done;
        queuedPatches -= job.patches;
      }
      queuedPatches += patches.length;
      scoring = scoring.then(async () => {
        try {
          check();
          prog.phase = `level ${hop}: scoring on ${model.backend === "webgpu" ? "GPU" : "CPU"}`;
          report();
          const p = await timer.time("model", () =>
            model.scorePatches(patches),
          );
          patches.length = 0;
          prog.patches += scoreIdx.length;
          prog.msPerPatch = +(timer.ms.model / prog.patches).toFixed(1);
          scoreIdx.forEach((i, k) => {
            const b = built[i];
            const prev = best.get(b.pair);
            const scored = (prev?.scored ?? 0) + 1;
            const found = Math.max(
              prev?.found ?? 0,
              foundPerPair.get(b.pair) ?? 0,
            );
            if (!prev || p[k] > prev.p)
              best.set(b.pair, { ...b, p: p[k], found, scored, hop });
            else Object.assign(prev, { found, scored });
          });
          prog.found = [...best.values()].filter(
            (b) => b.p > P.reportThreshold,
          ).length;
          report();
        } catch (e) {
          // As a region whose scoring failed before: counted, logged, and
          // the run goes on. A cancel stops the run when drained.
          if ((e as any)?.name !== "AbortError") {
            regionErrors++;
            logEvent(
              "warn",
              `region ${coord.join(",")} (level ${hop}) scoring failed`,
              e,
            );
          }
        } finally {
          for (const id of pairs) {
            const n = queuedPairs.get(id)! - 1;
            if (n) queuedPairs.set(id, n);
            else queuedPairs.delete(id);
          }
        }
      });
      scoringJobs.push({ done: scoring, patches: scoreIdx.length });
    }
  };

  // Levels: 1 searches the neuron; each further level adds the previous
  // level's confident suggestions to the object and searches around them.
  const object = new Set<bigint>([rootBig]);
  let frontier: string[] = [root];
  for (let hop = 1; hop <= Math.max(1, opts.level) && frontier.length; ++hop) {
    prog.phase = `level ${hop}: L2 graph`;
    report();
    const all = await regionsFor(frontier, hop === 1);
    const regions = spreadRegions(all, opts.maxRegions);
    regionsTotal += all.length;
    prog.phase = `level ${hop}: regions`;
    prog.done = 0;
    prog.total = regions.length;
    report();
    coverageCells = new Set();
    // A region is prepared (prepareRegion) while the previous one fetches
    // its EM and finishes; `next` is the region after this one, if any.
    const prepared = new Map<string, Promise<Prepared>>();
    const prepare = (c: [number, number, number]) => {
      const key = c.join(",");
      let p = prepared.get(key);
      if (!p) {
        p = prepareRegion(c, object);
        p.catch(() => {}); // surfaced when the region is finished
        prepared.set(key, p);
      }
      return p;
    };
    const search = async (
      coord: [number, number, number],
      next?: [number, number, number],
    ) => {
      check();
      try {
        const prep = await prepare(coord);
        prepared.delete(coord.join(","));
        if (next) prepare(next);
        await finishRegion(prep, coord, hop);
      } catch (e) {
        prepared.delete(coord.join(","));
        if ((e as any)?.name === "AbortError") throw e;
        regionErrors++;
        logEvent("warn", `region ${coord.join(",")} (level ${hop}) failed`, e);
      }
      regionsSearched++;
      prog.done++;
      report();
    };

    for (let r = 0; r < regions.length; ++r) {
      prog.phase = `level ${hop}: regions`;
      prefetch(regions.slice(r, r + 1 + PREFETCH_REGIONS));
      await search(regions[r], regions[r + 1]);
    }
    // Coverage reads only what the regions themselves recorded, so the
    // scoring queue need not drain first; it drains before the next level.
    // Coverage, part 2: extra regions on whatever part of a searched L2 node
    // no band covered, then the same check on those, up to coveragePasses.
    // Contact sites are claimed per site, so overlap adds no duplicates.
    const centres = regions.map(regionCentre);
    for (let pass = 0; pass < P.coveragePasses; ++pass) {
      const G = COVERAGE_GRID;
      const pts = [...coverageCells].map(
        (k) =>
          k.split(",").map((v, d) => Number(v) * G[d] + (G[d] >> 1)) as [
            number,
            number,
            number,
          ],
      );
      coverageCells = new Set();
      const extra = coverageCentres(pts, centres);
      if (!extra.length) break;
      prog.total += extra.length;
      const coords = extra.map(
        (c) => [c[0] * 2, c[1] * 2, c[2]] as [number, number, number],
      );
      for (let r = 0; r < extra.length; ++r) {
        centres.push(extra[r]);
        coverageRegions++;
        regionsTotal++;
        prog.phase = `level ${hop}: coverage`;
        prefetch(coords.slice(r, r + 1 + PREFETCH_REGIONS));
        await search(coords[r], coords[r + 1]);
      }
    }
    await drainScoring();
    frontier = [...best.entries()]
      .filter(([, b]) => b.hop === hop && b.p >= opts.chainThreshold)
      .map(([id]) => id);
    for (const id of frontier) object.add(BigInt(id));
  }

  // Pairs that were scored but fall at or below the report threshold are not
  // rows; log them so the result accounts for every pair it looked at.
  for (const [id, b] of best) {
    if (b.p > P.reportThreshold) continue;
    rejected.below_threshold = (rejected.below_threshold ?? 0) + 1;
    rejectLog.push({
      target_segment_id: id,
      reason: "below_threshold",
      prob_merge: +b.p.toFixed(4),
      contact_voxels: b.row.contact_voxels,
      cand_voxels: b.row.cand_voxels,
    });
  }

  // 10. Result JSON, schema of run_twigcapture_v2.py.
  const rows = [...best.values()]
    .filter((b) => b.p > P.reportThreshold)
    .sort((a, b) => a.row.hop - b.row.hop || b.p - a.p);
  const candidates = rows.map((b, i) => {
    const { _emCentre, _reject, ...row } = b.row;
    _emCentre;
    _reject;
    return {
      id: `cand_${i}`,
      ...row,
      confidence: +b.p.toFixed(3),
      prob_merge: +b.p.toFixed(4),
      pred_local: +b.p.toFixed(2),
      pred_context: null,
      contact_sites_found: b.found,
      contact_sites_scored: b.scored,
      synapses_local: null,
      synapses_wide: null,
      cand_synapses_total: null,
      description:
        `browser: max over ${b.scored} site(s)` +
        (b.row.hop > 1 ? `; level ${b.row.hop}` : ""),
    };
  });
  const now = new Date();
  return {
    name: `browser_${root}_${stamp(now)}.json`,
    root_id: root,
    requested_root_id: root,
    timestamp: now.toISOString(),
    dataset: ds.datastack,
    mode: "browser",
    models_used: `local (in-browser, ${model.backend})`,
    local_model: opts.modelUrl,
    connectivity: 3,
    contact_gap_voxels: gap,
    contact_gap_mode: gapMode,
    decision_rule: "raw_local_prob",
    merge_threshold: P.reportThreshold,
    level: opts.level,
    chain_threshold: opts.chainThreshold,
    total_l2_nodes_searched: totalL2,
    binned_regions_searched: regionsSearched,
    binned_regions_total: regionsTotal,
    filters: {
      min_contact_voxels: P.minContactVoxels,
      max_contact_voxels: P.maxContactVoxels,
      max_contact_sites: opts.maxSites,
      min_contact_blob_voxels: P.minContactBlob,
      min_segment_voxels: P.minSegmentVoxels,
      max_candidate_l2_nodes: opts.maxCandidateL2 ?? P.maxCandidateL2,
      bin_size_xy: P.binXY,
      bin_size_z: P.binZ,
      coverage_passes: P.coveragePasses,
      max_em_zero_frac: P.maxEmZeroFrac,
      l2_max_dt_nm: P.l2MaxDtNm,
      l2_max_size_nm3: P.l2MaxSizeNm3,
      synapse_gate: "off (browser)",
      em_source: ds.emUrl,
      supervoxel_source: ds.supervoxelUrl,
    },
    rejected,
    rejected_candidates: rejectLog,
    candidate_count: candidates.length,
    candidates,
    benchmark: {
      total_seconds: +((performance.now() - t0) / 1000).toFixed(1),
      patches_scored: prog.patches,
      // Busy seconds per phase (see PhaseTimer); "sites" is the CPU work of
      // a region between its cutouts: roots -> labels, contacts, masks.
      phase_seconds: timer.seconds(),
      model_ms_per_patch: prog.patches
        ? +((timer.ms.model ?? 0) / prog.patches).toFixed(1)
        : null,
      model_runtime: model.runtime,
      region_errors: regionErrors,
      coverage_regions: coverageRegions,
      coverage_errors: coverageErrors,
      device: model.backend,
    },
  };
}

/**
 * At most `max` regions, spread over the whole neuron.
 *
 * Bins come in chunkedgraph leaf order, which is spatially clustered, so
 * taking the first `max` searched one end of a big neuron and none of the
 * rest. Farthest-point sampling instead: start from the first bin, then
 * repeatedly add the bin farthest (in nm, z scaled by its resolution as the
 * bins are) from every bin picked so far. Deterministic, and with `max` at
 * or above the count it returns every bin unchanged.
 */
export function spreadRegions(
  all: [number, number, number][],
  max: number,
): [number, number, number][] {
  if (all.length <= max) return all;
  if (max <= 0) return [];
  const n = all.length;
  // Bin coordinates are bin units (16 nm; FANC 17.2) in x/y and z slices;
  // compare in bin units,
  // which is what the search windows are sized in.
  const key = (c: [number, number, number]) => [
    c[0] / 70,
    c[1] / 70,
    c[2] / 28,
  ];
  const pts = all.map(key);
  const dist = new Float64Array(n).fill(Infinity);
  const picked: number[] = [0];
  const update = (j: number) => {
    const q = pts[j];
    for (let i = 0; i < n; ++i) {
      const d =
        (pts[i][0] - q[0]) ** 2 +
        (pts[i][1] - q[1]) ** 2 +
        (pts[i][2] - q[2]) ** 2;
      if (d < dist[i]) dist[i] = d;
    }
  };
  update(0);
  while (picked.length < max) {
    let best = -1;
    let bestD = -1;
    for (let i = 0; i < n; ++i) {
      if (dist[i] > bestD) {
        bestD = dist[i];
        best = i;
      }
    }
    if (best < 0 || bestD === 0) break;
    picked.push(best);
    update(best);
  }
  return picked.map((i) => all[i]);
}

/**
 * Region centres for a set of L2 nodes, one per bin, exactly as a run picks
 * them: backbone/soma nodes (max_dt or size above the runner's limits) are
 * skipped when `skipBackbone`, coordinates are in bin units (half the
 * supervoxel xy voxel: 16 nm, FANC 17.2 nm) and z slices, and the first node
 * in each bin is its centre. `svRes` is the dataset's supervoxelResolution.
 */
export function binL2(
  leaves: string[],
  attrs: Record<string, any>,
  svRes: [number, number, number],
  skipBackbone: boolean,
): [number, number, number][] {
  const P = TWIG_PARAMS;
  const binNm = svRes[0] / 2;
  const zRes = svRes[2];
  const bins = new Map<string, [number, number, number]>();
  for (const id of leaves) {
    const a = attrs[id];
    if (!a?.rep_coord_nm) continue;
    if (
      skipBackbone &&
      ((a.max_dt_nm ?? 0) > P.l2MaxDtNm ||
        Number(a.size_nm3 ?? 0) > P.l2MaxSizeNm3)
    )
      continue;
    const c: [number, number, number] = [
      Math.floor(a.rep_coord_nm[0] / binNm),
      Math.floor(a.rep_coord_nm[1] / binNm),
      Math.floor(a.rep_coord_nm[2] / zRes),
    ];
    const key = `${Math.floor(c[0] / P.binXY)},${Math.floor(c[1] / P.binXY)},${Math.floor(c[2] / P.binZ)}`;
    if (!bins.has(key)) bins.set(key, c);
  }
  return [...bins.values()];
}

/** The L2 nodes a run searches around: binL2's filter, without the binning. */
export function keptL2Ids(
  leaves: string[],
  attrs: Record<string, any>,
  skipBackbone: boolean,
): string[] {
  const P = TWIG_PARAMS;
  return leaves.filter((id) => {
    const a = attrs[id];
    if (!a?.rep_coord_nm) return false;
    return !(
      skipBackbone &&
      ((a.max_dt_nm ?? 0) > P.l2MaxDtNm ||
        Number(a.size_nm3 ?? 0) > P.l2MaxSizeNm3)
    );
  });
}

// ---- Coverage of each L2 node's full extent -------------------------------
//
// A region is centred on an L2 node's rep_coord and finds contacts only in
// its search band, P.border voxels in from each face of the cutout: +/-80
// supervoxel voxels in xy and +/-40 sections in z. An L2 node is not a
// point but a piece of the neuron filling up to one L2 chunk (256 x 256 x
// 128 voxels in Aedes, 256^3 in BANC), several times the band. Binning
// guaranteed that every kept rep_coord lies in some band, not that every
// kept node's voxels do, so contacts on the far side of a big L2 node were
// never searched, silently. Same bug and same fix as the cluster runner
// (twigcapture scripts/run_twigcapture_v2.py, "Coverage of each L2 node's
// full extent"): on GT, 3.0% (Aedes) and 5.3% (BANC) of touching merge
// sites lay outside every band; on a BANC FB4Z neuron the fix costs +25%
// regions and +13% time.

/** Out-of-band voxels are kept per cell of this size (supervoxel voxels). */
export const COVERAGE_GRID: [number, number, number] = [4, 4, 2];
/** Band shrink for a point to count as covered: cell size + contact reach. */
export const COVERAGE_MARGIN: [number, number, number] = [5, 5, 3];

/** Half-size of a region's search band, in supervoxel voxels. */
export function bandHalf(): [number, number, number] {
  const P = TWIG_PARAMS;
  const s = P.cutout;
  return [s[0] / 2 - P.border, s[1] / 2 - P.border, s[2] / 2 - P.border];
}

/** A region coord (bin units in xy, sections in z) -> its cutout centre. */
export function regionCentre(
  coord: [number, number, number],
): [number, number, number] {
  return [Math.floor(coord[0] / 2), Math.floor(coord[1] / 2), coord[2]];
}

/**
 * Greedy extra centres (supervoxel voxels) so that every point lies in some
 * centre's band, shrunk by `margin`. Points already covered by `centres`
 * add nothing. Deterministic, as the runner: points are visited in
 * lexicographic order and each new centre sits on the first one uncovered.
 */
export function coverageCentres(
  points: [number, number, number][],
  centres: [number, number, number][],
  margin: [number, number, number] = COVERAGE_MARGIN,
): [number, number, number][] {
  const h = bandHalf().map((v, d) => v - margin[d]);
  const inBand = (p: number[], c: number[]) =>
    p.every((v, d) => v >= c[d] - h[d] && v < c[d] + h[d]);
  const left = points
    .filter((p) => !centres.some((c) => inBand(p, c)))
    .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const out: [number, number, number][] = [];
  let todo = left;
  while (todo.length) {
    const c = todo[0];
    out.push(c);
    todo = todo.filter((p) => !inBand(p, c));
  }
  return out;
}

/** A neuron's size before running: L2 nodes, and level-1 regions a run would search. */
export async function neuronSize(
  graph: GraphAccess,
  root: string,
  svRes: [number, number, number],
): Promise<{ l2: number; regions: number }> {
  const leaves = await getLeavesL2(graph, root);
  const attrs = await getL2Attributes(graph, leaves);
  return {
    l2: leaves.length,
    regions: binL2(leaves, attrs, svRes, true).length,
  };
}
