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
 * - EM is the dataset's public 8 nm mip;
 * - synapse counts are not computed (no synapse tables in the browser).
 */

import type { TwigDataset } from "#src/layer/segmentation/twig_inference/datasets.js";
import { TWIG_PARAMS } from "#src/layer/segmentation/twig_inference/datasets.js";
import { logEvent } from "#src/layer/segmentation/twig_inference/diagnostics.js";
import type { MergeModel } from "#src/layer/segmentation/twig_inference/model.js";
import {
  IdTable,
  PrecomputedVolume,
} from "#src/layer/segmentation/twig_inference/precomputed_reader.js";
import type { Shape } from "#src/layer/segmentation/twig_inference/volume_ops.js";
import {
  PATCH_VOXELS,
  findContactSites,
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
  modelUrl: string;
}

export interface Progress {
  phase: string;
  done: number;
  total: number;
  patches: number;
  found: number;
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

/** Roots of many supervoxels (chunkedgraph roots_binary, as caveclient). */
async function getRoots(g: GraphAccess, svs: bigint[]): Promise<bigint[]> {
  const out: bigint[] = [];
  for (let i = 0; i < svs.length; i += 20000) {
    const part = BigUint64Array.from(svs.slice(i, i + 20000));
    let got: BigUint64Array | undefined;
    for (let attempt = 0; attempt < 2 && !got; ++attempt) {
      try {
        const r = await g.fetchGraph("/roots_binary", {
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
  const zRes = ds.supervoxelResolution[2];
  const t0 = performance.now();
  const prog: Progress = {
    phase: "L2 graph",
    done: 0,
    total: 0,
    patches: 0,
    found: 0,
  };
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
  const half = [16 * 32, 16 * 32, 9 * zRes];
  let regionErrors = 0;
  let regionsSearched = 0;
  let regionsTotal = 0;
  let totalL2 = 0;

  /** L2 nodes of `roots` -> binned region centres (first node per bin). */
  const regionsFor = async (roots: string[], skipBackbone: boolean) => {
    const bins = new Map<string, [number, number, number]>();
    for (const r of roots) {
      const leaves = await getLeavesL2(graph, r);
      totalL2 += leaves.length;
      check();
      const attrs = await getL2Attributes(graph, leaves);
      check();
      for (const id of leaves) {
        const a = attrs[id];
        if (!a?.rep_coord_nm) continue;
        if (
          skipBackbone &&
          ((a.max_dt_nm ?? 0) > P.l2MaxDtNm ||
            (a.size_nm3 ?? 0) > P.l2MaxSizeNm3)
        )
          continue;
        const c: [number, number, number] = [
          Math.floor(a.rep_coord_nm[0] / 16),
          Math.floor(a.rep_coord_nm[1] / 16),
          Math.floor(a.rep_coord_nm[2] / zRes),
        ];
        const key = `${Math.floor(c[0] / P.binXY)},${Math.floor(c[1] / P.binXY)},${Math.floor(c[2] / P.binZ)}`;
        if (!bins.has(key)) bins.set(key, c);
      }
    }
    return [...bins.values()];
  };

  /**
   * One region: every root in `object` counts as the neuron (merges already
   * accepted are simulated by relabelling them), and new neighbours are
   * proposed with hop = `hop`.
   */
  const processRegion = async (
    coord: [number, number, number],
    object: Set<bigint>,
    hop: number,
  ) => {
    // Supervoxel cutout, 200 x 200 x 120 at [32, 32, z].
    const g = [Math.floor(coord[0] / 2), Math.floor(coord[1] / 2), coord[2]];
    const origin = [g[0] - s[0] / 2, g[1] - s[1] / 2, g[2] - s[2] / 2];
    const svLab = (await ctx.sv.cutout(origin, s, signal)) as Uint32Array;
    check();
    const svIds = ctx.sv.idTable.ids;
    // Supervoxels -> roots (cached; sv roots do not change mid-run).
    const present = new Set<number>();
    for (let i = 0; i < svLab.length; ++i) present.add(svLab[i]);
    present.delete(0);
    const need = [...present]
      .map((l) => svIds[l])
      .filter((sv) => !ctx.svRoot.has(sv));
    if (need.length) {
      const roots = await getRoots(graph, need);
      need.forEach((sv, i) => ctx.svRoot.set(sv, roots[i]));
    }
    check();
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
    // Neighbours touching the (simulated) neuron in the centre box.
    const { keep, rejects } = touchingLabelsCenter(rootLab, s, target, {
      border: P.border,
      minContact: P.minContactVoxels,
      maxContact: P.maxContactVoxels,
      minSegment: P.minSegmentVoxels,
    });
    for (const [l, [why, ct, sz]] of rejects) {
      const id = String(rootTable.ids[l]);
      if (!best.has(id))
        reject(id, why, { contact_voxels: ct, cand_voxels: sz });
    }
    // A pair scored at an earlier level keeps that level's score.
    const cands = [...keep.keys()].filter((l) => {
      const b = best.get(String(rootTable.ids[l]));
      return !b || b.hop === hop;
    });
    // Candidate L2 sizes (cached per root), in parallel.
    await pool(cands, 8, async (l) => {
      const id = String(rootTable.ids[l]);
      if (!ctx.l2Count.has(id)) {
        try {
          ctx.l2Count.set(id, (await getLeavesL2(graph, id)).length);
        } catch {
          ctx.l2Count.set(id, 0);
        }
      }
    });
    check();
    // Contact sites -> patches.
    const built: SiteBuilt[] = [];
    const pending: Float32Array[] = [];
    const foundPerPair = new Map<string, number>();
    for (const l of cands) {
      const candId = String(rootTable.ids[l]);
      const stats = keep.get(l)!;
      const nL2 = ctx.l2Count.get(candId) ?? 0;
      if (nL2 > P.maxCandidateL2) {
        reject(candId, "candidate_l2", { ...stats, num_l2_nodes: nL2 });
        continue;
      }
      const sites = findContactSites(
        rootLab,
        s,
        target,
        l,
        P.minContactBlob,
        opts.maxSites,
      );
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
          (coord[0] + (c[0] - 100) * 2) * 16,
          (coord[1] + (c[1] - 100) * 2) * 16,
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
        const ctrNm = [gc[0] * 32, gc[1] * 32, gc[2] * zRes];
        let sinkRoot: bigint | undefined;
        const side = (lab: number) => {
          const pt = voxelNear(rootLab, s, c, lab);
          if (!pt) return null;
          const sv = svIds[svLab[pt[0] + s[0] * (pt[1] + s[1] * pt[2])]];
          if (lab === target) sinkRoot = ctx.svRoot.get(sv);
          return {
            supervoxel_id: String(sv),
            coordinate_nm: [
              (origin[0] + pt[0]) * 32,
              (origin[1] + pt[1]) * 32,
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
            cand_size_um3: +(
              (stats.cand_voxels * 32 * 32 * zRes) /
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
            _emCentre: [gc[0] * 4, gc[1] * 4, gc[2]],
          },
        });
        pending.push(x);
      });
      if (reason && !built.some((b) => b.pair === candId))
        reject(candId, reason, stats);
    }
    // EM for each built site; drop out-of-volume / empty windows.
    const scoreIdx: number[] = [];
    await pool(
      built.map((_, i) => i),
      6,
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
    // Score, and keep the best site per pair (max over sites).
    if (scoreIdx.length) {
      prog.phase = `level ${hop}: scoring on ${model.backend === "webgpu" ? "GPU" : "CPU"}`;
      report();
      const x = new Float32Array(scoreIdx.length * 3 * PATCH_VOXELS);
      scoreIdx.forEach((i, k) => x.set(pending[i], k * 3 * PATCH_VOXELS));
      const p = await model.score(x, scoreIdx.length);
      prog.patches += scoreIdx.length;
      scoreIdx.forEach((i, k) => {
        const b = built[i];
        const prev = best.get(b.pair);
        const scored = (prev?.scored ?? 0) + 1;
        const found = Math.max(prev?.found ?? 0, foundPerPair.get(b.pair) ?? 0);
        if (!prev || p[k] > prev.p)
          best.set(b.pair, { ...b, p: p[k], found, scored, hop });
        else Object.assign(prev, { found, scored });
      });
      prog.found = [...best.values()].filter(
        (b) => b.p > P.reportThreshold,
      ).length;
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
    const regions = all.slice(0, opts.maxRegions);
    regionsTotal += all.length;
    prog.phase = `level ${hop}: regions`;
    prog.done = 0;
    prog.total = regions.length;
    report();
    for (const coord of regions) {
      check();
      try {
        await processRegion(coord, object, hop);
      } catch (e) {
        if ((e as any)?.name === "AbortError") throw e;
        regionErrors++;
        logEvent("warn", `region ${coord.join(",")} (level ${hop}) failed`, e);
      }
      regionsSearched++;
      prog.done++;
      prog.phase = `level ${hop}: regions`;
      report();
    }
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
      max_candidate_l2_nodes: P.maxCandidateL2,
      bin_size_xy: P.binXY,
      bin_size_z: P.binZ,
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
      region_errors: regionErrors,
      device: model.backend,
    },
  };
}
