/**
 * @license
 * Copyright 2026 Kevin Delgado
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
 * Twig Capture: review automated merge proposals inside Neuroglancer.
 *
 * Reads the per-neuron JSON that scripts/run_twigcapture.py writes (one file per
 * target neuron, e.g. <root>_<stamp>.json),
 * lists the candidates it proposed, and lets a proofreader jump to each contact
 * site with both segments selected.
 *
 * A shared results directory is reached through a CATALOG: point the tab at a
 * directory served over HTTP and it offers type-ahead over the root ids it
 * holds; picking one fetches only that neuron's file. The directory needs to
 * send Access-Control-Allow-Origin, and it is read one of two ways:
 *
 *   index.json from scripts/make_results_index.py, preferred, because it
 *   carries candidate and review counts that cost a download to discover; or
 *
 *   the server's own autoindex, parsed for hrefs. Server-specific and
 *   count-free, but it means a directory is usable the moment nginx serves
 *   it, with nothing generated and no step for anyone to forget.
 *
 * The URL is remembered in localStorage, so a reviewer who overrides
 * DEFAULT_CATALOG_URL keeps their choice.
 *
 * The Merge button WRITES TO THE CHUNKEDGRAPH. It calls graphene's
 * mergeSegmentsWithOperation with the two supervoxels and positions that
 * run_twigcapture.py recorded, which is a real edit to shared data, attributed
 * to whoever clicked and visible to everyone. Undo is itself a further edit,
 * not a retraction. Treat every button in here as live.
 *
 * DECISIONS are written to the CAVE table twig_capture_decisions -- see
 * twig_capture_decisions.ts. "Not a merge" and "Unsure" record a judgement
 * and change nothing else; merging records `merge` WITH the operation id, so
 * every row claiming a merge is backed by a real edit. Decisions for the
 * loaded neuron are fetched on load and on Refresh, which is what stops ten
 * reviewers redoing each other's work. A failure to read or write them is
 * reported but never blocks reviewing: the table is a record of review, not a
 * prerequisite for it.
 */

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import type {
  DecisionRow,
  DecisionTarget,
  Verdict,
} from "#src/layer/segmentation/twig_capture_decisions.js";
import {
  DecisionStore,
  decisionsAllowed,
  isNotAMerge,
} from "#src/layer/segmentation/twig_capture_decisions.js";
import type {
  NeuronField,
  NeuronFilters,
  NeuronMeta,
} from "#src/layer/segmentation/twig_capture_neurons.js";
import {
  fetchNeuronMeta,
  fieldOptions,
  hasNeuronFilters,
  NEURON_FIELD_LABELS,
  NEURON_FIELDS,
  NEURON_META_FILE,
  neuronLabel,
  neuronTooltip,
  passesNeuronFilters,
} from "#src/layer/segmentation/twig_capture_neurons.js";
import type { CandidateTable } from "#src/layer/segmentation/twig_capture_table.js";
import {
  CANDIDATE_TABLE_FILE,
  fetchCandidateTable,
  passesProb,
} from "#src/layer/segmentation/twig_capture_table.js";
import type { TwigDataset } from "#src/layer/segmentation/twig_inference/datasets.js";
import { datasetForUrls } from "#src/layer/segmentation/twig_inference/datasets.js";
import {
  LOCAL_RESULT_PREFIX,
  type LocalResultJson,
} from "#src/layer/segmentation/twig_local_results.js";
import { StatusMessage } from "#src/status.js";
import { Uint64Set } from "#src/uint64_set.js";
import { removeChildren } from "#src/util/dom.js";
import { NullarySignal } from "#src/util/signal.js";
import { Tab } from "#src/widget/tab_view.js";

/** The layer JSON key this tab's state lives under. */
export const TWIG_CAPTURE_JSON_KEY = "twigCapture";

/**
 * The part of this tab worth putting in a shared link.
 *
 * It lives on the LAYER, not on the tab, because the tab is constructed
 * lazily (`getter: () => new TwigCaptureTab(this)`) and so does not exist
 * until a reviewer opens it -- while the layer's state has to serialise
 * whether or not anyone has looked.
 *
 * What is here, and why each earns its bytes:
 *
 *   catalogUrl   without it the recipient falls back to their own
 *                localStorage, and the link is not self-contained.
 *   file         the filename, not the root id: a neuron can have several
 *                runs, and a followed-forward root means the file may be
 *                named after a successor. This is what makes the tab
 *                non-empty.
 *   probOp       ">=" (default) or "<=": which side of minProb is shown.
 *                "<=" is for low-probability fragments that carry many
 *                synapses, which a reviewer may still want to look at.
 *   minProb, mito, review, sortKey
 *                four scalars that decide WHICH rows are shown and in what
 *                order. Share "the top one by orphan synapses" without the
 *                sort and the row you meant is forty down. mito and review
 *                are facets ("any" / ...); mito replaced an older pair of
 *                hideMito/mitoOnly booleans, which restoreState still reads
 *                so that links made before 2026-10-06 keep working.
 *   neuron       SeaTable filters on the neuron (super_class, cell_class,
 *                cell_type, status, flow -> chosen value), from the
 *                catalog's neurons.json. Narrow the all-neurons list and the
 *                root-id picker.
 *   candidate    the candidate's target_segment_id, so the row can be
 *                highlighted. The segment id and not `cand_22`, because the
 *                cand_N numbering shifts with ordering and threshold.
 *
 * What is deliberately ABSENT: the merge history and per-row merge status.
 * The chunkedgraph is the record of what has been merged, and the tab
 * re-derives it on load. Shipping a stale claim about what is already merged
 * is the one error here that could lead someone to a bad edit.
 *
 * The candidate list and the catalog are also absent: both are refetched,
 * and embedding them would bloat every link and go stale.
 */
export class TwigCaptureState {
  changed = new NullarySignal();
  catalogUrl = "";
  file = "";
  candidate = "";
  minProb = 0.75;
  /** ">=" | "<=": keep candidates at or above minProb, or at or below it. */
  probOp: ">=" | "<=" = ">=";
  /** "any" | "hide" | "only" -- replaced the hideMito/mitoOnly pair. */
  mito = "any";
  /** "any" | "none" | "done" | or one of the verdicts. */
  review = "any";
  sortKey = "prob";
  /** SeaTable neuron filters: field -> value; absent means any. */
  neuron: NeuronFilters = {};

  toJSON() {
    // Only the fields that differ from a fresh tab, so an untouched layer
    // adds nothing to the state. Neuroglancer links get pasted into chat
    // windows and issue trackers; every key costs.
    const x: { [k: string]: any } = {};
    if (this.catalogUrl) x.catalogUrl = this.catalogUrl;
    if (this.file) x.file = this.file;
    if (this.candidate) x.candidate = this.candidate;
    if (this.minProb !== 0.75) x.minProb = this.minProb;
    if (this.probOp !== ">=") x.probOp = this.probOp;
    if (this.mito !== "any") x.mito = this.mito;
    if (this.review !== "any") x.review = this.review;
    if (this.sortKey !== "prob") x.sortKey = this.sortKey;
    if (hasNeuronFilters(this.neuron)) {
      const n: { [k: string]: string } = {};
      for (const f of NEURON_FIELDS) if (this.neuron[f]) n[f] = this.neuron[f]!;
      x.neuron = n;
    }
    return Object.keys(x).length ? x : undefined;
  }

  /**
   * Restore, tolerating anything. A link can outlive the file it names, the
   * directory it was served from, or a field's type; none of that may throw,
   * because a restoreState that throws takes the whole viewer down rather
   * than one tab.
   */
  restoreState(obj: unknown) {
    if (obj === undefined || obj === null || typeof obj !== "object") return;
    const o = obj as { [k: string]: any };
    if (typeof o.catalogUrl === "string") this.catalogUrl = o.catalogUrl;
    if (typeof o.file === "string") this.file = o.file;
    if (typeof o.candidate === "string") this.candidate = o.candidate;
    if (typeof o.minProb === "number" && Number.isFinite(o.minProb))
      this.minProb = o.minProb;
    this.probOp = o.probOp === "<=" ? "<=" : ">=";
    // Links made before 2026-10-06 carry the hideMito/mitoOnly booleans, and
    // those links are already pasted into chats and tickets. Read them, then
    // let the new key win if both somehow appear.
    this.mito =
      o.hideMito === true ? "hide" : o.mitoOnly === true ? "only" : "any";
    if (["any", "hide", "only"].includes(o.mito)) this.mito = o.mito;
    this.review = [
      "any",
      "none",
      "done",
      "merge",
      "no_merge",
      "unsure",
      "new_neuron",
      "split_needed",
    ].includes(o.review)
      ? o.review
      : "any";
    if (["prob", "size", "contact", "synapses"].includes(o.sortKey))
      this.sortKey = o.sortKey;
    // Only known fields with string values; anything else is dropped.
    const neuron: NeuronFilters = {};
    if (o.neuron !== null && typeof o.neuron === "object") {
      for (const f of NEURON_FIELDS) {
        const v = o.neuron[f];
        if (typeof v === "string" && v !== "") neuron[f] = v;
      }
    }
    this.neuron = neuron;
    this.changed.dispatch();
  }
}

/**
 * The seed of an orphan-first result: the result's root id (seed_id), which
 * is what the sink side of every hop grew from. Falls back to the
 * candidate's parent_root.
 */
function ref0Seed(
  active: { root_id: string } | undefined,
  candidate: TwigCandidate,
): string {
  return active?.root_id ?? candidate._root ?? candidate.parent_root ?? "";
}

/** One candidate as run_twigcapture.py emits it. */
interface TwigCandidate {
  id: string;
  target_segment_id: string;
  /**
   * Orphan-first results only (run_twigcapture_v2_orph.py): the root the
   * candidate joins, i.e. the growing seed object (the merge_sink side), and
   * the hop it was found on. Their presence is what marks a candidate as
   * orphan-first.
   */
  parent_root?: string;
  /**
   * In-browser results with level > 1 (twig_inference/pipeline.ts): the
   * earlier suggestion this candidate attaches to, when that is not the
   * neuron itself. Its merge_sink lies on via_root.
   */
  via_root?: string;
  hop?: number;
  confidence?: number;
  prob_merge: number;
  pred_local?: number;
  pred_context?: number;
  coordinate_nm: [number, number, number];
  num_l2_nodes: number;
  // added 2026-09-23; absent in files written before then
  contact_voxels?: number | null;
  cand_voxels?: number | null;
  cand_size_um3?: number | null;
  segment_size_um3?: number | null;
  segment_max_dt_nm?: number | null;
  em_zero_frac?: number | null;
  enclosure?: number | null;
  cand_em_mean?: number | null;
  likely_mito?: boolean;
  description?: string;
  // added 2026-09-24: what the chunkedgraph /merge endpoint needs. Supervoxels
  // are immutable, so these stay valid however much the graph is edited --
  // unlike target_segment_id, which is a root as of inference time.
  merge_sink?: MergeEndpoint | null;
  merge_source?: MergeEndpoint | null;
  // added 2026-09-30. Null means NOT COUNTED, which is a different fact from
  // a zero count: null when the run had no synapse index. Files written
  // before 2026-09-30 omit these keys entirely.
  synapses_local?: SynapseCounts | null;
  /**
   * The wide counting window, sized by filtering.synapse_count_box_nm.
   *
   * Called synapses_context in files written on 2026-09-30 before the window
   * was decoupled from which models run; read both so those still display.
   */
  synapses_wide?: SynapseCounts | null;
  synapses_context?: SynapseCounts | null;
  /**
   * Synapses on the candidate's WHOLE extent, not just a counted window.
   *
   * This is the number a merge actually adds to the completion rate, and it
   * is what ranking should use. The windowed counts above see a mean of 76%
   * of a fragment's synapses and correlate at Spearman 0.70 with the true
   * total -- fine as a filter, too noisy as an ordering. Null in runs
   * without a supervoxel_synapses table.
   */
  cand_synapses_total?: number | null;
  /**
   * Which result file this candidate came from, set on load (parseResult) or
   * by the all-neurons table. `id` (cand_N) is unique only within one file,
   * so per-row state is keyed by file and id together -- see rowKey().
   */
  _file?: string;
  /** The neuron's root as recorded in that file (inference time). */
  _root?: string;
  /**
   * A row from the all-neurons table: display fields only. Acting on it
   * fetches the neuron's file for the full record (ensureFull).
   */
  _lite?: boolean;
}

/** Per-row state key: cand_N is only unique within one result file. */
function rowKey(c: TwigCandidate): string {
  return `${c._file ?? ""}#${c.id}`;
}

/** Is `root` one of the staged neurons (not a candidate)? */
function isStagedNeuron(
  stage: { neurons: Map<string, bigint> },
  root: bigint,
): boolean {
  for (const r of stage.neurons.values()) if (r === root) return true;
  return false;
}

/** The result files a list of rows comes from, in order of first appearance. */
function rowFiles(rows: TwigCandidate[], fallback: { name: string }): string[] {
  return [...new Set(rows.map((c) => c._file ?? fallback.name))];
}

/**
 * How one merge attempt ended. `skipped` is a staged run declining a row it
 * would otherwise have to ask about; `cancelled` is a reviewer saying no.
 */
type MergeOutcome = "merged" | "already" | "skipped" | "failed" | "cancelled";

/** A full candidate record together with the result file it belongs to. */
interface RowRef {
  result: TwigResult;
  candidate: TwigCandidate;
}

/**
 * The all-neurons view renders this many rows at a time; "Show next" adds
 * another page. Rows are DOM nodes with listeners, and BANC alone has 1.56M
 * candidates, so the list is paged rather than drawn whole.
 */
const ALL_NEURONS_PAGE = 100;

/**
 * Real synapses inside one scoring window, attributed by supervoxel.
 *
 * `total` is every synapse in the box from any neuron, so the four
 * attributed fields do not sum to it -- most of what is in the box belongs
 * to neither of these two segments.
 */
interface SynapseCounts {
  total: number;
  target_pre: number;
  target_post: number;
  cand_pre: number;
  cand_post: number;
  /** Synapses FROM one of this pair TO the other. Argues against merging. */
  between: number;
}

interface MergeEndpoint {
  supervoxel_id: string;
  coordinate_nm: [number, number, number];
}

/**
 * The slice of graphene's GrapheneGraphSource this tab uses.
 *
 * Probed structurally rather than imported: the abstract SegmentationGraphSource
 * exposes merge(a, b), but graphene's implementation of it is a stub that
 * returns 0n and performs no edit. The working path is graphServer, and a
 * non-graphene graph simply will not have it.
 */
interface GrapheneLike {
  getRoot?(id: bigint, timestamp?: number): Promise<bigint>;
  graphServer?: {
    mergeSegmentsWithOperation(
      first: { segmentId: bigint; rootId: bigint; position: Float32Array },
      second: { segmentId: bigint; rootId: bigint; position: Float32Array },
    ): Promise<{ newRoot: bigint; operationId: string | undefined }>;
    undoOperation(operationId: string): Promise<{
      newRoots: bigint[];
      operationId: string | undefined;
    }>;
  };
  state?: { replaceSegments(oldValues: Uint64Set, newValues: Uint64Set): void };
}

/** One row of index.json, as scripts/make_results_index.py writes it. */
interface CatalogEntry {
  file: string;
  root_id: string;
  timestamp?: string;
  candidate_count?: number;
  n_review?: number;
  n_likely_mito?: number;
}

interface TwigResult {
  name: string;
  root_id: string;
  timestamp?: string;
  dataset?: string;
  models_used?: string;
  merge_threshold?: number;
  connectivity?: number;
  candidate_count?: number;
  candidates: TwigCandidate[];
  rejected?: Record<string, number>;
  /**
   * The wide counting box in nm, from filters.synapse_box_wide_nm.
   *
   * Carried so the tooltip can state the real range rather than a hardcoded
   * "4 um": the box is a config value (filtering.synapse_count_box_nm) and
   * is expected to widen.
   */
  synapse_box_wide_nm?: [number, number, number];
  /**
   * Set on a combined list: the result files of the neurons it joins. Each
   * row keeps its own file (_file) and neuron (_root); the graph check,
   * decisions and staging work per member neuron.
   */
  members?: string[];
}

/** One list of the candidates of several neurons' results. */
function combineResults(results: TwigResult[]): TwigResult {
  const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
  return {
    name: `${LOCAL_RESULT_PREFIX}multi_${results.length}_${stamp}.json`,
    root_id: results.map((r) => r.root_id).join(","),
    timestamp: new Date().toISOString(),
    dataset: results[0]?.dataset,
    models_used: [...new Set(results.map((r) => r.models_used))].join("; "),
    merge_threshold: results[0]?.merge_threshold,
    candidates: results.flatMap((r) => r.candidates),
    members: results.map((r) => r.name),
  };
}

function parseResult(name: string, text: string): TwigResult {
  // run_twigcapture writes NaN for the unused branch, which JSON.parse rejects.
  // Only replace bare NaN tokens in value position, never inside a string.
  const cleaned = text.replace(/:\s*NaN\s*([,}])/g, ": null$1");
  return parseResultObject(name, JSON.parse(cleaned));
}

/**
 * A parsed result object -> TwigResult. Also the entry point for results
 * computed in this browser (twig_local_results.ts), which are already
 * objects. Candidates are copied so the stored object is not stamped.
 */
function parseResultObject(name: string, obj: any): TwigResult {
  if (!Array.isArray(obj.candidates)) {
    throw new Error(`${name}: no "candidates" array -- not an AMP result file`);
  }
  // Orphan-first files (run_twigcapture_v2_orph.py) name their root seed_id.
  const rootId = String(obj.root_id ?? obj.seed_id ?? "unknown");
  const candidates = (obj.candidates as TwigCandidate[]).map((c) => ({
    ...c,
    _file: name,
    _root: rootId,
  }));
  return {
    name,
    root_id: rootId,
    timestamp: obj.timestamp,
    dataset: obj.dataset,
    models_used: obj.models_used,
    merge_threshold: obj.merge_threshold,
    connectivity: obj.connectivity,
    candidate_count: obj.candidate_count,
    candidates,
    rejected: obj.rejected,
    synapse_box_wide_nm: obj.filters?.synapse_box_wide_nm ?? undefined,
  };
}

// The catalog URL is per-browser, not per-link: a lab points every reviewer at
// the same served directory once. localStorage can throw (private windows,
// blocked site data), so every access is guarded and an empty string is a
// perfectly good fallback.
const CATALOG_URL_KEY = "neuroglancer-twig-capture-catalog-url";

/**
 * Where results live unless a reviewer has pointed this browser elsewhere.
 *
 * Aedes only for now. When BANC results are published this stops being a
 * constant and becomes a per-dataset choice -- the two cannot share a
 * directory, because a candidate's ids are only meaningful in its own
 * datastack.
 */
/**
 * Version of THIS TAB, shown in its footer. Bump it with any change a
 * reviewer could notice.
 *
 * It is here rather than derived from the build because the thing a reviewer
 * needs to quote in a bug report is "which Twig Capture", not "which
 * Neuroglancer" -- this tab moves far faster than the fork around it, and a
 * git hash tells them nothing they can say out loud.
 *
 * v1.0 (2026-09-30): first version fit to hand to someone else. Catalog from
 * a served directory, per-candidate synapse counts, merge with undo/redo, the
 * already-merged and changed-hands checks on load.
 *
 * v1.1 (2026-11-21): an all-neurons view when no root id is chosen (every
 * candidate in the catalog that passes the filters, from candidates.json.gz,
 * with merge and verdicts inline), and a p >= / p <= toggle.
 *
 * v1.2 (2026-11-21): Stage all -> review -> Merge staged, one neuron at a
 * time, replacing the never-enabled Merge all; and optional neuron filters
 * (super class, class, type, status, flow) from the catalog's neurons.json,
 * a cache of the project's SeaTable.
 */
const TWIG_CAPTURE_VERSION = "v1.3";
const TWIG_CAPTURE_DATE = "20261121";

const DEFAULT_CATALOG_URL =
  "https://catmaid2.hms.harvard.edu/twigcapture_aedes/";

/**
 * The remembered catalog, per dataset: a reviewer who works on BANC and on
 * Aedes must not have one's catalog carried over to the other. Falls back to
 * the dataset's own catalog, then to the old shared key and default.
 */
function readStoredCatalogUrl(dataset?: TwigDataset): string {
  const fallback = dataset?.catalogUrl ?? DEFAULT_CATALOG_URL;
  try {
    if (dataset !== undefined) {
      return (
        localStorage.getItem(`${CATALOG_URL_KEY}:${dataset.key}`) ?? fallback
      );
    }
    return localStorage.getItem(CATALOG_URL_KEY) ?? fallback;
  } catch {
    return fallback;
  }
}

function storeCatalogUrl(url: string, dataset?: TwigDataset) {
  const key =
    dataset !== undefined
      ? `${CATALOG_URL_KEY}:${dataset.key}`
      : CATALOG_URL_KEY;
  try {
    if (url) localStorage.setItem(key, url);
    else localStorage.removeItem(key);
  } catch {
    // no persistence available; the tab still works for this session
  }
}

/**
 * Catalog entries recovered from a plain directory listing.
 *
 * Parses the HTML an autoindex serves, taking each link's HREF and never its
 * text: nginx truncates the displayed filename to ~50 characters and appends
 * "..&gt;", so the text of a link to
 * a long result filename reads "648518347555941346_2026093...>" with the
 * extension cut off. The href is always complete.
 *
 * Root id and timestamp come from the filename, which run_twigcapture.py
 * builds as <root>_<YYYYmmdd>_<HHMMSS>.json. Anything not matching that
 * shape is ignored rather than guessed at, which is also what keeps
 * index.json out of the list.
 *
 * candidate_count is left undefined: a listing cannot know it without
 * downloading every file, and the row renders "? cands" rather than a lie.
 * Sorted newest-first within a root, matching what make_results_index.py
 * guarantees, because openFromCatalog() takes the first hit for a root.
 */
async function fetchDirectoryListing(base: string): Promise<CatalogEntry[]> {
  const response = await fetch(base);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const html = await response.text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  const entries: CatalogEntry[] = [];
  const seen = new Set<string>();
  for (const a of Array.from(doc.querySelectorAll("a[href]"))) {
    const href = a.getAttribute("href") ?? "";
    // Take the last path component: Apache emits absolute hrefs, nginx
    // relative ones, and a query string is used for column sorting.
    const file = decodeURIComponent(href.split("?")[0].split("/").pop() ?? "");
    // <root>_<YYYYmmdd>_<HHMMSS>.json as run_twigcapture.py writes it, plus
    // the amp_results_ prefix it used before 2026-09-30, because directories
    // written under the old name are still served.
    const m = /^(?:amp_results_)?(\d+)_(\d{8})_(\d{6})\.json$/.exec(file);
    if (m === null || seen.has(file)) continue;
    seen.add(file);
    entries.push({
      file,
      root_id: m[1],
      timestamp:
        `${m[2].slice(0, 4)}-${m[2].slice(4, 6)}-${m[2].slice(6, 8)}` +
        `T${m[3].slice(0, 2)}:${m[3].slice(2, 4)}:${m[3].slice(4, 6)}`,
    });
  }
  entries.sort((a, b) =>
    a.root_id === b.root_id
      ? (b.timestamp ?? "").localeCompare(a.timestamp ?? "")
      : a.root_id.localeCompare(b.root_id),
  );
  return entries;
}

/**
 * Synapses carried by the CANDIDATE, for ranking.
 *
 * pre + post, on the candidate only -- the target's own synapses say nothing
 * about whether this twig is worth attaching, and `total` is the neuropil's
 * density, which the search already filtered on.
 *
 * Prefers the wider window where a run has both: the local box is ~1 um^3
 * and reads zero for the great majority of candidates, so ranking by it
 * sorts almost nothing. Absent counts rank as 0, which puts files with no
 * synapse index at the bottom rather than the top.
 */
function candSynapses(c: TwigCandidate): number {
  // The whole-fragment count when the run has it. Falling back to a window
  // is a last resort: the wide box sees a mean of 76% of a fragment's
  // synapses, so ranking by it demotes exactly the fragments that extend
  // furthest -- which tend to be the ones worth attaching.
  if (c.cand_synapses_total != null) return c.cand_synapses_total;
  const w = c.synapses_wide ?? c.synapses_context ?? c.synapses_local;
  return w === null || w === undefined ? 0 : w.cand_pre + w.cand_post;
}

function fmt(x: number | null | undefined, digits = 3): string {
  return x === null || x === undefined || Number.isNaN(x)
    ? "–"
    : x.toFixed(digits);
}

export class TwigCaptureTab extends Tab {
  private results: TwigResult[] = [];
  private active: TwigResult | undefined;

  private fileSelect = document.createElement("select");
  private rootInput = document.createElement("input");
  private rootList = document.createElement("datalist");
  private catalogStatus = document.createElement("span");
  private catalog: CatalogEntry[] = [];
  private catalogUrl = "";
  private mitoSelect: HTMLSelectElement | undefined;
  private probOpSelect: HTMLSelectElement | undefined;
  private reviewSelect: HTMLSelectElement | undefined;
  private mergeAllButton = document.createElement("button");
  /** candidate id -> outcome, so a row is not offered twice. */
  private mergeStatus = new Map<
    string,
    "merging" | "done" | "failed" | "already" | "skipped"
  >();
  /** Why a staged merge was skipped rather than made, by rowKey. */
  private skipReason = new Map<string, string>();
  private refreshButton = document.createElement("button");

  /**
   * Decisions already recorded in CAVE for the active neuron, keyed by
   * target_segment_id. Refilled on load and on Refresh, so a reviewer sees
   * what other people have already judged instead of redoing it.
   */
  /**
   * Which result file the list was last rendered for, so render() can tell a
   * re-render of the SAME list from a switch to a different one.
   */
  private lastRenderedFile: string | undefined;
  private decisions = new Map<string, DecisionRow>();
  private decisionStore: DecisionStore | undefined;
  /** Rows (rowKey) whose decision POST is in flight, so the row can show it. */
  private deciding = new Set<string>();
  /**
   * Each neuron's root TODAY, by result file name; changes with every merge
   * in this tab. Per file rather than one value because the all-neurons view
   * merges into many neurons.
   */
  private targetRoots = new Map<string, bigint>();

  // ---- all-neurons view ----------------------------------------------------
  /** The catalog's candidates.json.gz, once fetched. */
  private table: CandidateTable | undefined;
  /** Which catalog `table` (and tableState) belong to. */
  private tableFor = "";
  private tableState: "none" | "loading" | "ready" | "missing" | "error" =
    "none";
  private tableError = "";
  /** Every decision in the table, by candidate id (all-neurons view). */
  private allDecisions = new Map<string, DecisionRow>();
  /** Bumped whenever allDecisions changes; part of the filter cache key. */
  private allDecisionsVersion = 0;
  /** Rows of the filtered all-neurons list currently rendered. */
  private allShown = ALL_NEURONS_PAGE;
  private allFiltered:
    | { table: CandidateTable; key: string; rows: TwigCandidate[] }
    | undefined;
  /** Full result files fetched to act on table rows, by file name. */
  private fileCache = new Map<string, Promise<TwigResult>>();
  private allButton = document.createElement("button");

  // ---- SeaTable neuron filters ----------------------------------------------
  /** The catalog's neurons.json, once fetched; undefined when it has none. */
  private neuronMeta: NeuronMeta | undefined;
  /** Which catalog neuronMeta belongs to. */
  private neuronMetaFor = "";
  private neuronFilters: NeuronFilters = {};
  private neuronRow = document.createElement("div");

  // ---- stage, review, merge staged ------------------------------------------
  /**
   * Candidates of ONE neuron staged for a sequential merge. `rows` is frozen
   * when staged -- what is listed is exactly what Merge staged will act on,
   * whatever the filters do meanwhile -- and `selected` is what is still
   * ticked. `roots` holds each candidate's root today, for showing it, and
   * `neuron` the neuron's, which stays shown whatever is unticked.
   */
  private stage:
    | {
        file: string;
        rows: TwigCandidate[];
        selected: Set<string>;
        roots: Map<string, bigint>;
        /**
         * Each staged neuron's root today, by its result file. One entry for
         * a single neuron; several when a combined list (several neurons
         * opened together from the Twig Selector) is staged.
         */
        neurons: Map<string, bigint>;
      }
    | undefined;
  /** A Merge staged run in progress. */
  private batch: { stop: boolean; done: number; total: number } | undefined;
  private clearStageButton = document.createElement("button");
  /**
   * The staged list and the 3D view are kept in step, both ways. Pointing at
   * a staged fragment in the view marks its row (and scrolls it into sight);
   * hiding a fragment in the view (double-click, or unticking it in the
   * segment list) unticks its row, and showing it again ticks it.
   * `stageRowEls` is the rows as last drawn, by rowKey; `hoverKey` the row
   * last pointed at -- sticky, so it survives the mouse crossing from the
   * view to the panel. `ownSegmentEdits` is > 0 while the tab itself is
   * changing the segment sets, which must not read as the reviewer's choice.
   */
  private stageRowEls = new Map<string, HTMLElement>();
  private hoverKey: string | undefined;
  private ownSegmentEdits = 0;
  private stageRenderPending = false;
  /**
   * Progress of the check that runs on load, or undefined when not running.
   *
   * While it is set the candidate list is WITHHELD rather than shown and
   * then corrected: every row's merge button is about to change meaning --
   * "Merge" becoming "In neuron" for anything already done -- and a list
   * that rewrites itself under the cursor invites a click on a row whose
   * state was stale when it was drawn.
   */
  private checking: { done: number; total: number } | undefined;
  /** Whether the shared link's row has already been scrolled to. */
  private scrolledToShared = false;
  /**
   * Operation id per candidate, and which way round it currently stands.
   *
   * `undone` is what makes redo possible: an undo is itself an operation, so
   * undoing THAT re-applies the merge. The graph only ever moves forward --
   * nothing is erased from the log -- so this is a pointer into that history,
   * not a buffer of reversible state.
   */
  private history = new Map<string, { operationId: string; undone: boolean }>();
  private summary = document.createElement("div");
  private listEl = document.createElement("div");
  private countEl = document.createElement("div");

  // filters
  private minProb = 0.75;
  private probOp: ">=" | "<=" = ">=";
  private mito = "any";
  private review = "any";
  private sortKey: "prob" | "size" | "contact" | "synapses" = "prob";

  /**
   * Copy the tab's shareable fields onto the layer, so a link carries them.
   *
   * Called after anything a recipient would need to see the same thing. The
   * layer dispatches specificationChanged, which is what makes the address
   * bar and the Share button pick it up.
   */
  private publish() {
    const t = this.layer.twigCapture;
    t.catalogUrl = this.catalogUrl;
    t.file = this.active?.name ?? "";
    t.minProb = this.minProb;
    t.probOp = this.probOp;
    t.mito = this.mito;
    t.review = this.review;
    t.sortKey = this.sortKey;
    t.neuron = { ...this.neuronFilters };
    t.changed.dispatch();
  }

  constructor(public layer: SegmentationUserLayer) {
    super();
    // Seed from the layer BEFORE the controls are built, so they render with
    // the shared link's values rather than snapping to them afterwards.
    const saved = layer.twigCapture;
    this.minProb = saved.minProb;
    this.probOp = saved.probOp;
    this.mito = saved.mito;
    this.review = saved.review;
    this.sortKey = saved.sortKey as typeof this.sortKey;
    this.neuronFilters = { ...saved.neuron };
    const { element } = this;
    element.classList.add("neuroglancer-twig-capture-tab");
    // Results computed in this browser (Twig Selector). The store is on the
    // layer because this tab is disposed whenever the panel hides.
    const local = layer.localTwigResults;
    this.registerDisposer(local.changed.add(() => this.syncLocal()));
    void local.ready.then(() => {
      if (!this.wasDisposed) this.syncLocal();
    });
    element.appendChild(this.makeCatalogRow());
    element.appendChild(this.makeResultSelectRow());
    element.appendChild(this.makeFilterRow());
    this.neuronRow.classList.add("neuroglancer-twig-capture-filters");
    this.neuronRow.classList.add("neuroglancer-twig-capture-neuron-filters");
    this.neuronRow.style.display = "none";
    element.appendChild(this.neuronRow);
    this.summary.classList.add("neuroglancer-twig-capture-summary");
    element.appendChild(this.summary);
    const countRow = document.createElement("div");
    countRow.classList.add("neuroglancer-twig-capture-countrow");
    this.countEl.classList.add("neuroglancer-twig-capture-count");
    countRow.appendChild(this.countEl);
    this.refreshButton.classList.add("neuroglancer-twig-capture-merge");
    this.refreshButton.textContent = "Refresh";
    this.refreshButton.title =
      "Re-resolve every listed candidate against the live graph and flag any " +
      "that are already part of this neuron. Does not recompute contact, " +
      "enclosure or EM -- those come from the inference run.";
    this.refreshButton.addEventListener(
      "click",
      () => void this.refreshAgainstGraph(),
    );
    countRow.appendChild(this.refreshButton);
    // BULK MERGING, IN TWO DELIBERATE STEPS, ONE NEURON AT A TIME.
    //
    // "Stage all" edits nothing: it freezes the listed candidates of the
    // open neuron into a ticked list and shows them in the view, so the
    // reviewer sees every proposed join at once and unticks the wrong ones.
    // Only then does the same button read "Merge staged", which merges the
    // ticked ones sequentially -- each one re-resolved and re-checked first,
    // because every merge changes the neuron's root. While that runs the
    // button is "Stop". Never across neurons: a bulk edit to many neurons at
    // once is too much to have checked by eye.
    this.mergeAllButton.classList.add("neuroglancer-twig-capture-merge");
    this.mergeAllButton.textContent = "Stage all";
    this.mergeAllButton.addEventListener("click", () => {
      if (this.batch !== undefined) {
        this.batch.stop = true;
        this.updateBulkButtons();
      } else if (this.stage !== undefined) {
        void this.mergeStaged();
      } else {
        void this.stageAll();
      }
    });
    countRow.appendChild(this.mergeAllButton);
    this.clearStageButton.classList.add("neuroglancer-twig-capture-merge");
    this.clearStageButton.textContent = "Clear";
    this.clearStageButton.title =
      "Drop the staged list and go back to the filtered candidates. Nothing " +
      "is edited; merges already made stay.";
    this.clearStageButton.style.display = "none";
    this.clearStageButton.addEventListener("click", () => this.clearStage());
    countRow.appendChild(this.clearStageButton);
    element.appendChild(countRow);
    this.listEl.classList.add("neuroglancer-twig-capture-list");
    element.appendChild(this.listEl);

    const footer = document.createElement("div");
    footer.classList.add("neuroglancer-twig-capture-version");
    footer.textContent = `Twig Capture ${TWIG_CAPTURE_VERSION} \u00b7 ${TWIG_CAPTURE_DATE}`;
    footer.title =
      "Quote this when reporting anything odd. Merges go to the " +
      "chunkedgraph and are attributed to you; Undo is a further edit, not " +
      "a retraction.";
    element.appendChild(footer);

    const display = layer.displayState;
    this.registerDisposer(
      display.segmentSelectionState.changed.add(() => this.onViewHover()),
    );
    this.registerDisposer(
      display.segmentationGroupState.value.visibleSegments.changed.add(
        (x, add) => this.onVisibleChanged(x, add),
      ),
    );

    this.render();
  }

  private makeCatalogRow(): HTMLElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-twig-capture-controls");

    const urlInput = document.createElement("input");
    urlInput.type = "text";
    urlInput.classList.add("neuroglancer-twig-capture-url");
    urlInput.placeholder = "results directory URL (serves index.json)";
    urlInput.title =
      "A directory of result .json files served over HTTP, containing an " +
      "index.json from scripts/make_results_index.py. Remembered in this browser.";
    // A link's catalog wins over this browser's remembered one: the point of
    // putting it in the state is that the link is self-contained.
    urlInput.value = this.catalogUrl =
      this.layer.twigCapture.catalogUrl ||
      readStoredCatalogUrl(datasetForUrls(this.sourceUrls()));
    const connect = () => {
      const value = urlInput.value.trim();
      this.catalogUrl = value;
      this.publish();
      storeCatalogUrl(value, datasetForUrls(this.sourceUrls()));
      if (value) void this.loadCatalog(value);
    };
    urlInput.addEventListener("keydown", (event: KeyboardEvent) => {
      if (event.key === "Enter") connect();
    });
    urlInput.addEventListener("blur", () => {
      if (urlInput.value.trim() !== this.catalogUrl) connect();
    });

    const button = document.createElement("button");
    button.textContent = "Connect";
    button.addEventListener("click", connect);

    // Re-fetch the catalog.
    //
    // loadCatalog otherwise runs ONCE, when this row is built, so a tab left
    // open showed whatever index.json held at page load. Results arriving
    // from a running batch were invisible until the whole viewer was
    // reloaded, which looked like make_results_index.py had not run -- it
    // had; nothing was asking for it again.
    //
    // cache: "reload" because index.json is served with an ETag and no
    // Cache-Control, so an ordinary fetch may revalidate and the browser can
    // hand back the copy it already has. A refresh button that returns the
    // same stale list is worse than no button.
    const reload = document.createElement("button");
    reload.textContent = "↻";
    reload.title =
      "Re-read index.json. New results from a running batch do " +
      "not appear on their own.";
    reload.addEventListener("click", () => {
      if (this.catalogUrl) void this.loadCatalog(this.catalogUrl, true);
    });

    this.catalogStatus.classList.add("neuroglancer-twig-capture-status");

    row.appendChild(urlInput);
    row.appendChild(button);
    row.appendChild(reload);
    row.appendChild(this.catalogStatus);

    // Type-ahead over root ids. A datalist gives native filtering as the
    // reviewer types, with no keyboard handling of our own to get wrong.
    const search = document.createElement("div");
    search.classList.add("neuroglancer-twig-capture-controls");
    this.rootInput.type = "text";
    this.rootInput.classList.add("neuroglancer-twig-capture-url");
    this.rootInput.placeholder = "type a root id\u2026";
    this.rootInput.disabled = true;
    this.rootList.id = `neuroglancer-twig-roots-${Math.random().toString(36).slice(2)}`;
    this.rootInput.setAttribute("list", this.rootList.id);
    const open = () => this.openFromCatalog(this.rootInput.value.trim());
    // Open as soon as a suggestion is PICKED, but not while typing or pasting.
    //
    // Picking from a datalist fires `input` at once but defers `change` until
    // the field is committed -- usually a blur. Bound to `change` alone,
    // clicking a suggestion did nothing until the reviewer clicked somewhere
    // else, which reads as the tab being broken rather than a focus rule.
    //
    // Distinguished by what PRECEDED the input event, not by the event's own
    // inputType. Two earlier attempts failed on that: matching "the value is
    // an exact catalog hit" fired mid-paste, and `inputType ===
    // "insertReplacementText"` -- which the spec gives for a datalist
    // selection -- never matched in Chrome here, so the delay came back.
    //
    // A mouse pick is the only one of the three that arrives with no
    // keystroke and no paste just before it, and that holds whatever the
    // browser reports.
    // Open as soon as the field holds a root id the catalog knows, however it
    // got there.
    //
    // Three narrower rules were tried and all failed, so this is written from
    // a logged event trace rather than from the spec:
    //
    //   inputType === "insertReplacementText"  never fired in Chrome here
    //   "no keystroke just before the input"   same
    //   wait for `change`                      only fires on blur, so a pick
    //                                          did nothing until you clicked
    //                                          somewhere else
    //
    // The reason none of them can work: when the id is already complete --
    // pasted, or fully typed -- CLICKING THE MATCHING SUGGESTION FIRES NO
    // EVENT AT ALL, because the value does not change. The trace shows only a
    // keyless `keydown` artifact and then `change` on blur. There is no
    // "picked it" signal to wait for, so waiting for one means waiting
    // forever.
    //
    // Hence: react to the value, not to the gesture. Pasting a complete id
    // opens it immediately and the suggestion click becomes unnecessary
    // rather than required; picking after typing a few digits also lands an
    // exact match and opens. Partial input never matches, so nothing fires
    // mid-type. openFromCatalog ignores a file already shown, so the later
    // `change` on blur cannot load it twice.
    this.rootInput.addEventListener("input", () => {
      const value = this.rootInput.value.trim();
      if (value && this.catalog.some((e) => e.root_id === value)) open();
      // Emptying the box is "no root id selected": every neuron.
      else if (value === "" && this.active !== undefined) this.showAll();
    });
    this.rootInput.addEventListener("change", open);
    this.rootInput.addEventListener("keydown", (event: KeyboardEvent) => {
      if (event.key === "Enter") open();
    });
    this.allButton.textContent = "All neurons";
    this.allButton.title =
      "List candidates from every neuron in the catalog that pass the " +
      "filters (from " +
      CANDIDATE_TABLE_FILE +
      "). Merge and verdicts work on these rows too.";
    this.allButton.disabled = true;
    this.allButton.addEventListener("click", () => this.showAll());
    search.appendChild(this.rootInput);
    search.appendChild(this.rootList);
    search.appendChild(this.allButton);
    row.appendChild(search);

    if (this.catalogUrl) void this.loadCatalog(this.catalogUrl);
    return row;
  }

  /**
   * @param force bypass the HTTP cache. index.json is served with an ETag and
   *   no Cache-Control, so a plain fetch may be answered from cache; the
   *   refresh button passes true so it cannot return the same stale list.
   */
  private async loadCatalog(url: string, force = false) {
    const base = url.endsWith("/") ? url : url + "/";
    this.catalogStatus.textContent = "loading\u2026";
    try {
      let index: any;
      try {
        const response = await fetch(
          base + "index.json",
          force ? { cache: "reload" } : {},
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        index = await response.json();
      } catch {
        // No manifest. A plain nginx/Apache autoindex of the directory is
        // enough to work from, so fall back to that rather than refusing:
        // it means a directory is usable the moment it is served, before
        // make_results_index.py has ever run over it. The manifest is still
        // preferred where it exists, because it carries candidate counts and
        // review sizes that a file listing cannot.
        index = { results: await fetchDirectoryListing(base) };
      }
      const results = Array.isArray(index?.results) ? index.results : [];
      this.catalog = results as CatalogEntry[];
      this.rootInput.disabled = this.catalog.length === 0;
      this.allButton.disabled = this.catalog.length === 0;
      // A different catalog, or a forced reload: the table is stale.
      if (this.tableFor !== base || force) {
        this.table = undefined;
        this.tableState = "none";
        this.tableFor = base;
      }
      void this.loadNeuronMeta(base, force);
      this.catalogStatus.textContent = `${this.catalog.length} file(s)`;
      this.catalogStatus.title = index?.generated
        ? `index generated ${index.generated}`
        : "";
      // A shared link names a FILE. Open it once, here, rather than when the
      // tab was constructed: the catalog has to be in hand first so the file
      // can be resolved to a URL, and a tab opened before the fetch returned
      // would show an empty list under a link that promised a neuron.
      const want = this.layer.twigCapture.file;
      // A result computed in this browser is not in any catalog; syncLocal
      // opens it from the layer's store instead.
      if (want.startsWith(LOCAL_RESULT_PREFIX)) {
        // nothing to fetch
      } else if (want && !this.results.some((r) => r.name === want)) {
        void this.loadUrl(base + want);
      } else if (!want && this.active === undefined) {
        // No neuron named: the all-neurons view.
        void this.loadTable(force);
      }
      this.rebuildRootList();
    } catch (e) {
      this.catalog = [];
      this.rootInput.disabled = true;
      this.catalogStatus.textContent = "unreachable";
      StatusMessage.showTemporaryMessage(
        `Twig Capture: could not read ${base} (${e}). The directory needs ` +
          "either an index.json from scripts/make_results_index.py or a " +
          "browsable listing, and must send Access-Control-Allow-Origin.",
        6000,
      );
    }
  }

  /**
   * The root-id picker's options: the catalog's neurons that pass the neuron
   * filters, labelled with their SeaTable type where there is one.
   */
  private rebuildRootList() {
    removeChildren(this.rootList);
    for (const entry of this.catalog) {
      if (!this.neuronPasses(entry.root_id)) continue;
      const option = document.createElement("option");
      option.value = entry.root_id;
      const bits: string[] = [];
      const label = neuronLabel(this.neuronMeta?.neurons.get(entry.root_id));
      if (label) bits.push(label);
      bits.push(`${entry.candidate_count ?? "?"} cands`);
      if (entry.n_review !== undefined)
        bits.push(`${entry.n_review} to review`);
      if (entry.timestamp) bits.push(entry.timestamp.slice(0, 10));
      option.label = bits.join(", ");
      this.rootList.appendChild(option);
    }
  }

  // ---- SeaTable neuron filters ----------------------------------------------

  /** Fetch the catalog's neurons.json (optional) and show its filters. */
  private async loadNeuronMeta(base: string, force: boolean) {
    if (!force && this.neuronMetaFor === base) return;
    this.neuronMetaFor = base;
    this.neuronMeta = undefined;
    try {
      const meta = await fetchNeuronMeta(base, force);
      if (this.neuronMetaFor !== base) return; // catalog switched meanwhile
      this.neuronMeta = meta;
    } catch (e) {
      if (this.neuronMetaFor !== base) return;
      // Optional: without it the tab simply has no neuron filters.
      console.warn(`[twig-capture] could not read ${NEURON_META_FILE}: ${e}`);
    }
    this.renderNeuronFilters();
    this.rebuildRootList();
    this.render();
  }

  /** Neuron filters in force: chosen, and a neurons.json to apply them to. */
  private neuronFiltersActive(): boolean {
    return (
      this.neuronMeta !== undefined && hasNeuronFilters(this.neuronFilters)
    );
  }

  private neuronPasses(root: string): boolean {
    if (!this.neuronFiltersActive()) return true;
    return passesNeuronFilters(
      this.neuronMeta!.neurons.get(root),
      this.neuronFilters,
    );
  }

  private catalogRoots(): string[] {
    return [...new Set(this.catalog.map((e) => e.root_id))];
  }

  /**
   * One select per annotated field, each offering only values that some
   * neuron has under the other choices, with how many. A second row, so the
   * candidate filters above keep their place whether or not a catalog has
   * annotations.
   */
  private renderNeuronFilters() {
    const row = this.neuronRow;
    removeChildren(row);
    const meta = this.neuronMeta;
    if (meta === undefined || meta.fields.length === 0) {
      row.style.display = "none";
      return;
    }
    row.style.display = "";
    const roots = this.catalogRoots();
    const head = document.createElement("span");
    head.classList.add("neuroglancer-twig-capture-neuron-head");
    head.textContent = "neuron";
    head.title =
      `Filter neurons by their ${meta.source ?? "SeaTable"} annotations ` +
      `(${NEURON_META_FILE}${meta.generated ? `, ${meta.generated.slice(0, 16)}` : ""}). ` +
      "Narrows the all-neurons list and the root-id picker. A neuron edited " +
      "since inference is matched through its latest root; one not in the " +
      "table counts as blank.";
    row.appendChild(head);
    for (const field of NEURON_FIELDS) {
      if (!meta.fields.includes(field)) continue;
      const wrap = document.createElement("label");
      wrap.textContent = NEURON_FIELD_LABELS[field];
      const sel = document.createElement("select");
      const any = document.createElement("option");
      any.value = "";
      any.textContent = "any";
      sel.appendChild(any);
      const current = this.neuronFilters[field] ?? "";
      const options = fieldOptions(meta, field, roots, this.neuronFilters);
      if (current && !options.some(([v]) => v === current)) {
        options.unshift([current, 0]);
      }
      for (const [value, n] of options) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = `${value} (${n})`;
        sel.appendChild(opt);
      }
      sel.value = current;
      sel.addEventListener("change", () => {
        this.setNeuronFilter(field, sel.value);
      });
      wrap.appendChild(sel);
      row.appendChild(wrap);
    }
    const matching = roots.filter((r) => this.neuronPasses(r)).length;
    const count = document.createElement("span");
    count.classList.add("neuroglancer-twig-capture-neuron-count");
    count.textContent = `${matching} of ${roots.length} neurons`;
    row.appendChild(count);
    if (hasNeuronFilters(this.neuronFilters)) {
      const clear = document.createElement("button");
      clear.classList.add("neuroglancer-twig-capture-merge");
      clear.textContent = "any";
      clear.title = "Clear the neuron filters.";
      clear.addEventListener("click", () => this.setNeuronFilter());
      row.appendChild(clear);
    }
  }

  /** Set one neuron filter ("" = any), or clear them all with no field. */
  private setNeuronFilter(field?: NeuronField, value = "") {
    if (field === undefined) this.neuronFilters = {};
    else if (value)
      this.neuronFilters = { ...this.neuronFilters, [field]: value };
    else {
      const next = { ...this.neuronFilters };
      delete next[field];
      this.neuronFilters = next;
    }
    this.publish();
    this.renderNeuronFilters();
    this.rebuildRootList();
    // Neuron filters choose NEURONS, so choosing one with a single neuron
    // open goes to the all-neurons list, narrowed to the matching neurons:
    // otherwise the filter visibly does nothing, since one neuron's own
    // candidates are not narrowed by what that neuron is. Not mid-run.
    if (
      this.active !== undefined &&
      this.batch === undefined &&
      hasNeuronFilters(this.neuronFilters)
    ) {
      this.showAll();
      return;
    }
    this.render();
  }

  private async openFromCatalog(rootId: string) {
    if (!rootId) return;
    // Entries are sorted newest-first per root, so the first hit is the
    // latest run for that neuron.
    const entry = this.catalog.find((e) => e.root_id === rootId);
    if (entry === undefined) {
      StatusMessage.showTemporaryMessage(
        `Twig Capture: no results for root ${rootId} in this catalog.`,
        4000,
      );
      return;
    }
    // Already showing it. Both `input` and `change` reach here for a single
    // pick -- the second on blur -- and re-fetching would restart the graph
    // check and throw away the reviewer's scroll position for no gain. The
    // Refresh button is how you deliberately re-read a file.
    if (this.active?.name === entry.file) return;
    const base = this.catalogUrl.endsWith("/")
      ? this.catalogUrl
      : this.catalogUrl + "/";
    await this.loadUrl(base + entry.file);
  }

  // ---- all-neurons view ----------------------------------------------------

  /** No neuron chosen and a catalog to list from. */
  private isAllMode(): boolean {
    return this.active === undefined && this.catalog.length > 0;
  }

  private catalogBase(): string {
    return this.catalogUrl.endsWith("/")
      ? this.catalogUrl
      : this.catalogUrl + "/";
  }

  /** Leave the current neuron and list candidates from every neuron. */
  private showAll() {
    this.active = undefined;
    this.rootInput.value = "";
    this.layer.twigCapture.candidate = "";
    this.publish();
    this.render();
    void this.loadTable();
  }

  /**
   * Fetch the catalog's candidate table (once per catalog unless forced),
   * then every decision, then check the listed rows against the graph.
   */
  private async loadTable(force = false) {
    if (!this.catalogUrl || this.catalog.length === 0) return;
    const base = this.catalogBase();
    if (!force && this.tableFor === base && this.tableState !== "none") {
      if (this.tableState === "ready") void this.loadAllDecisions();
      return;
    }
    this.tableFor = base;
    this.tableState = "loading";
    this.render();
    try {
      const table = await fetchCandidateTable(base, force);
      if (this.tableFor !== base) return; // catalog switched meanwhile
      this.table = table;
      this.tableState = table === undefined ? "missing" : "ready";
    } catch (e) {
      if (this.tableFor !== base) return;
      this.table = undefined;
      this.tableState = "error";
      this.tableError = String(e);
    }
    this.render();
    if (this.tableState === "ready") {
      await this.loadAllDecisions();
      if (this.isAllMode()) void this.refreshAllRows();
    }
  }

  /** The dataset (datastack) of the rows being listed. */
  private listDataset(): string | undefined {
    return (
      this.active?.dataset ??
      this.table?.files.find((f) => f.dataset)?.dataset ??
      undefined
    );
  }

  private async loadAllDecisions() {
    if (!this.decisionsAllowed()) return;
    const store = this.store(this.listDataset());
    if (store === undefined) return;
    try {
      this.allDecisions = await store.fetchAll();
      this.allDecisionsVersion++;
      this.render();
    } catch (e) {
      StatusMessage.showTemporaryMessage(
        `Twig Capture: could not read decisions: ${e}`,
        5000,
      );
    }
  }

  /** The recorded verdict for a row, from whichever view it is listed in. */
  private decisionFor(c: TwigCandidate): DecisionRow | undefined {
    return c._lite
      ? this.allDecisions.get(c.target_segment_id)
      : this.decisions.get(c.target_segment_id);
  }

  /** A result file, fetched once and cached; the active file if it is that. */
  private fetchResultFile(file: string): Promise<TwigResult> {
    const loaded = this.results.find((r) => r.name === file);
    if (loaded !== undefined) return Promise.resolve(loaded);
    let p = this.fileCache.get(file);
    if (p === undefined) {
      const url = this.catalogBase() + file;
      p = (async () => {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
        return parseResult(file, await response.text());
      })();
      // A failure must not stick: drop it so the next click retries.
      p.catch(() => this.fileCache.delete(file));
      this.fileCache.set(file, p);
    }
    return p;
  }

  /**
   * The full record behind a row, with its result file. A row from the
   * all-neurons table carries display fields only; merging and recording a
   * verdict need merge_sink / merge_source, which are in the neuron's file.
   */
  private async ensureFull(c: TwigCandidate): Promise<RowRef | undefined> {
    const file = c._file ?? this.active?.name;
    if (file === undefined) return undefined;
    try {
      const result = await this.fetchResultFile(file);
      const candidate = c._lite
        ? result.candidates.find((x) => x.id === c.id)
        : c;
      if (candidate === undefined) {
        throw new Error(`${c.id} is not in ${file}; the catalog may be stale`);
      }
      return { result, candidate };
    } catch (e) {
      StatusMessage.showTemporaryMessage(
        `Twig Capture: could not load the neuron's result file: ${e}`,
        6000,
      );
      return undefined;
    }
  }

  /**
   * Mark rows that are already part of their neuron, for the rows the
   * all-neurons view shows. One getRoot per row plus one per neuron, rather
   * than the single-neuron check's file download: the table has no
   * supervoxels, so this resolves the recorded root ids instead. It only
   * labels rows; doMerge re-checks from supervoxels before any edit.
   */
  private async refreshAllRows() {
    const graph = this.graphene();
    if (graph?.getRoot === undefined || !this.isAllMode()) return;
    const rows = this.filtered().slice(0, this.allShown);
    const neuronRoot = new Map<string, Promise<bigint>>();
    const rootOf = (c: TwigCandidate) => {
      const file = c._file ?? "";
      const known = this.targetRoots.get(file);
      if (known !== undefined) return Promise.resolve(known);
      let p = neuronRoot.get(file);
      if (p === undefined) {
        p = graph.getRoot!(BigInt(c._root ?? "0"));
        neuronRoot.set(file, p);
      }
      return p;
    };
    this.refreshButton.disabled = true;
    this.refreshButton.textContent = "Refreshing…";
    let already = 0;
    try {
      const queue = [...rows];
      const worker = async () => {
        for (;;) {
          const c = queue.shift();
          if (c === undefined) return;
          try {
            const [target, cand] = await Promise.all([
              rootOf(c),
              graph.getRoot!(BigInt(c.target_segment_id)),
            ]);
            const key = rowKey(c);
            if (cand === target) {
              already++;
              if (this.mergeStatus.get(key) !== "done") {
                this.mergeStatus.set(key, "already");
              }
            } else if (this.mergeStatus.get(key) === "already") {
              this.mergeStatus.delete(key);
            }
          } catch {
            // one failed lookup says nothing about the row either way
          }
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(8, rows.length) }, worker),
      );
      if (rows.length > 0) {
        StatusMessage.showTemporaryMessage(
          `Twig Capture: checked the ${rows.length} listed candidate(s); ` +
            `${already} already part of their neuron.`,
          5000,
        );
      }
    } finally {
      this.refreshButton.disabled = false;
      this.refreshButton.textContent = "Refresh";
      this.render();
    }
  }

  /**
   * Which of the fetched results is being shown.
   *
   * Only a picker now. Results arrive solely from the catalog, so there is
   * nothing here to type into: no local file input (a reviewer's own copy of
   * a file is not what anyone else is looking at) and no free URL box (the
   * catalog already resolves every file in the served directory).
   */
  private makeResultSelectRow(): HTMLElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-twig-capture-controls");
    this.fileSelect.classList.add("neuroglancer-twig-capture-select");
    this.fileSelect.addEventListener("change", () => {
      this.active = this.results[this.fileSelect.selectedIndex];
      this.publish();
      this.render();
    });
    row.appendChild(this.fileSelect);
    return row;
  }

  private makeFilterRow(): HTMLElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-twig-capture-filters");

    const addNumber = (
      label: string,
      value: number,
      step: number,
      min: number,
      max: number,
      onChange: (v: number) => void,
      title: string,
    ) => {
      const wrap = document.createElement("label");
      wrap.textContent = label;
      wrap.title = title;
      const input = document.createElement("input");
      input.type = "number";
      input.value = String(value);
      input.step = String(step);
      input.min = String(min);
      input.max = String(max);
      input.addEventListener("change", () => {
        onChange(Number(input.value));
        this.render();
      });
      wrap.appendChild(input);
      row.appendChild(wrap);
    };

    // p, then a >= / <= choice, then the threshold. "<=" finds what the
    // model doubts but that may still be worth a look -- typically a
    // low-probability fragment carrying many synapses (sort by synapses).
    {
      const wrap = document.createElement("label");
      wrap.textContent = "p";
      wrap.title =
        "Merge probability filter: keep candidates at or above (≥) or " +
        "at or below (≤) the threshold.";
      const op = document.createElement("select");
      for (const [value, text] of [
        [">=", "≥"],
        ["<=", "≤"],
      ]) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = text;
        op.appendChild(opt);
      }
      op.value = this.probOp;
      op.addEventListener("change", () => {
        this.probOp = op.value === "<=" ? "<=" : ">=";
        this.publish();
        this.render();
      });
      this.probOpSelect = op;
      wrap.appendChild(op);
      row.appendChild(wrap);
    }
    addNumber(
      "",
      this.minProb,
      0.05,
      0,
      1,
      (v) => {
        this.minProb = v;
        this.publish();
      },
      "Merge probability threshold",
    );
    // Two facets, each a SELECT rather than checkboxes.
    //
    // hide-mito and mito-only were a pair of booleans that could both be
    // ticked, which showed nothing at all and needed each to clear the other
    // on change. A tri-state cannot express the contradiction in the first
    // place. The same argument rules out a multi-select for review state:
    // "reviewed" and "not reviewed" together mean either everything or
    // nothing, depending on how you read it, and neither reading is useful.
    //
    // Review folds the verdicts in as extra options rather than a third
    // control, because "show me the unsure ones" is the question a reviewer
    // actually asks, and it is mutually exclusive with the others anyway.
    const addFacet = (
      label: string,
      title: string,
      options: [string, string][],
      get: () => string,
      set: (v: string) => void,
    ) => {
      const wrap = document.createElement("label");
      wrap.textContent = label;
      wrap.title = title;
      const sel = document.createElement("select");
      for (const [value, text] of options) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = text;
        sel.appendChild(opt);
      }
      sel.value = get();
      sel.addEventListener("change", () => {
        set(sel.value);
        this.publish();
        this.render();
      });
      wrap.appendChild(sel);
      row.appendChild(wrap);
      return sel;
    };

    this.mitoSelect = addFacet(
      "mito",
      "likely_mito = enclosure > 0.6 and mean EM < 100. Provisional: derived " +
        "from the GT distribution, not from a reviewed sample.",
      [
        ["any", "any"],
        ["hide", "hide mito"],
        ["only", "mito only"],
      ],
      () => this.mito,
      (v) => (this.mito = v),
    );

    this.reviewSelect = addFacet(
      "review",
      "Filters on decisions recorded in twig_capture_decisions (CAVE) for " +
        "this neuron, including other people's. Disabled when decisions are " +
        "not enabled for this segmentation source, since nothing is loaded " +
        "and every row would look unreviewed.",
      [
        ["any", "any"],
        ["none", "not reviewed"],
        ["done", "reviewed"],
        ["merge", "merge"],
        ["no_merge", "not a merge"],
        ["unsure", "unsure"],
        ["new_neuron", "new neuron"],
        ["split_needed", "split needed"],
      ],
      () => this.review,
      (v) => (this.review = v),
    );

    const sortWrap = document.createElement("label");
    sortWrap.textContent = "sort";
    const sort = document.createElement("select");
    for (const [value, label] of [
      ["prob", "probability"],
      ["size", "size"],
      ["contact", "contact"],
      ["synapses", "orphan synapses (total)"],
    ] as const) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      sort.appendChild(opt);
    }
    sort.title =
      "orphan synapses (total) ranks by cand_synapses_total: synapses on the " +
      "candidate's whole extent, i.e. what attaching it recovers. Falls back " +
      "to the wide window for older files, which understates it. Files with " +
      "no synapse counts at all sort as zero.";
    sort.value = this.sortKey;
    sort.addEventListener("change", () => {
      this.sortKey = sort.value as typeof this.sortKey;
      this.publish();
      this.render();
    });
    sortWrap.appendChild(sort);
    row.appendChild(sortWrap);

    return row;
  }

  /**
   * Pick up results computed in this browser: list every one in the result
   * chooser, and open the one the Twig Selector asked for (or the one a
   * shared link names, if this browser has it).
   */
  private syncLocal() {
    const local = this.layer.localTwigResults;
    const want =
      local.pendingOpen ??
      (this.active === undefined &&
      this.layer.twigCapture.file.startsWith(LOCAL_RESULT_PREFIX)
        ? this.layer.twigCapture.file
        : undefined);
    local.pendingOpen = undefined;
    const combined = local.pendingCombined;
    local.pendingCombined = undefined;
    // Drop local results that were deleted from the store.
    this.results = this.results.filter(
      (r) => !r.name.startsWith(LOCAL_RESULT_PREFIX) || local.get(r.name),
    );
    if (this.active && !this.results.includes(this.active))
      this.active = undefined;
    for (const json of local.results) {
      const existing = this.results.find((r) => r.name === json.name);
      if (existing && json.name !== want) continue;
      const result = parseResultObject(json.name, json as LocalResultJson);
      if (json.name === want) {
        this.addResult(result);
        this.publish();
        void this.refreshAgainstGraph(result);
      } else {
        this.results.push(result);
        this.fileCache.set(result.name, Promise.resolve(result));
      }
    }
    if (combined !== undefined && combined.length) {
      const members = combined
        .map((n) => this.results.find((r) => r.name === n))
        .filter((r): r is TwigResult => r !== undefined);
      if (members.length) {
        const result = combineResults(members);
        this.addResult(result);
        this.publish();
        void this.refreshAgainstGraph(result);
      }
    }
    this.render();
  }

  private async loadUrl(url: string) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const name = url.split("/").pop() || url;
      const result = parseResult(name, await response.text());
      this.addResult(result);
      this.publish();
      this.render();
      // Check the graph straight away rather than waiting for a button.
      // Measured over the 202-neuron production run, the default filters
      // leave a median of 112 rows and at worst 501, which at the measured
      // 94 getRoot/s is 1.2 s typical and 5.3 s worst. Cheap enough that
      // making the reviewer ask for it only means the first thing they see
      // is out of date -- retired roots, and candidates someone else has
      // already merged shown as available.
      void this.refreshAgainstGraph(result);
    } catch (e) {
      StatusMessage.showTemporaryMessage(
        `Twig Capture: could not fetch ${url}: ${e}`,
        5000,
      );
    }
  }

  private addResult(result: TwigResult) {
    // Merge status and history are keyed by file AND cand_N (rowKey), so
    // they survive switching neurons -- a merge made in the all-neurons view
    // still reads "Merged" when that neuron is opened. The target root is
    // re-resolved by the load-time check, so a reloaded file drops it.
    this.targetRoots.delete(result.name);
    this.fileCache.set(result.name, Promise.resolve(result));
    // Fresh decisions. These are keyed by candidate id, which says
    // nothing about which neuron they belong to, so leaving them showed the
    // PREVIOUS neuron's verdicts against the new one's candidates until
    // loadDecisions returned. Where decisions are not enabled for the layer
    // loadDecisions returns early, so they would have persisted indefinitely
    // -- a chip claiming a candidate had been judged when it had not.
    this.decisions.clear();
    // Re-loading the same file replaces it rather than stacking duplicates.
    const existing = this.results.findIndex((r) => r.name === result.name);
    if (existing >= 0) {
      this.results[existing] = result;
    } else {
      this.results.push(result);
    }
    this.active = result;
  }

  /** nm -> voxels of the viewer's global coordinate space. */
  private nmToPosition(nm: readonly number[]): Float32Array | undefined {
    const space = this.layer.manager.root.coordinateSpace.value;
    const { rank, scales, units } = space;
    if (!space.valid || rank < 3) return undefined;
    const position = Float32Array.from(
      this.layer.manager.root.globalPosition.value,
    );
    for (let i = 0; i < 3 && i < rank; ++i) {
      // scales are in canonical units (metres for space); the JSON is in nm.
      if (units[i] !== "m") return undefined;
      position[i] = (nm[i] * 1e-9) / scales[i];
    }
    return position;
  }

  private goTo(candidate: TwigCandidate) {
    // Which row was being looked at, by segment id rather than cand_N: the
    // numbering shifts with threshold and sort order, the id does not.
    this.layer.twigCapture.candidate = candidate.target_segment_id;
    this.layer.twigCapture.changed.dispatch();
    // Re-render so the row marks itself as the one being looked at. Nothing
    // listens to twigCapture.changed here, so without this the highlight only
    // appeared on the NEXT render -- after a merge or a filter change -- which
    // made it look like it was tracking something else entirely.
    //
    // scrolledToShared is set first: the reader just clicked this row, so it
    // is on screen, and scrolling it to centre would yank the list under the
    // cursor for no reason. The once-only scroll exists for shared links,
    // where the row may be hundreds down.
    this.scrolledToShared = true;
    this.render();
    const position = this.nmToPosition(candidate.coordinate_nm);
    if (position === undefined) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: viewer coordinate space is not in metres; cannot navigate.",
        4000,
      );
      return;
    }
    // The setter is a no-op if the rank disagrees, so write through
    // globalPosition and dispatch, as setLayerPosition does.
    const { globalPosition } = this.layer.manager.root;
    globalPosition.value.set(position);
    globalPosition.changed.dispatch();

    // In a staged list the view already shows the whole proposal -- the
    // neuron and every ticked candidate -- and swapping it for this one pair
    // would also untick the rest. Just fly there.
    if (this.activeStage() !== undefined) return;

    // Show both sides of the proposed merge: the neuron under review and the
    // candidate.
    let ids: bigint[];
    try {
      // targetRoots tracks merges made in this session; fall back to the
      // root recorded in the file (showSegments resolves it to today's).
      const file = candidate._file ?? this.active?.name ?? "";
      ids = [
        this.targetRoots.get(file) ??
          BigInt(candidate._root ?? this.active!.root_id),
        BigInt(candidate.target_segment_id),
      ];
    } catch {
      StatusMessage.showTemporaryMessage(
        `Twig Capture: "${candidate.target_segment_id}" is not a valid segment id.`,
        4000,
      );
      return;
    }
    void this.showSegments(ids);
  }

  /**
   * Select and display segments, resolving each to its CURRENT root first.
   *
   * The ids in the file are roots as they stood when inference ran. Every
   * proofreading edit since then has retired some of them, and a retired root
   * has no mesh and no voxels to display -- the candidate is silently
   * invisible, which is what "the mesh does not pop up" looks like. Asking the
   * graph for today's root fixes that; a layer with no graph (or an id already
   * current) falls through unchanged.
   */
  /**
   * Show EXACTLY these segments, replacing whatever was displayed.
   *
   * This used to only add. Clicking down a list therefore accumulated every
   * candidate visited, so after a handful of rows the viewer showed a pile of
   * neurons and the pair actually under consideration was indistinguishable
   * among them -- which defeats the point of clicking a row.
   *
   * Both sets are cleared, not just visibleSegments: selectedSegments is what
   * keeps a segment in the visible set, and showSegments is what put the
   * earlier ones there, so leaving it would re-accumulate the same pile one
   * layer down.
   */
  private async showSegments(ids: bigint[]) {
    const group = this.layer.displayState.segmentationGroupState.value;
    group.visibleSegments.clear();
    group.selectedSegments.clear();
    // getRoot is on GrapheneGraphSource, not on the abstract base, so probe
    // for it structurally rather than widening the base class.
    const graph = group.graph.value as
      | { getRoot?(id: bigint, timestamp?: number): Promise<bigint> }
      | undefined;
    let remapped = 0;
    for (const id of ids) {
      let show = id;
      if (graph?.getRoot !== undefined) {
        try {
          show = await graph.getRoot(id);
          if (show !== id) remapped++;
        } catch {
          show = id; // keep the stored id; it may still be current
        }
      }
      // selected before visible: this is the order the segment list uses, and
      // a segment that is not selected is not kept in the visible set.
      group.selectedSegments.add(show);
      group.visibleSegments.set(show, true);
    }
    if (remapped > 0) {
      StatusMessage.showTemporaryMessage(
        `Twig Capture: ${remapped} segment(s) had been edited since inference; ` +
          "showing today's root.",
        4000,
      );
    }
  }

  /**
   * Re-sync the facet selects with state restored from a link, and grey out
   * the review facet when there are no decisions to filter on.
   */
  private renderFilters() {
    if (this.mitoSelect !== undefined) this.mitoSelect.value = this.mito;
    if (this.probOpSelect !== undefined) this.probOpSelect.value = this.probOp;
    if (this.reviewSelect !== undefined) {
      this.reviewSelect.value = this.review;
      this.reviewSelect.disabled = !this.decisionsAllowed();
    }
  }

  /**
   * Re-resolve every listed candidate against the live graph.
   *
   * The geometry columns (contact, enclosure, EM) are NOT refreshed -- they
   * were measured against the neuron as it stood when inference ran, and only
   * re-running run_twigcapture.py would update them. What this catches is the
   * thing that actually bites in a shared environment: a candidate that is
   * already part of the neuron, because an earlier row merged it, because
   * someone else merged it, or because it was merged between the inference run
   * and now. Offering "Merge" on those is at best a wasted click and at worst
   * confusing when the server refuses it.
   *
   * Cheap by construction: one getRoot per candidate, no re-inference. Roots
   * come from the SUPERVOXELS, so this works no matter how stale the file is.
   */

  /**
   * Undo a merge, or redo one that was undone.
   *
   * Both directions are the SAME call: /undo takes an operation id, and the
   * undo it performs is itself a new operation. So undoing the undo re-applies
   * the merge, and the id to act on is simply whichever operation currently
   * stands. Nothing is ever removed from the graph's history.
   *
   * Root ids change again on both paths, so the list is re-resolved afterwards
   * rather than assumed.
   */
  private async undoOrRedo(candidate: TwigCandidate) {
    const server = this.graphene()?.graphServer;
    const key = rowKey(candidate);
    const entry = this.history.get(key);
    if (server === undefined || entry === undefined) return;
    const redoing = entry.undone;
    this.mergeStatus.set(key, "merging");
    this.render();
    try {
      const { operationId } = await server.undoOperation(entry.operationId);
      // Act on the operation that now stands: after an undo, that is the undo
      // itself; undoing it again is the redo.
      this.history.set(key, {
        operationId: operationId ?? entry.operationId,
        undone: !entry.undone,
      });
      // The neuron's root changed again; let the next check resolve it.
      this.targetRoots.delete(candidate._file ?? "");
      if (redoing) {
        this.mergeStatus.set(key, "done");
      } else {
        // Undone: the candidate stands apart again, so offer Merge afresh.
        this.mergeStatus.delete(key);
      }
      StatusMessage.showTemporaryMessage(
        redoing
          ? `Redone: ${candidate.target_segment_id} merged again.`
          : `Undone: ${candidate.target_segment_id} split back off.`,
        5000,
      );
      if (operationId === undefined) {
        StatusMessage.showTemporaryMessage(
          "The server did not return an operation id for that undo, so it " +
            "cannot be redone from here.",
          6000,
        );
        this.history.delete(key);
      }
      // Every root involved has changed; re-resolve rather than guess.
      await this.refreshAgainstGraph();
    } catch (e) {
      this.mergeStatus.set(key, "failed");
      console.error(
        `[twig-capture] ${redoing ? "redo" : "undo"} failed`,
        candidate.id,
        entry.operationId,
        e,
      );
      StatusMessage.showTemporaryMessage(
        `Twig Capture: ${redoing ? "redo" : "undo"} failed for ` +
          `${candidate.target_segment_id}: ${e}`,
        6000,
      );
      this.render();
    }
  }

  /**
   * Re-check the neuron and its candidates against the graph as it is now.
   *
   * `loadedFor` is set when this ran by itself on load, rather than from the
   * button. Two differences then: a missing graph is not worth a popup the
   * reviewer did not ask for, and the file may have been switched while the
   * requests were in flight, in which case the answers belong to a result
   * nobody is looking at and are dropped.
   */
  private async refreshAgainstGraph(loadedFor?: TwigResult) {
    const graph = this.graphene();
    const result = loadedFor ?? this.active;
    if (result === undefined) {
      // The all-neurons view: re-read decisions, re-check the listed rows.
      if (this.isAllMode()) {
        void this.loadAllDecisions();
        await this.refreshAllRows();
      }
      return;
    }
    // Decisions are independent of the graph check: they come from CAVE, not
    // the chunkedgraph, and a layer with no graph can still show what other
    // reviewers have already judged. Kick it off before the early return.
    void this.loadDecisions(result);
    if (graph?.getRoot === undefined) {
      if (loadedFor === undefined) {
        StatusMessage.showTemporaryMessage(
          "Twig Capture: this layer has no graph to refresh against.",
          4000,
        );
      }
      return;
    }
    if (result.members !== undefined) {
      await this.refreshMembers(result, loadedFor !== undefined);
      return;
    }
    const rows = this.filtered();
    this.refreshButton.disabled = true;
    this.refreshButton.textContent = "Refreshing\u2026";
    // Only the load-time run withholds the list. From the button the rows
    // are already on screen and already meaningful, so replacing them with
    // a spinner would lose the reviewer's place for no gain.
    if (loadedFor !== undefined) {
      this.checking = { done: 0, total: rows.length };
      this.render();
    }
    try {
      // The neuron's root today. Taken from a sink SUPERVOXEL when the file
      // has one -- the file's root_id may itself be retired.
      const anySink = result.candidates.find((c) => c.merge_sink)?.merge_sink;
      const targetRoot = anySink
        ? await graph.getRoot(BigInt(anySink.supervoxel_id))
        : await graph.getRoot(BigInt(result.root_id));
      if (loadedFor !== undefined && this.active !== result) return;
      this.targetRoots.set(result.name, targetRoot);
      if (String(targetRoot) !== result.root_id) {
        // A root is a version, not an identity. Say so, because the
        // alternative is a reviewer wondering why the header disagrees with
        // the id they searched for.
        StatusMessage.showTemporaryMessage(
          `Twig Capture: root ${result.root_id} has been edited since this ` +
            `file was written; the neuron is now ${targetRoot}. The ` +
            "candidates are still valid -- they are anchored on supervoxels.",
          6000,
        );
      }

      let already = 0;
      let checked = 0;
      // Bounded concurrency: this is one request per candidate against a
      // rate-limited server, and a list can run to hundreds of rows.
      const queue = [...rows];
      const worker = async () => {
        for (;;) {
          const candidate = queue.shift();
          if (candidate === undefined) return;
          const sv = candidate.merge_source?.supervoxel_id;
          try {
            const root = sv
              ? await graph.getRoot!(BigInt(sv))
              : await graph.getRoot!(BigInt(candidate.target_segment_id));
            checked++;
            if (this.checking !== undefined) {
              this.checking.done = checked;
              // Repaint the one line rather than the whole tab: a full
              // render per candidate would be 500 rebuilds on a big file.
              const el = this.listEl.querySelector(
                ".neuroglancer-twig-capture-checking span:last-child",
              );
              if (el !== null) {
                el.textContent =
                  `Checking ${this.checking.total} candidate(s) against ` +
                  `the chunkedgraph\u2026 ${checked}/${this.checking.total}`;
              }
            }
            const key = rowKey(candidate);
            if (root === targetRoot) {
              already++;
              // Do not overwrite a merge this session already recorded.
              if (this.mergeStatus.get(key) !== "done") {
                this.mergeStatus.set(key, "already");
              }
            } else if (this.mergeStatus.get(key) === "already") {
              this.mergeStatus.delete(key);
            }
          } catch {
            // leave this row's state alone; a single failed lookup is not
            // evidence about the candidate either way
          }
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(8, rows.length) }, worker),
      );
      if (loadedFor !== undefined && this.active !== result) return;
      StatusMessage.showTemporaryMessage(
        `Twig Capture: checked ${checked} candidate(s); ${already} already ` +
          `part of the neuron. Geometry columns are from the inference run ` +
          `and were not recomputed.`,
        6000,
      );
    } finally {
      this.checking = undefined;
      this.refreshButton.disabled = false;
      this.refreshButton.textContent = "Refresh";
      this.render();
    }
  }

  /**
   * The CAVE server this layer talks to, taken from the data source URL.
   *
   * A graphene source looks like
   *   graphene://middleauth+https://cave.fanc-fly.com/segmentation/...
   * so the first http(s) origin in any source is the CAVE deployment. Derived
   * rather than hardcoded because Aedes and BANC are different datastacks,
   * and a wrong server would write decisions into someone else's project.
   */
  private caveServer(): string | undefined {
    for (const ds of this.layer.dataSources) {
      const url = ds.spec?.url ?? "";
      const m = /https?:\/\/[^/\s]+/.exec(url);
      if (m !== null) return m[0];
    }
    return undefined;
  }

  /** Every data source URL on this layer. */
  private sourceUrls(): string[] {
    return this.layer.dataSources.map((ds) => ds.spec?.url ?? "");
  }

  /**
   * Is this layer one we are allowed to record decisions for?
   *
   * See DECISION_ALLOWLIST. Decisions are ground truth, and a row posted
   * against the wrong volume is a false label nothing in the row would
   * reveal, so an un-vouched-for volume gets no verdict control and no write.
   */
  private decisionsAllowed(): boolean {
    return decisionsAllowed(this.sourceUrls());
  }

  /**
   * The decision store for a dataset (default: the active result's, else the
   * all-neurons table's), or undefined when there is no datastack or server.
   */
  private store(dataset?: string): DecisionStore | undefined {
    const datastack = dataset ?? this.listDataset();
    const server = this.caveServer();
    if (datastack === undefined || server === undefined) return undefined;
    if (
      this.decisionStore === undefined ||
      this.decisionStore.datastack !== datastack ||
      this.decisionStore.server !== server
    ) {
      this.decisionStore = new DecisionStore(server, datastack);
    }
    return this.decisionStore;
  }

  /**
   * Load every decision recorded against this neuron.
   *
   * Failure is reported but never fatal: the decision table is a record of
   * review, not a prerequisite for doing any. A reviewer with no CAVE
   * credentials can still read candidates and merge.
   */
  /**
   * The neuron's root TODAY, for querying CAVE.
   *
   * NOT result.root_id. That is a snapshot from when inference ran, and a
   * root is a version rather than an identity: one merge in this tab retires
   * it. CAVE's live query rejects a retired root outright --
   * "Some root_ids passed are not valid at the query timestamp" -- so using
   * the file's id made decisions fail to load immediately after any merge.
   *
   * Resolved the same way refreshAgainstGraph does it: from a sink
   * SUPERVOXEL, which is immutable. targetRoots is preferred when set
   * because it reflects merges made in this session without another round
   * trip. Falls back to the file's root only when there is no graph to ask,
   * which is also the only case where it can still be stale.
   */
  private async currentNeuronRoot(result: TwigResult): Promise<string> {
    const known = this.targetRoots.get(result.name);
    if (known !== undefined) return String(known);
    const graph = this.graphene();
    const anySink = result.candidates.find((c) => c.merge_sink)?.merge_sink;
    if (graph?.getRoot !== undefined && anySink) {
      try {
        return String(await graph.getRoot(BigInt(anySink.supervoxel_id)));
      } catch {
        // Fall through: a failed lookup should cost the decisions panel, not
        // the refresh that is about to run anyway.
      }
    }
    return result.root_id;
  }

  private async loadDecisions(result: TwigResult) {
    if (!this.decisionsAllowed()) return;
    const store = this.store(result.dataset);
    if (store === undefined) return;
    if (result.members !== undefined) {
      // A combined list: every member neuron's decisions, in one map (keyed
      // by candidate segment, which does not collide across neurons).
      try {
        const all = new Map<string, DecisionRow>();
        for (const file of result.members) {
          const own = await this.resultForFile(file);
          if (own === undefined) continue;
          const root = await this.currentNeuronRoot(own);
          for (const [k, v] of await store.fetchForNeuron(root)) all.set(k, v);
          if (this.active !== result) return;
        }
        this.decisions = all;
        this.render();
      } catch (e) {
        StatusMessage.showTemporaryMessage(
          `Twig Capture: could not read decisions: ${e}`,
          5000,
        );
      }
      return;
    }
    try {
      const root = await this.currentNeuronRoot(result);
      if (this.active !== result) return;
      const found = await store.fetchForNeuron(root);
      if (this.active !== result) return;
      this.decisions = found;
      this.render();
    } catch (e) {
      StatusMessage.showTemporaryMessage(
        `Twig Capture: could not read decisions: ${e}`,
        5000,
      );
    }
  }

  /**
   * Record one verdict in CAVE.
   *
   * `op` is set only when the merge was actually executed, so a row carrying
   * op= is evidence an edit happened and a row without it is a judgement
   * alone. The table is append-only: reviewing the same candidate twice
   * leaves both rows and the later one wins, which keeps the audit trail.
   */
  private async recordDecision(
    candidate: TwigCandidate,
    verdict: Verdict,
    op?: string,
    cell?: { side: "sink" | "source"; root: string },
  ): Promise<boolean> {
    if (!this.decisionsAllowed()) {
      // The control is hidden for these layers, so reaching here means a
      // stale render or a caller that did not check. Refuse loudly rather
      // than write ground truth against an unvouched volume.
      StatusMessage.showTemporaryMessage(
        "Twig Capture: decisions are not enabled for this segmentation " +
          "source, so nothing was written. See DECISION_ALLOWLIST.",
        6000,
      );
      return false;
    }
    const key = rowKey(candidate);
    // The row's own result file, not the active one: in the all-neurons view
    // every row belongs to a different neuron, and src= must name its file.
    this.deciding.add(key);
    this.render();
    const ref = await this.ensureFull(candidate);
    this.deciding.delete(key);
    if (ref === undefined) {
      this.render();
      return false;
    }
    const { result } = ref;
    const store = this.store(result.dataset);
    if (store === undefined) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: no CAVE datastack for this layer; decision not saved.",
        5000,
      );
      this.render();
      return false;
    }
    const sink = ref.candidate.merge_sink;
    const source = ref.candidate.merge_source;
    if (!sink || !source) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: this result file has no merge_sink / merge_source, so " +
          "the two sides cannot be bound. Re-run run_twigcapture.py.",
        6000,
      );
      this.render();
      return false;
    }
    // A sided verdict names its cell by the root it has NOW, looked up from
    // that side's supervoxel: the file's ids are a snapshot from inference,
    // and the cell has often been proofread since. cell= is what downstream
    // reads; pt/pt2 are contact-face points and can bind to the neighbour.
    if (cell !== undefined) {
      const graph = this.graphene();
      const sv = (cell.side === "sink" ? sink : source).supervoxel_id;
      try {
        if (graph?.getRoot !== undefined) {
          cell = {
            side: cell.side,
            root: String(await graph.getRoot(BigInt(sv))),
          };
        }
      } catch {
        // keep the file's id; it is still the right cell, only maybe stale
      }
      if (!/^\d+$/.test(cell.root)) {
        StatusMessage.showTemporaryMessage(
          "Twig Capture: could not tell which cell this is; nothing was written.",
          6000,
        );
        this.render();
        return false;
      }
    }
    // The result file stem, which is what src= has to point at.
    const src = result.name.replace(/\.json$/, "");
    const target: DecisionTarget = {
      sink,
      source,
      // the full record's: the table rounds coordinates to whole nm
      coordinate_nm: ref.candidate.coordinate_nm,
      cand: candidate.target_segment_id,
      src,
      cell,
    };
    this.deciding.add(key);
    this.render();
    try {
      const id = await store.post(target, verdict, op);
      const row: DecisionRow = {
        id,
        tag: verdict,
        cand: candidate.target_segment_id,
        tag2:
          `src=${src};cand=${candidate.target_segment_id}` +
          (cell ? `;cell=${cell.root};side=${cell.side}` : ""),
        created: Date.now(),
      };
      // Both views: the single-neuron map only if this is the neuron shown.
      if (this.active === result) {
        this.decisions.set(candidate.target_segment_id, row);
      }
      this.allDecisions.set(candidate.target_segment_id, row);
      this.allDecisionsVersion++;
      return true;
    } catch (e) {
      StatusMessage.showTemporaryMessage(
        `Twig Capture: decision NOT saved: ${e}`,
        7000,
      );
      return false;
    } finally {
      this.deciding.delete(key);
      this.render();
    }
  }

  private graphene(): GrapheneLike | undefined {
    return this.layer.displayState.segmentationGroupState.value.graph.value as
      | GrapheneLike
      | undefined;
  }

  /**
   * Perform one merge against the chunkedgraph.
   *
   * THIS WRITES TO SHARED DATA. It adds an edge to the segmentation graph,
   * attributed to the signed-in user, and every other proofreader sees it.
   *
   * Roots are resolved from the SUPERVOXELS at call time, never taken from the
   * file: the file's roots are a snapshot from when inference ran, and the
   * server rejects a stale root. Supervoxels never change, which is why
   * run_twigcapture.py records them.
   */
  private async doMerge(
    row: TwigCandidate,
    batch?: { neuron: bigint },
  ): Promise<MergeOutcome> {
    const graph = this.graphene();
    const server = graph?.graphServer;
    if (server === undefined) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: this layer has no graphene graph to merge on.",
        4000,
      );
      return "failed";
    }
    const key = rowKey(row);
    this.skipReason.delete(key);
    this.mergeStatus.set(key, "merging");
    this.render();
    // In a staged run nothing ever asks: whatever would need a confirm on a
    // single merge is SKIPPED instead, with the reason on the row, and left
    // for the reviewer to merge by hand.
    const skip = (reason: string): MergeOutcome => {
      this.mergeStatus.set(key, "skipped");
      this.skipReason.set(key, reason);
      this.render();
      return "skipped";
    };
    // The full record: a row from the all-neurons table has no merge
    // endpoints of its own.
    const ref = await this.ensureFull(row);
    if (ref === undefined) {
      this.mergeStatus.delete(key);
      this.render();
      return "failed";
    }
    const { candidate, result } = ref;
    // SAFEGUARD: the record fetched for a table row must be the same
    // candidate. cand_N is matched within the file; the segment id has to
    // agree too, or the table and the file are out of step and the merge
    // endpoints may belong to a different candidate.
    if (candidate.target_segment_id !== row.target_segment_id) {
      if (batch) return skip("the catalog and the result file disagree");
      StatusMessage.showTemporaryMessage(
        `Twig Capture: NOT merged. ${row.id} in ${result.name} is ` +
          `${candidate.target_segment_id}, but the list says ` +
          `${row.target_segment_id}. The catalog is stale: refresh it (\u21bb).`,
        8000,
      );
      this.mergeStatus.delete(key);
      this.render();
      return "cancelled";
    }
    const sink = candidate.merge_sink;
    const source = candidate.merge_source;
    if (!sink || !source) {
      if (batch) return skip("no merge endpoints in the result file");
      StatusMessage.showTemporaryMessage(
        "Twig Capture: this result file predates the merge endpoints " +
          "(merge_sink / merge_source). Re-run run_twigcapture.py to add them.",
        6000,
      );
      this.mergeStatus.delete(key);
      this.render();
      return "failed";
    }

    try {
      const sinkSv = BigInt(sink.supervoxel_id);
      const sourceSv = BigInt(source.supervoxel_id);
      // Today's roots for both sides.
      const [sinkRoot, sourceRoot] = await Promise.all([
        graph!.getRoot ? graph!.getRoot(sinkSv) : Promise.resolve(sinkSv),
        graph!.getRoot ? graph!.getRoot(sourceSv) : Promise.resolve(sourceSv),
      ]);
      if (sinkRoot === sourceRoot) {
        this.mergeStatus.set(key, "done");
        StatusMessage.showTemporaryMessage(
          "Already merged: both sides are the same segment today.",
          4000,
        );
        this.render();
        return "already";
      }
      // HAS THE CANDIDATE CHANGED HANDS SINCE INFERENCE?
      //
      // target_segment_id is the candidate's root at inference time, and the
      // filters guarantee it was a twig of at most 20 L2 nodes. If someone
      // has merged that twig into something else since, sourceRoot is now
      // that OTHER segment -- and merging would attach all of it to this
      // neuron. The row would still read as a 1-node twig at p=0.99, and
      // nothing above catches it: sinkRoot !== sourceRoot is perfectly true
      // when the source is an unrelated whole neuron.
      //
      // A changed root does not by itself mean the merge is wrong -- the twig
      // may simply have grown by one node -- so this asks rather than
      // refuses. It is the one case in this tab where proceeding silently
      // could produce an arbitrarily large edit.
      if (String(sourceRoot) !== candidate.target_segment_id) {
        if (batch) {
          return skip(
            `changed hands since inference: now part of ${sourceRoot}, ` +
              "which may be far larger than the twig scored",
          );
        }
        const ok = window.confirm(
          `Candidate ${candidate.target_segment_id} has been edited since ` +
            `this file was written -- it is now part of ${sourceRoot}.\n\n` +
            `Merging joins THAT segment to neuron ${result.root_id}, which may be much ` +
            "larger than the twig scored here " +
            `(${candidate.num_l2_nodes} L2 node(s)).\n\n` +
            "Check it in the viewer first. Merge anyway?",
        );
        if (!ok) {
          this.mergeStatus.delete(key);
          this.render();
          return "cancelled";
        }
      }
      // SAFEGUARD: IS THE CONTACT STILL ON THE NEURON?
      //
      // sinkRoot is whatever segment the neuron-side supervoxel belongs to
      // today. If the neuron has been split since inference, that may be a
      // detached piece rather than the neuron -- and the candidate would be
      // joined to the wrong object. The neuron's own root is taken from
      // several of its OTHER contact points (supervoxels, so valid however
      // stale the file); a sink that disagrees with them is refused from the
      // all-neurons list and needs explicit confirmation on the one-neuron
      // view, where the reviewer is looking at the neuron.
      // A staged run checks against the neuron root it is tracking instead:
      // established by the same consensus before the run, then advanced to
      // each merge's new root. A sink anywhere else is skipped.
      if (batch && sinkRoot !== batch.neuron) {
        return skip(
          `its contact is on segment ${sinkRoot}, not on the neuron ` +
            `(${batch.neuron})`,
        );
      }
      const neuron = batch
        ? undefined
        : await this.neuronRootConsensus(result, candidate);
      const fromAll = this.active !== result;
      if (neuron !== undefined && neuron.root !== sinkRoot) {
        const why =
          `This candidate's contact is on segment ${sinkRoot}, but ` +
          `${neuron.agree} of ${neuron.total} other contact points of neuron ` +
          `${result.root_id} are on ${neuron.root}. The neuron may have been ` +
          "split or edited since inference.";
        if (fromAll) {
          StatusMessage.showTemporaryMessage(
            `Twig Capture: NOT merged. ${why} Open it with "Neuron" to check.`,
            10000,
          );
          this.mergeStatus.delete(key);
          this.render();
          return "cancelled";
        }
        if (!window.confirm(`${why}\n\nMerge onto ${sinkRoot} anyway?`)) {
          this.mergeStatus.delete(key);
          this.render();
          return "cancelled";
        }
      }
      // SAFEGUARD: from the all-neurons list every row is a different
      // neuron, so say which one before editing.
      if (fromAll && !batch) {
        const syn = candidate.cand_synapses_total;
        const ok = window.confirm(
          `Merge candidate ${candidate.target_segment_id}\n` +
            `  (${candidate.num_l2_nodes} L2 node(s), p ${fmt(candidate.prob_merge)}` +
            `${syn != null ? `, ${syn} synapses` : ""})\n` +
            `into neuron ${result.root_id}` +
            `${String(sinkRoot) !== result.root_id ? ` (today ${sinkRoot})` : ""}?\n\n` +
            "This edits the shared segmentation and is attributed to you.",
        );
        if (!ok) {
          this.mergeStatus.delete(key);
          this.render();
          return "cancelled";
        }
      }
      // positions are already in nanometres in the file, which is the unit
      // mergeSegments expects -- graphene's own callers convert only because
      // their positions come from annotations in layer voxel space.
      console.log("[twig-capture] merge request", {
        candidate: candidate.id,
        sink: {
          supervoxel: String(sinkSv),
          root: String(sinkRoot),
          position_nm: sink.coordinate_nm,
        },
        source: {
          supervoxel: String(sourceSv),
          root: String(sourceRoot),
          position_nm: source.coordinate_nm,
        },
      });
      const { newRoot, operationId } = await server.mergeSegmentsWithOperation(
        {
          segmentId: sinkSv,
          rootId: sinkRoot,
          position: Float32Array.from(sink.coordinate_nm),
        },
        {
          segmentId: sourceSv,
          rootId: sourceRoot,
          position: Float32Array.from(source.coordinate_nm),
        },
      );
      // Swap the two old roots for the new one, exactly as graphene's own
      // submitMerge does -- otherwise the view keeps showing retired roots.
      const oldValues = new Uint64Set();
      oldValues.add(sinkRoot);
      oldValues.add(sourceRoot);
      const newValues = new Uint64Set();
      newValues.add(newRoot);
      this.ownEdit(() => {
        graph!.state?.replaceSegments(oldValues, newValues);
        const group = this.layer.displayState.segmentationGroupState.value;
        group.selectedSegments.delete(sinkRoot);
        group.selectedSegments.delete(sourceRoot);
        group.selectedSegments.add(newRoot);
        group.visibleSegments.set(newRoot, true);
      });

      // The neuron under review now has a new root; later merges in this
      // session must target it.
      console.log("[twig-capture] merge ok", {
        candidate: candidate.id,
        newRoot: String(newRoot),
        operationId,
      });
      this.targetRoots.set(result.name, newRoot);
      this.mergeStatus.set(key, "done");
      // The edit happened; record the judgement that caused it, carrying the
      // operation id so the row is evidence of a real graph change rather
      // than an opinion. Deliberately not awaited into the merge's own
      // success: a failed POST must not make a completed merge look failed.
      // recordDecision reports its own failure loudly.
      //
      // Skipped silently on a volume that is not in DECISION_ALLOWLIST: the
      // merge is perfectly valid there, there is simply nowhere vouched-for
      // to record it, and an error toast after every successful merge would
      // train people to ignore toasts.
      if (this.decisionsAllowed()) {
        void this.recordDecision(candidate, "merge", operationId);
      }
      if (operationId !== undefined) {
        this.history.set(key, { operationId, undone: false });
      }
      StatusMessage.showTemporaryMessage(
        `Merged ${candidate.target_segment_id} \u2192 new root ${newRoot}`,
        5000,
      );
      this.render();
      return "merged";
    } catch (e) {
      this.mergeStatus.set(key, "failed");
      // Full object, not just the message: graphene wraps HttpError and the
      // server's reason is often only in the stack or the response body.
      console.error("[twig-capture] merge failed", candidate.id, e);
      StatusMessage.showTemporaryMessage(
        `Twig Capture: merge failed for ${candidate.target_segment_id}: ${e}`,
        6000,
      );
      this.render();
      return "failed";
    }
  }

  /**
   * The neuron's root TODAY, by majority over up to five of its contact
   * points other than `except`'s (sink supervoxels spread through the file).
   * undefined when there are fewer than two to ask or no graph.
   */
  private async neuronRootConsensus(
    result: TwigResult,
    except?: TwigCandidate,
  ): Promise<{ root: bigint; agree: number; total: number } | undefined> {
    const graph = this.graphene();
    if (graph?.getRoot === undefined) return undefined;
    const own = except?.merge_sink?.supervoxel_id;
    // Level-2+ rows (via_root) have their sink on a suggestion, not on the
    // neuron, so they say nothing about the neuron's root.
    const svs = [
      ...new Set(
        result.candidates
          .filter((c) => !c.via_root && (c.hop ?? 1) <= 1)
          .map((c) => c.merge_sink?.supervoxel_id)
          .filter((sv): sv is string => !!sv && sv !== own),
      ),
    ];
    if (svs.length < 2) return undefined;
    const n = Math.min(5, svs.length);
    const picked = Array.from(
      { length: n },
      (_, i) => svs[Math.floor((i * svs.length) / n)],
    );
    const roots = await Promise.all(
      picked.map((sv) =>
        graph.getRoot!(BigInt(sv)).then(
          (r) => r,
          () => undefined,
        ),
      ),
    );
    const counts = new Map<bigint, number>();
    for (const r of roots) {
      if (r !== undefined) counts.set(r, (counts.get(r) ?? 0) + 1);
    }
    let best: bigint | undefined;
    let agree = 0;
    for (const [r, c] of counts) {
      if (c > agree) {
        best = r;
        agree = c;
      }
    }
    const total = [...counts.values()].reduce((a, b) => a + b, 0);
    if (best === undefined || total < 2) return undefined;
    return { root: best, agree, total };
  }

  // ---- stage, review, merge staged ------------------------------------------

  /** The staged list, if it belongs to the neuron on screen. */
  private activeStage() {
    const stage = this.stage;
    return stage !== undefined && stage.file === this.active?.name
      ? stage
      : undefined;
  }

  private isJoined(c: TwigCandidate): boolean {
    const st = this.mergeStatus.get(rowKey(c));
    return st === "done" || st === "already" || st === "merging";
  }

  /**
   * What Stage all would take: the listed candidates of the open neuron that
   * can still be merged. Leaves out rows already joined, rows someone has
   * judged "no_merge", "new_neuron" or "split_needed", and rows without
   * merge endpoints.
   */
  private stageable(): TwigCandidate[] {
    if (this.active === undefined) return [];
    return this.filtered().filter(
      (c) =>
        !this.isJoined(c) &&
        !isNotAMerge(this.decisionFor(c)?.tag) &&
        this.decisionFor(c)?.tag !== "split_needed" &&
        !!c.merge_sink &&
        !!c.merge_source,
    );
  }

  /** Staged, still ticked, and not merged yet. */
  private stagedPicks(): TwigCandidate[] {
    const stage = this.activeStage();
    if (stage === undefined) return [];
    return stage.rows.filter(
      (c) => stage.selected.has(rowKey(c)) && !this.isJoined(c),
    );
  }

  private updateBulkButtons() {
    const b = this.mergeAllButton;
    const stage = this.activeStage();
    this.clearStageButton.style.display = stage ? "" : "none";
    this.clearStageButton.disabled = this.batch !== undefined;
    if (this.batch !== undefined) {
      const { done, total, stop } = this.batch;
      b.textContent = stop ? "Stopping\u2026" : `Stop (${done}/${total})`;
      b.disabled = stop;
      b.title =
        "Stop after the merge in progress. Merges already made stay; each " +
        "has its own Undo.";
      return;
    }
    if (stage !== undefined) {
      const n = this.stagedPicks().length;
      b.textContent = `Merge staged (${n})`;
      b.disabled = n === 0;
      b.title =
        `Merge the ticked candidates into ${this.active?.members ? "their neurons, neuron by neuron" : "this neuron"}, one at a time. Each is ` +
        "re-resolved from its supervoxels first; one that changed hands " +
        "since inference, or whose contact is no longer on the neuron, is " +
        "skipped and left for you. A failed merge stops the run.";
      return;
    }
    if (this.active === undefined) {
      b.textContent = "Stage all";
      b.disabled = true;
      b.title =
        "Pick one neuron (root id above) to stage its candidates. Bulk " +
        "merging works one neuron at a time.";
      return;
    }
    const n = this.stageable().length;
    b.textContent = `Stage all (${n})`;
    b.disabled =
      n === 0 ||
      this.checking !== undefined ||
      this.graphene()?.graphServer === undefined;
    b.title =
      `Stage every listed candidate of ${this.active.members ? `these ${this.active.members.length} neurons` : "this neuron"} that can still be merged ` +
      "(not already joined, not judged no_merge, new_neuron or " +
      "split_needed) and show them " +
      "all in the " +
      "view. Nothing is edited: untick the ones you do not want, then Merge " +
      "staged.";
  }

  /** Select-and-show one segment: the segment list keeps only selected ones visible. */
  private showSegment(root: bigint) {
    const group = this.layer.displayState.segmentationGroupState.value;
    this.ownEdit(() => {
      group.selectedSegments.add(root);
      group.visibleSegments.set(root, true);
    });
  }

  private hideSegment(root: bigint) {
    const group = this.layer.displayState.segmentationGroupState.value;
    this.ownEdit(() => {
      group.visibleSegments.delete(root);
      group.selectedSegments.delete(root);
    });
  }

  /** Change the segment sets without it reading as the reviewer's doing. */
  private ownEdit(fn: () => void) {
    this.ownSegmentEdits++;
    try {
      fn();
    } finally {
      this.ownSegmentEdits--;
    }
  }

  /** The staged row whose candidate is shown in the view as `root`. */
  private stagedKeyForRoot(root: bigint): string | undefined {
    const stage = this.activeStage();
    if (stage === undefined || isStagedNeuron(stage, root)) return undefined;
    for (const [key, r] of stage.roots) if (r === root) return key;
    return undefined;
  }

  /**
   * The mouse is over a segment in the view: if it is a staged candidate,
   * mark its row and bring it into sight. Pointing at nothing, or at the
   * neuron, keeps the last mark, so it is still there once the mouse has
   * crossed to the panel to untick it.
   */
  private onViewHover() {
    const root = this.layer.displayState.segmentSelectionState.value;
    if (root === undefined) return;
    const key = this.stagedKeyForRoot(root);
    if (key === undefined || key === this.hoverKey) return;
    if (this.hoverKey !== undefined) {
      this.stageRowEls
        .get(this.hoverKey)
        ?.classList.remove("neuroglancer-twig-capture-row-hover");
    }
    this.hoverKey = key;
    const el = this.stageRowEls.get(key);
    if (el !== undefined) {
      el.classList.add("neuroglancer-twig-capture-row-hover");
      el.scrollIntoView({ block: "nearest" });
    }
  }

  /**
   * The visible set changed. A staged candidate hidden by the reviewer -- in
   * the 3D view, the segment list, or by clearing the layer -- is unticked;
   * one shown again is ticked. Changes the tab makes itself, and anything
   * during a Merge staged run, are ignored.
   */
  private onVisibleChanged(
    x: bigint | bigint[] | BigUint64Array | null,
    add: boolean,
  ) {
    if (this.ownSegmentEdits > 0 || this.batch !== undefined) return;
    const stage = this.activeStage();
    if (stage === undefined || stage.roots.size === 0) return;
    const ids =
      x === null ? null : new Set(typeof x === "bigint" ? [x] : Array.from(x));
    let changed = false;
    for (const c of stage.rows) {
      const key = rowKey(c);
      const root = stage.roots.get(key);
      if (
        root === undefined ||
        isStagedNeuron(stage, root) ||
        this.isJoined(c)
      ) {
        continue;
      }
      if (ids !== null && !ids.has(root)) continue;
      const want = ids === null ? false : add;
      if (stage.selected.has(key) === want) continue;
      if (want) stage.selected.add(key);
      else stage.selected.delete(key);
      changed = true;
    }
    if (changed && !this.stageRenderPending) {
      // One redraw for a burst of changes (a clear, a multi-select).
      this.stageRenderPending = true;
      setTimeout(() => {
        this.stageRenderPending = false;
        this.render();
      }, 0);
    }
  }

  /**
   * Freeze the listed, mergeable candidates into a ticked list, and make the
   * view show exactly the proposal: everything else in the layer is
   * deselected, then the neuron (at its root today) and every staged
   * candidate (at ITS root today, from its own supervoxel -- the file's
   * segment ids may be retired) are selected.
   */
  private async stageAll() {
    const result = this.active;
    if (result === undefined || this.stage !== undefined) return;
    const rows = this.stageable();
    if (rows.length === 0) return;
    const stage = {
      file: result.name,
      rows,
      selected: new Set(rows.map(rowKey)),
      roots: new Map<string, bigint>(),
      neurons: new Map<string, bigint>(),
    };
    this.stage = stage;
    this.render();
    const graph = this.graphene();
    if (graph?.getRoot === undefined) return;
    const group = this.layer.displayState.segmentationGroupState.value;
    this.hoverKey = undefined;
    this.ownEdit(() => {
      group.visibleSegments.clear();
      group.selectedSegments.clear();
    });
    // Each neuron in the list (one, or several in a combined list), at its
    // root today.
    for (const file of rowFiles(rows, result)) {
      const own = await this.resultForFile(file);
      if (this.stage !== stage) return;
      const consensus = own ? await this.neuronRootConsensus(own) : undefined;
      if (this.stage !== stage) return;
      const root = consensus?.root ?? this.targetRoots.get(file);
      if (root !== undefined) {
        stage.neurons.set(file, root);
        this.showSegment(root);
      }
    }
    const queue = [...rows];
    const worker = async () => {
      for (;;) {
        const c = queue.shift();
        if (c === undefined || this.stage !== stage) return;
        try {
          const root = await graph.getRoot!(
            BigInt(c.merge_source!.supervoxel_id),
          );
          if (this.stage !== stage) return;
          if (root === stage.neurons.get(c._file ?? result.name)) {
            // Already part of the neuron: nothing to merge.
            this.mergeStatus.set(rowKey(c), "already");
            continue;
          }
          stage.roots.set(rowKey(c), root);
          if (stage.selected.has(rowKey(c))) this.showSegment(root);
        } catch {
          // not shown; the merge re-resolves it anyway
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(8, rows.length) }, worker));
    if (this.stage === stage) this.render();
  }

  /** Tick or untick one staged row; its segment is shown or hidden to match. */
  private toggleStaged(c: TwigCandidate, on: boolean) {
    const stage = this.activeStage();
    if (stage === undefined) return;
    const key = rowKey(c);
    if (on) stage.selected.add(key);
    else stage.selected.delete(key);
    const root = stage.roots.get(key);
    if (
      root !== undefined &&
      !isStagedNeuron(stage, root) &&
      !this.isJoined(c)
    ) {
      if (on) this.showSegment(root);
      else this.hideSegment(root);
    }
    this.render();
  }

  /**
   * Drop the staged list. Candidates not merged are deselected; the neuron
   * (with whatever was merged into it) stays.
   */
  private clearStage() {
    const stage = this.stage;
    if (stage === undefined || this.batch !== undefined) return;
    for (const c of stage.rows) {
      const root = stage.roots.get(rowKey(c));
      if (root !== undefined && !this.isJoined(c)) this.hideSegment(root);
    }
    this.stage = undefined;
    this.render();
  }

  /**
   * Merge the ticked staged candidates, one after another.
   *
   * Sequential because each merge gives the neuron a new root, which the
   * next merge must name. The neuron's root is established ONCE, up front,
   * from several of its contact points, and the run refuses to start if they
   * disagree (the neuron may have been split). From then on it is advanced
   * to each merge's new root, and every candidate's contact must resolve to
   * it -- anything else is skipped, never merged.
   */
  private async mergeStaged() {
    const result = this.active;
    const stage = this.activeStage();
    if (result === undefined || stage === undefined) return;
    if (this.batch !== undefined) return;
    const picks = this.stagedPicks();
    if (picks.length === 0) return;
    // One group per neuron, in list order. Each neuron's root is
    // established ONCE, up front, from several of its own contact points;
    // the run refuses to start if any neuron's points disagree.
    const groups = new Map<string, TwigCandidate[]>();
    for (const c of picks) {
      const file = c._file ?? result.name;
      let g = groups.get(file);
      if (g === undefined) groups.set(file, (g = []));
      g.push(c);
    }
    const expectedRoot = new Map<string, bigint>();
    for (const file of groups.keys()) {
      const own = await this.resultForFile(file);
      const neuron = own ? await this.neuronRootConsensus(own) : undefined;
      const label = own?.root_id ?? file;
      if (neuron === undefined) {
        StatusMessage.showTemporaryMessage(
          `Twig Capture: could not establish neuron ${label}'s root today ` +
            "from its contact points, so nothing was merged. Merge one at a time.",
          8000,
        );
        return;
      }
      if (neuron.agree !== neuron.total) {
        StatusMessage.showTemporaryMessage(
          `Twig Capture: nothing merged. Only ${neuron.agree} of ` +
            `${neuron.total} of neuron ${label}'s contact points are on ` +
            `${neuron.root}; it may have been split since inference. Check ` +
            "it, then merge one at a time.",
          10000,
        );
        return;
      }
      expectedRoot.set(file, neuron.root);
    }
    const syn = picks.reduce((a, c) => a + (c.cand_synapses_total ?? 0), 0);
    const into =
      groups.size === 1
        ? `neuron ${(await this.resultForFile([...groups.keys()][0]))?.root_id ?? result.root_id}`
        : `${groups.size} neurons`;
    const ok = window.confirm(
      `Merge ${picks.length} staged candidate(s) into ${into}?\n` +
        `  ${syn} synapses on them in total.\n\n` +
        "They are merged one at a time" +
        (groups.size > 1 ? ", neuron by neuron" : "") +
        ". Before each, both sides are " +
        "re-resolved from their supervoxels; a candidate that has changed " +
        "hands since inference, or whose contact is no longer on its " +
        "neuron, is SKIPPED and left for you. A failed merge stops the run, " +
        "and Stop ends it after the current merge.\n\n" +
        "Each merge edits the shared segmentation and is attributed to you.",
    );
    if (!ok) return;

    const batch = { stop: false, done: 0, total: picks.length };
    this.batch = batch;
    const tally = { merged: 0, already: 0, skipped: 0 };
    let failed: TwigCandidate | undefined;
    try {
      outer: for (const [file, rows] of groups) {
        let expected = expectedRoot.get(file)!;
        for (const c of rows) {
          if (batch.stop) break outer;
          // Unticked while the run was going.
          if (!stage.selected.has(rowKey(c))) {
            batch.total--;
            continue;
          }
          const out = await this.doMerge(c, { neuron: expected });
          batch.done++;
          if (out === "merged") {
            tally.merged++;
            expected = this.targetRoots.get(file) ?? expected;
            stage.neurons.set(file, expected);
          } else if (out === "already") {
            tally.already++;
          } else if (out === "skipped") {
            tally.skipped++;
          } else {
            failed = c;
            break outer;
          }
          this.updateBulkButtons();
        }
      }
    } finally {
      this.batch = undefined;
      this.render();
    }
    const bits = [`${tally.merged} merged`];
    if (tally.already) bits.push(`${tally.already} already joined`);
    if (tally.skipped) bits.push(`${tally.skipped} skipped (see rows)`);
    const left = this.stagedPicks().length;
    StatusMessage.showTemporaryMessage(
      `Twig Capture: ${bits.join(", ")}` +
        (failed
          ? `; stopped at ${failed.target_segment_id}, which failed`
          : batch.stop
            ? "; stopped"
            : "") +
        (left ? `. ${left} still staged.` : "."),
      10000,
    );
  }

  /**
   * The graph check for a combined list: each member neuron's root today
   * (from one of its own level-1 sinks), then every row against ITS neuron.
   */
  private async refreshMembers(result: TwigResult, atLoad: boolean) {
    const graph = this.graphene();
    if (graph?.getRoot === undefined || result.members === undefined) return;
    const rows = this.filtered();
    this.refreshButton.disabled = true;
    this.refreshButton.textContent = "Refreshing\u2026";
    if (atLoad) {
      this.checking = { done: 0, total: rows.length };
      this.render();
    }
    try {
      for (const file of result.members) {
        const own = await this.resultForFile(file);
        const sink = own?.candidates.find(
          (c) => c.merge_sink && !c.via_root && (c.hop ?? 1) <= 1,
        )?.merge_sink;
        const root = sink
          ? await graph.getRoot(BigInt(sink.supervoxel_id))
          : own
            ? await graph.getRoot(BigInt(own.root_id))
            : undefined;
        if (root !== undefined) this.targetRoots.set(file, root);
      }
      if (atLoad && this.active !== result) return;
      let checked = 0;
      const queue = [...rows];
      const worker = async () => {
        for (;;) {
          const c = queue.shift();
          if (c === undefined) return;
          const target = this.targetRoots.get(c._file ?? "");
          const sv = c.merge_source?.supervoxel_id;
          try {
            const root = sv
              ? await graph.getRoot!(BigInt(sv))
              : await graph.getRoot!(BigInt(c.target_segment_id));
            const key = rowKey(c);
            if (target !== undefined && root === target) {
              if (this.mergeStatus.get(key) !== "done") {
                this.mergeStatus.set(key, "already");
              }
            } else if (this.mergeStatus.get(key) === "already") {
              this.mergeStatus.delete(key);
            }
          } catch {
            // leave this row alone
          }
          checked++;
          if (this.checking !== undefined) this.checking.done = checked;
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(8, rows.length || 1) }, worker),
      );
    } finally {
      this.checking = undefined;
      this.refreshButton.disabled = false;
      this.refreshButton.textContent = "Refresh";
      this.render();
    }
  }

  /** A loaded result by file name (the combined list's member neurons). */
  private async resultForFile(file: string): Promise<TwigResult | undefined> {
    try {
      return await this.fetchResultFile(file);
    } catch {
      return undefined;
    }
  }

  /** The rows to filter: the active neuron's, else the all-neurons table. */
  private sourceRows(): TwigCandidate[] {
    if (this.active !== undefined) return this.active.candidates;
    if (this.isAllMode() && this.table !== undefined) return this.table.rows;
    return [];
  }

  private filtered(): TwigCandidate[] {
    // The all-neurons table can hold millions of rows (BANC: 1.56M), and
    // every merge or verdict re-renders, so its filtered + sorted list is
    // cached until a filter, the table or the decisions change.
    if (this.active === undefined && this.table !== undefined) {
      const filterKey = JSON.stringify([
        this.probOp,
        this.minProb,
        this.mito,
        this.review,
        this.sortKey,
        this.neuronFiltersActive() ? this.neuronFilters : null,
        this.neuronMeta?.generated ?? null,
      ]);
      const key =
        filterKey +
        `|${this.decisionsAllowed() ? this.allDecisionsVersion : -1}`;
      const cached = this.allFiltered;
      if (cached?.table === this.table && cached.key === key) {
        return cached.rows;
      }
      const rows = this.filterAndSort(this.sourceRows());
      this.allFiltered = { table: this.table, key, rows };
      // New filters: start again at the first page. Only new decisions (a
      // verdict just recorded): keep the reviewer's place.
      if (cached?.key.slice(0, cached.key.lastIndexOf("|")) !== filterKey) {
        this.allShown = ALL_NEURONS_PAGE;
      }
      return rows;
    }
    return this.filterAndSort(this.sourceRows());
  }

  private filterAndSort(source: TwigCandidate[]): TwigCandidate[] {
    const rows = source.filter((c) => {
      if (!passesProb(c.prob_merge, this.probOp, this.minProb)) return false;
      // Neuron filters pick neurons, so they apply to the all-neurons list
      // only; an opened neuron was chosen explicitly.
      if (c._lite && !this.neuronPasses(c._root ?? "")) return false;
      // No L2-node filter. The candidate search already caps it at
      // max_candidate_l2_nodes (20), so every row here is a twig by
      // construction and the control only ever hid real candidates.
      // num_l2_nodes is still shown on every row, and the changed-hands
      // dialog still quotes it.
      const mito = c.likely_mito === true;
      if (this.mito === "hide" && mito) return false;
      if (this.mito === "only" && !mito) return false;
      // Review state. Skipped entirely when decisions are not enabled for
      // this layer: this.decisions is then empty by construction, and
      // filtering on it would hide every row while claiming they are all
      // unreviewed, which is a lie rather than an empty result.
      if (this.review !== "any" && this.decisionsAllowed()) {
        const d = this.decisionFor(c);
        if (this.review === "none" && d !== undefined) return false;
        if (this.review === "done" && d === undefined) return false;
        if (
          [
            "merge",
            "no_merge",
            "unsure",
            "new_neuron",
            "split_needed",
          ].includes(this.review) &&
          d?.tag !== this.review
        ) {
          return false;
        }
      }
      return true;
    });
    const key = this.sortKey;
    rows.sort((a, b) => {
      if (key === "prob") return b.prob_merge - a.prob_merge;
      if (key === "size")
        return (b.cand_size_um3 ?? 0) - (a.cand_size_um3 ?? 0);
      if (key === "synapses") {
        const d = candSynapses(b) - candSynapses(a);
        // Ties are the common case -- most candidates carry no synapses at
        // all -- so fall through to probability rather than leaving hundreds
        // of rows in whatever order the file happened to list them.
        return d !== 0 ? d : b.prob_merge - a.prob_merge;
      }
      return (b.contact_voxels ?? 0) - (a.contact_voxels ?? 0);
    });
    return rows;
  }

  private render() {
    // Keeps the facet selects in step with state restored from a link, and
    // disables `review` once we know whether any decisions can be loaded.
    this.renderFilters();
    // A staged list belongs to one neuron; leaving it drops the list (the
    // segments it showed stay visible -- hiding them unasked would be worse).
    if (
      this.stage !== undefined &&
      this.batch === undefined &&
      this.stage.file !== this.active?.name
    ) {
      this.stage = undefined;
    }
    this.updateBulkButtons();

    // PRESERVE THE SCROLL POSITION ACROSS A RE-RENDER.
    //
    // listEl is the scroller, and render() empties it with removeChildren
    // before rebuilding every row, which resets scrollTop to 0. Every merge
    // and every verdict calls render(), so acting on a row two hundred down
    // threw the reviewer back to the top and they had to find their place
    // again -- on a list of several hundred candidates that is the difference
    // between a usable tool and an infuriating one.
    //
    // Only when the SAME file is still loaded. Switching files should start
    // at the top, and a restored scrollTop from the previous neuron's list
    // would be meaningless. This also keeps out of the way of the
    // scrollIntoView that a shared link performs on first render, which runs
    // exactly when the file has just changed.
    // The all-neurons list counts as one "file" for this purpose.
    const listName =
      this.active?.name ?? (this.isAllMode() ? "\u0000all" : undefined);
    const sameFile =
      listName !== undefined && this.lastRenderedFile === listName;
    const keptScrollTop = this.listEl.scrollTop;
    const restoreScroll = () => {
      // The browser clamps to the new scrollHeight, so a list that shrank
      // under a filter change lands at the bottom rather than out of bounds.
      if (sameFile) this.listEl.scrollTop = keptScrollTop;
      this.lastRenderedFile = listName;
    };
    // file chooser
    removeChildren(this.fileSelect);
    for (const result of this.results) {
      const option = document.createElement("option");
      option.textContent =
        `${result.root_id}  (${result.candidates.length})` +
        (result.name.startsWith(LOCAL_RESULT_PREFIX) ? "  [this browser]" : "");
      option.title = result.name;
      this.fileSelect.appendChild(option);
    }
    this.fileSelect.style.display = this.results.length ? "" : "none";
    if (this.active !== undefined) {
      this.fileSelect.selectedIndex = this.results.indexOf(this.active);
    }

    removeChildren(this.summary);
    removeChildren(this.listEl);

    const result = this.active;
    if (result === undefined) {
      if (this.isAllMode()) {
        this.renderAll();
      } else {
        this.countEl.textContent = "";
      }
      restoreScroll();
      return;
    }

    const bits = [`neuron ${result.root_id}`];
    const annot = this.neuronMeta?.neurons.get(result.root_id);
    if (neuronLabel(annot)) bits.push(neuronLabel(annot));
    if (result.models_used) bits.push(`model ${result.models_used}`);
    if (result.connectivity !== undefined)
      bits.push(`${result.connectivity === 3 ? 26 : 6}-conn`);
    if (result.merge_threshold !== undefined)
      bits.push(`thr ${result.merge_threshold}`);
    if (result.timestamp) bits.push(result.timestamp.slice(0, 16));
    this.summary.textContent = bits.join("  ·  ");

    const stage = this.activeStage();
    const rows = stage !== undefined ? stage.rows : this.filtered();
    this.stageRowEls.clear();
    if (stage !== undefined) {
      const picks = this.stagedPicks();
      const syn = picks.reduce((a, c) => a + (c.cand_synapses_total ?? 0), 0);
      this.countEl.textContent =
        `${picks.length} of ${rows.length} staged to merge` +
        (syn ? `  ·  ${syn} syn` : "");
    } else {
      this.countEl.textContent = `${rows.length} of ${result.candidates.length} candidates`;
    }

    if (this.checking !== undefined) {
      const { done, total } = this.checking;
      const wait = document.createElement("div");
      wait.classList.add("neuroglancer-twig-capture-checking");
      const spinner = document.createElement("span");
      spinner.classList.add("neuroglancer-twig-capture-spinner");
      const text = document.createElement("span");
      text.textContent =
        `Checking ${total} candidate(s) against the chunkedgraph\u2026 ` +
        `${done}/${total}`;
      wait.appendChild(spinner);
      wait.appendChild(text);
      const why = document.createElement("div");
      why.classList.add("neuroglancer-twig-capture-hint");
      why.textContent =
        "Resolving today's root for the neuron and each candidate, so " +
        "anything already merged is marked before you can click it.";
      this.listEl.appendChild(wait);
      this.listEl.appendChild(why);
      restoreScroll();
      return;
    }

    if (stage !== undefined) {
      const note = document.createElement("div");
      note.classList.add("neuroglancer-twig-capture-hint");
      note.classList.add("neuroglancer-twig-capture-stage-note");
      note.textContent =
        `Staged: ${rows.length} candidate(s) of ${this.active?.members ? `${this.active.members.length} neurons` : "this neuron"}, frozen from the ` +
        "filters when staged. The view shows only the neuron and the ticked " +
        "candidates. Untick any you do not want, then Merge staged. Clear " +
        "returns to the filtered list.";
      this.listEl.appendChild(note);
    }
    for (const candidate of rows) {
      this.listEl.appendChild(this.makeRow(candidate));
    }
    if (rows.length === 0) {
      const empty = document.createElement("div");
      empty.classList.add("neuroglancer-twig-capture-hint");
      empty.textContent = "No candidates pass the current filters.";
      this.listEl.appendChild(empty);
    }
    restoreScroll();
  }

  /** The all-neurons list: every neuron's candidates that pass the filters. */
  private renderAll() {
    const hint = (text: string) => {
      const el = document.createElement("div");
      el.classList.add("neuroglancer-twig-capture-hint");
      el.textContent = text;
      this.listEl.appendChild(el);
    };
    if (this.tableState !== "ready" || this.table === undefined) {
      this.countEl.textContent = "";
      this.summary.textContent = "all neurons";
      if (this.tableState === "loading" || this.tableState === "none") {
        hint(
          `Loading every neuron's candidates (${CANDIDATE_TABLE_FILE})… ` +
            "or pick a root id above.",
        );
      } else if (this.tableState === "missing") {
        hint(
          `This catalog has no ${CANDIDATE_TABLE_FILE}, so candidates can ` +
            "only be listed one neuron at a time: pick a root id above. " +
            "Re-run scripts/make_results_index.py on the directory to add it.",
        );
      } else {
        hint(
          `Could not read ${CANDIDATE_TABLE_FILE}: ${this.tableError}. ` +
            "Pick a root id above.",
        );
      }
      return;
    }
    const { files, rows: all } = this.table;
    const neurons = this.neuronFiltersActive()
      ? `${files.filter((f) => this.neuronPasses(f.root_id)).length} of ` +
        `${files.length} neuron(s)`
      : `${files.length} neuron(s)`;
    this.summary.textContent =
      `all neurons  ·  ${neurons}  ·  latest run of each` +
      (this.table.generated
        ? `  ·  table ${this.table.generated.slice(0, 16)}`
        : "");
    const rows = this.filtered();
    const shown = rows.slice(0, this.allShown);
    this.countEl.textContent =
      rows.length > shown.length
        ? `${shown.length} of ${rows.length} matching shown (${all.length} candidates)`
        : `${rows.length} of ${all.length} candidates`;
    for (const candidate of shown) {
      this.listEl.appendChild(this.makeRow(candidate));
    }
    if (rows.length === 0) {
      hint("No candidates pass the current filters.");
    } else if (rows.length > shown.length) {
      const more = document.createElement("button");
      more.classList.add("neuroglancer-twig-capture-merge");
      more.classList.add("neuroglancer-twig-capture-more");
      const next = Math.min(ALL_NEURONS_PAGE, rows.length - shown.length);
      more.textContent = `Show next ${next} (${rows.length - shown.length} more)`;
      more.addEventListener("click", () => {
        this.allShown += ALL_NEURONS_PAGE;
        this.render();
      });
      this.listEl.appendChild(more);
    }
  }

  private makeRow(candidate: TwigCandidate): HTMLElement {
    const row = document.createElement("div");
    row.classList.add("neuroglancer-twig-capture-row");
    if (candidate.likely_mito) {
      row.classList.add("neuroglancer-twig-capture-row-mito");
    }
    // The row being looked at: either clicked here, or pointed at by a shared
    // link. Both are the same thing -- twigCapture.candidate is what a link
    // carries and what goTo sets. Marked, and scrolled to once --
    // a link that says "look at this one" should not land the reader at the
    // top of four hundred rows. Cleared after scrolling so that later
    // re-renders, e.g. after a merge, do not keep yanking the list back.
    if (
      this.layer.twigCapture.candidate &&
      candidate.target_segment_id === this.layer.twigCapture.candidate
    ) {
      row.classList.add("neuroglancer-twig-capture-row-shared");
      if (!this.scrolledToShared) {
        this.scrolledToShared = true;
        setTimeout(() => row.scrollIntoView({ block: "center" }), 0);
      }
    }

    const main = document.createElement("div");
    main.classList.add("neuroglancer-twig-capture-main");

    // Staged list: a tick per row, for what Merge staged will include.
    const stage = this.activeStage();
    if (stage !== undefined) {
      const box = document.createElement("input");
      box.type = "checkbox";
      box.classList.add("neuroglancer-twig-capture-stagebox");
      box.checked = stage.selected.has(rowKey(candidate));
      box.disabled = this.isJoined(candidate);
      box.title = box.disabled
        ? "Already part of the neuron."
        : "Ticked rows are merged by Merge staged. Untick to leave this one " +
          "out; it is hidden in the view too.";
      box.addEventListener("click", (event: MouseEvent) => {
        event.stopPropagation();
        this.toggleStaged(candidate, box.checked);
      });
      main.appendChild(box);
      if (!box.checked && !box.disabled) {
        row.classList.add("neuroglancer-twig-capture-row-unstaged");
      }
      // Linked to the view: this row is marked when its fragment is pointed
      // at there, and pointing at the row highlights the fragment.
      const key = rowKey(candidate);
      this.stageRowEls.set(key, row);
      if (key === this.hoverKey) {
        row.classList.add("neuroglancer-twig-capture-row-hover");
      }
      row.addEventListener("mouseenter", () => {
        const root = stage.roots.get(key);
        if (root !== undefined) {
          this.layer.displayState.segmentSelectionState.set(root);
        }
      });
      row.addEventListener("mouseleave", () => {
        this.layer.displayState.segmentSelectionState.set(null);
      });
    }

    const prob = document.createElement("span");
    prob.classList.add("neuroglancer-twig-capture-prob");
    prob.textContent = fmt(candidate.prob_merge);
    main.appendChild(prob);

    const id = document.createElement("span");
    id.classList.add("neuroglancer-twig-capture-id");
    id.textContent = candidate.target_segment_id;
    main.appendChild(id);

    // All-neurons view: which neuron this candidate would join.
    if (this.active === undefined && candidate._root) {
      const neuron = document.createElement("span");
      neuron.classList.add("neuroglancer-twig-capture-tag");
      neuron.classList.add("neuroglancer-twig-capture-tag-neuron");
      const annot = this.neuronMeta?.neurons.get(candidate._root);
      const label = neuronLabel(annot);
      neuron.textContent = `→ ${candidate._root}${label ? ` ${label}` : ""}`;
      neuron.title =
        `Candidate for neuron ${candidate._root} (its root when inference ` +
        `ran; file ${candidate._file}). Merge joins it to that neuron.` +
        (this.neuronMeta ? `\n\n${neuronTooltip(annot)}` : "");
      main.appendChild(neuron);
    }

    if ((candidate.hop ?? 1) > 1 && candidate.parent_root === undefined) {
      const tag = document.createElement("span");
      tag.classList.add("neuroglancer-twig-capture-tag");
      tag.classList.add("neuroglancer-twig-capture-tag-hop");
      tag.textContent = `level ${candidate.hop}`;
      tag.title =
        `A level-${candidate.hop} suggestion: it attaches to ` +
        (candidate.via_root
          ? `${candidate.via_root}, an earlier suggestion, `
          : "the neuron ") +
        "and was found by treating the earlier confident suggestions as " +
        "already merged. Merge that one first.";
      main.appendChild(tag);
    }

    if (candidate.likely_mito) {
      const tag = document.createElement("span");
      tag.classList.add("neuroglancer-twig-capture-tag");
      tag.textContent = "mito?";
      tag.title =
        "enclosure > 0.6 and mean EM < 100 -- a provisional flag for triage, not a filter";
      main.appendChild(tag);
    }

    // A synapse whose two partners ARE this pair. Two segments that synapse
    // on each other are most likely two different neurons, so this is
    // counter-evidence and belongs next to the confidence, not buried in the
    // metadata line. Normally zero; when it is not, look before merging.
    const wide = candidate.synapses_wide ?? candidate.synapses_context;
    const between = Math.max(
      candidate.synapses_local?.between ?? 0,
      wide?.between ?? 0,
    );
    if (between > 0) {
      const tag = document.createElement("span");
      tag.classList.add("neuroglancer-twig-capture-tag");
      tag.classList.add("neuroglancer-twig-capture-tag-warn");
      tag.textContent = `${between} syn A\u2194B`;
      tag.title =
        `${between} synapse(s) near this contact run between the neuron and ` +
        "this candidate. A cell rarely synapses onto itself, so this is " +
        "evidence they are different neurons. Check before merging.";
      main.appendChild(tag);
    }

    // What merging this would actually recover. In the main row, not the
    // metadata line, because for BANC it is the point: completion rate
    // counts synapses whose two partners are both identified, so attaching
    // a fragment converts this many. Shown only when known and non-zero --
    // most candidates carry none, and a row of "0 syn" would be noise.
    const totalSyn = candidate.cand_synapses_total;
    if (totalSyn != null && totalSyn > 0) {
      const chip = document.createElement("span");
      chip.classList.add("neuroglancer-twig-capture-tag");
      chip.classList.add("neuroglancer-twig-capture-tag-syn");
      chip.textContent = `${totalSyn} syn`;
      chip.title =
        `${totalSyn} synapses on this candidate's whole extent -- what ` +
        "attaching it would add to the connectome's completeness. Counted " +
        "over every supervoxel of the candidate, not just the scoring " +
        "window, so it does not match the per-window numbers below.";
      main.appendChild(chip);
    }

    const key = rowKey(candidate);
    const status = this.mergeStatus.get(key);
    const decided = this.decisionFor(candidate);
    const pending = this.deciding.has(key);
    const allowed = this.decisionsAllowed();

    if (status === "skipped") {
      const chip = document.createElement("span");
      chip.classList.add("neuroglancer-twig-capture-tag");
      chip.classList.add("neuroglancer-twig-capture-tag-warn");
      chip.textContent = "skipped";
      chip.title =
        `Merge staged left this one alone: ${this.skipReason.get(key) ?? "?"}. ` +
        "Look at it, then Merge it on its own if it is right.";
      main.appendChild(chip);
    }

    // The recorded verdict stays on the TOP row, beside the other chips: it
    // is a fact about the candidate, like the synapse count, and belongs with
    // the things a reviewer scans rather than with the controls.
    if (decided !== undefined) {
      const chip = document.createElement("span");
      chip.classList.add("neuroglancer-twig-capture-tag");
      if (isNotAMerge(decided.tag) || decided.tag === "split_needed") {
        chip.classList.add("neuroglancer-twig-capture-tag-warn");
      }
      chip.textContent = decided.tag;
      chip.title =
        `Recorded in CAVE as "${decided.tag}" (annotation ${decided.id}). ` +
        "Reviewing again adds a new row; the latest wins.";
      main.appendChild(chip);
    }
    row.appendChild(main);

    // ---- SECOND LINE: the actions ----------------------------------------
    //
    // Three buttons, on their own row. They were briefly a hover-revealed
    // select, which kept the list compact but hid the primary workflow behind
    // a pointer and a second click; reviewers asked for the buttons back.
    //
    // They are NOT on the top row. That row carries the probability, the
    // synapse count and whether the candidate has already been judged -- what
    // you read while scanning. Controls mixed in among them made both harder:
    // the numbers were interrupted by buttons, and the buttons moved sideways
    // as chips appeared and disappeared.
    const actions = document.createElement("div");
    actions.classList.add("neuroglancer-twig-capture-actions");
    // Verdicts about ONE cell of the pair get their own line under the row,
    // one group per cell: who it is first, then what is wrong with it.
    const sidesBox = document.createElement("div");
    sidesBox.classList.add("neuroglancer-twig-capture-sides");

    const merge = document.createElement("button");
    merge.classList.add("neuroglancer-twig-capture-merge");
    merge.textContent =
      status === "done"
        ? "Merged"
        : status === "already"
          ? "In neuron"
          : status === "merging"
            ? "\u2026"
            : status === "failed"
              ? "Retry"
              : "Merge";
    // Also while a staged run is going: merges must be strictly one at a
    // time, since each changes the neuron's root.
    merge.disabled =
      status === "done" ||
      status === "merging" ||
      status === "already" ||
      this.batch !== undefined;
    merge.title =
      "Writes to the chunkedgraph: adds an edge joining this candidate to the " +
      "neuron, attributed to you and visible to everyone." +
      (allowed
        ? " The decision is also recorded in twig_capture_decisions, with the " +
          "operation id."
        : "");
    merge.addEventListener("click", (event: MouseEvent) => {
      event.stopPropagation();
      void this.doMerge(candidate);
    });
    actions.appendChild(merge);

    // Merging IS the merge verdict -- doMerge records it with the operation
    // id -- so these two only cover the cases that change nothing in the
    // graph. A third button writing `merge` without merging would produce
    // rows indistinguishable from executed ones.
    if (allowed) {
      // Already joined to the neuron -- merged here, merged by someone else
      // before this list loaded, or mid-merge right now. "Not a merge" and
      // "Unsure" are then incoherent: the two segments are one object, so
      // the question the verdict answers no longer exists, and a row saying
      // no_merge about an executed merge is a false label in the GT. Merge
      // itself is already disabled in these states.
      const joined =
        status === "already" || status === "done" || status === "merging";
      for (const [verdict, label, help] of [
        ["no_merge", "Not a merge", "Record that these are different neurons."],
        ["unsure", "Unsure", "Record that this one needs a second opinion."],
      ] as [Verdict, string, string][]) {
        const b = document.createElement("button");
        b.classList.add("neuroglancer-twig-capture-merge");
        b.textContent = pending ? "\u2026" : label;
        b.disabled = pending || joined;
        b.title = joined
          ? "This candidate is already part of the neuron, so there is no " +
            "merge decision left to record. Undo the merge first if it was " +
            "wrong."
          : `${help} Writes a row to twig_capture_decisions in CAVE, ` +
            "attributed to you. It does not change the segmentation.";
        b.addEventListener("click", (event: MouseEvent) => {
          event.stopPropagation();
          void this.recordDecision(candidate, verdict);
        });
        actions.appendChild(b);
      }

      // Verdicts about ONE side of the pair. Each button names the cell it
      // marks, by role and id, because "which end of the suggested merge?"
      // is otherwise ambiguous. Neuron-first: the sink is the neuron under
      // review, the source the candidate. Orphan-first (parent_root set):
      // the sink is the seed being grown, the source the candidate.
      // Orphan-first: a full row carries parent_root; an All-neurons row is
      // a compact table row without it, so fall back to the file name
      // (run_twigcapture_v2_orph.py writes orph_<seed>_<stamp>.json).
      const orph =
        candidate.parent_root !== undefined ||
        (candidate._file ?? "").startsWith("orph_");
      const sinkRoot = ref0Seed(this.active, candidate);
      const sides: Record<"sink" | "source", [string, string]> = {
        sink: candidate.via_root
          ? ["via", candidate.via_root]
          : [orph ? "seed" : "neuron", sinkRoot],
        source: ["candidate", candidate.target_segment_id],
      };
      const sided: [Verdict, string, ("sink" | "source")[], string][] = [
        [
          "new_neuron",
          "New neuron",
          // A neuron under review is already a neuron; only an orphan side
          // can be newly found to be one.
          orph ? ["sink", "source"] : ["source"],
          "is a whole neuron of its own: these two are not a merge, and that " +
            "cell is a new neuron to add to the table (and to backbone " +
            "proofread). Counts as a no_merge for training.",
        ],
        [
          "split_needed",
          "Split needed",
          ["sink", "source"],
          "contains a false merge that should be resolved before this pair " +
            "is judged. Left out of training, and not staged by Stage all.",
        ],
      ];
      for (const side of ["sink", "source"] as const) {
        const [role, root] = sides[side];
        if (!root) continue;
        const mine = sided.filter(([, , which]) => which.includes(side));
        if (mine.length === 0) continue;
        const group = document.createElement("div");
        group.classList.add("neuroglancer-twig-capture-side");
        const who = document.createElement("span");
        who.classList.add("neuroglancer-twig-capture-side-label");
        who.textContent = `${role} \u2026${root.slice(-6)}`;
        who.title =
          `${role === "candidate" ? "The candidate" : role === "seed" ? "The seed being grown" : "The neuron under review"}: ` +
          `${root}. The buttons beside it are about this cell only.`;
        group.appendChild(who);
        for (const [verdict, label, , help] of mine) {
          const b = document.createElement("button");
          b.classList.add("neuroglancer-twig-capture-merge");
          b.textContent = pending ? "\u2026" : label;
          b.disabled = pending || joined;
          b.title = joined
            ? "This candidate is already part of the neuron, so there is no " +
              "pair left to judge. Undo the merge first if it was wrong."
            : `Record that the ${role} (${root}) ${help} Writes a ${verdict} ` +
              `row to twig_capture_decisions with cell=${root};side=${side}. ` +
              "It does not change the segmentation.";
          b.addEventListener("click", (event: MouseEvent) => {
            event.stopPropagation();
            void this.recordDecision(candidate, verdict, undefined, {
              side,
              root,
            });
          });
          group.appendChild(b);
        }
        sidesBox.appendChild(group);
      }
    }

    // Undo / Redo, only where this session actually holds an operation id.
    const entry = this.history.get(key);
    if (entry !== undefined && status !== "merging") {
      const undo = document.createElement("button");
      undo.disabled = this.batch !== undefined;
      undo.classList.add("neuroglancer-twig-capture-merge");
      undo.textContent = entry.undone ? "Redo" : "Undo";
      undo.title = entry.undone
        ? `Re-apply operation ${entry.operationId} (a new graph edit).`
        : `Undo operation ${entry.operationId} (itself a new graph edit).`;
      undo.addEventListener("click", (event: MouseEvent) => {
        event.stopPropagation();
        void this.undoOrRedo(candidate);
      });
      actions.appendChild(undo);
    }

    // All-neurons view: open this candidate's neuron in the one-neuron view.
    if (this.active === undefined && candidate._root) {
      const open = document.createElement("button");
      open.classList.add("neuroglancer-twig-capture-merge");
      open.textContent = "Neuron";
      open.title =
        `Open neuron ${candidate._root} on its own, with this candidate ` +
        'selected. "All neurons" brings this list back.';
      open.addEventListener("click", (event: MouseEvent) => {
        event.stopPropagation();
        this.layer.twigCapture.candidate = candidate.target_segment_id;
        this.scrolledToShared = false; // scroll to it once the neuron opens
        this.rootInput.value = candidate._root ?? "";
        void this.openFromCatalog(candidate._root ?? "");
      });
      actions.appendChild(open);
    }

    if (status === "failed" || status === "skipped") {
      row.classList.add("neuroglancer-twig-capture-row-failed");
    } else if (status === "done" || status === "already") {
      row.classList.add("neuroglancer-twig-capture-row-done");
    }

    // The metadata and the buttons SHARE a line: text left, controls pinned
    // right. They were briefly separate rows, which cost a whole extra line
    // of height on every one of several hundred candidates for a row that
    // was already half empty.
    const metaLine = document.createElement("div");
    metaLine.classList.add("neuroglancer-twig-capture-metaline");
    const meta = document.createElement("div");
    meta.classList.add("neuroglancer-twig-capture-meta");
    const parts: string[] = [`${candidate.num_l2_nodes} L2`];
    if (candidate.cand_size_um3 != null)
      parts.push(`${fmt(candidate.cand_size_um3, 4)} µm³`);
    if (candidate.segment_max_dt_nm != null)
      parts.push(`${candidate.segment_max_dt_nm} nm thick`);
    if (candidate.contact_voxels != null)
      parts.push(`contact ${candidate.contact_voxels}`);
    if (candidate.enclosure != null)
      parts.push(`enc ${fmt(candidate.enclosure, 2)}`);
    if (candidate.cand_em_mean != null)
      parts.push(`EM ${fmt(candidate.cand_em_mean, 0)}`);
    meta.textContent = parts.join("  ·  ");
    metaLine.appendChild(meta);
    metaLine.appendChild(actions);
    row.appendChild(metaLine);
    if (sidesBox.childElementCount > 0) row.appendChild(sidesBox);

    const syn = this.makeSynapseLine(candidate);
    if (syn !== undefined) row.appendChild(syn);

    row.addEventListener("click", () => this.goTo(candidate));
    row.title = "Click to jump to the contact site and select both segments";
    return row;
  }

  /**
   * The synapse line: what merging this candidate would recover.
   *
   * Two numbers, and deliberately not three. The LOCAL window is no longer
   * shown: it is ~1 um^3, reads zero for the great majority of candidates,
   * and a column of zeroes beside a meaningful number invited the reading
   * that the candidate has no synapses at all. It stays in the JSON for
   * analysis.
   *
   *   orphan total  every synapse on the candidate's whole extent. What a
   *                 merge actually converts, and what the sort ranks on.
   *   wide          the same restricted to a box around the contact, which
   *                 is the only thing that can say what is NEAR the join --
   *                 and so the only thing `between` can be derived from.
   *
   * Absent entirely when nothing was counted: an older file, or a run with
   * no synapse index. The row then looks as it did before rather than
   * claiming zero.
   */
  private makeSynapseLine(candidate: TwigCandidate): HTMLElement | undefined {
    const wide = candidate.synapses_wide ?? candidate.synapses_context;
    const total = candidate.cand_synapses_total;
    if (wide == null && total == null) return undefined;

    const el = document.createElement("div");
    el.classList.add("neuroglancer-twig-capture-meta");
    el.classList.add("neuroglancer-twig-capture-syn");
    const bits: string[] = [];
    if (total != null) bits.push(`orphan total ${total}`);
    if (wide != null)
      bits.push(
        `wide T ${wide.target_pre}/${wide.target_post}` +
          ` O ${wide.cand_pre}/${wide.cand_post}`,
      );
    el.textContent = "syn  " + bits.join("  \u00b7  ");

    // The box in um, read from the run rather than hardcoded: it is a config
    // value (filtering.synapse_count_box_nm) and is expected to widen.
    const box = this.active?.synapse_box_wide_nm;
    const boxText = box
      ? `${(box[0] / 1000).toFixed(1)} \u00d7 ${(box[1] / 1000).toFixed(1)} ` +
        `\u00d7 ${(box[2] / 1000).toFixed(1)} \u00b5m`
      : "a fixed box";

    const lines: string[] = [
      "ORPHAN SYNAPSES \u2014 what attaching this candidate would recover.",
      "",
    ];
    if (total != null) {
      lines.push(
        `orphan total ${total}: every synapse on the candidate, over its ` +
          "whole extent, counted by supervoxel. This is the number a merge " +
          "converts from one-sided to fully identified, and what the " +
          '"orphan synapses (total)" sort uses.',
      );
    }
    if (wide != null) {
      lines.push(
        `wide: the same counting restricted to ${boxText} centred on the ` +
          "contact site. T is this neuron, O is the candidate, each as " +
          `pre/post. ${wide.total} synapses from all neurons lie in that ` +
          "box, so the attributed numbers do not sum to it.",
      );
      if (wide.between > 0) {
        lines.push(
          `between ${wide.between}: synapses running FROM one of this pair ` +
            "TO the other. A cell rarely synapses onto itself, so this is " +
            "evidence they are different neurons.",
        );
      }
    }
    if (total != null && wide != null) {
      lines.push(
        "The two differ because the box sees only part of a fragment: " +
          "measured over 80 fragments it captured a mean of 76% of their " +
          "synapses. Rank on the total; read wide for context.",
      );
    }
    el.title = lines.join("\n");
    return el;
  }
}
