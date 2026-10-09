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
} from "#src/layer/segmentation/twig_capture_decisions.js";
import { StatusMessage } from "#src/status.js";
import { Uint64Set } from "#src/uint64_set.js";
import { NullarySignal } from "#src/util/signal.js";
import { removeChildren } from "#src/util/dom.js";
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
 *   minProb, mito, review, sortKey
 *                four scalars that decide WHICH rows are shown and in what
 *                order. Share "the top one by orphan synapses" without the
 *                sort and the row you meant is forty down. mito and review
 *                are facets ("any" / ...); mito replaced an older pair of
 *                hideMito/mitoOnly booleans, which restoreState still reads
 *                so that links made before 2026-10-06 keep working.
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
  /** "any" | "hide" | "only" -- replaced the hideMito/mitoOnly pair. */
  mito = "any";
  /** "any" | "none" | "done" | "merge" | "no_merge" | "unsure". */
  review = "any";
  sortKey = "prob";

  toJSON() {
    // Only the fields that differ from a fresh tab, so an untouched layer
    // adds nothing to the state. Neuroglancer links get pasted into chat
    // windows and issue trackers; every key costs.
    const x: { [k: string]: any } = {};
    if (this.catalogUrl) x.catalogUrl = this.catalogUrl;
    if (this.file) x.file = this.file;
    if (this.candidate) x.candidate = this.candidate;
    if (this.minProb !== 0.75) x.minProb = this.minProb;
    if (this.mito !== "any") x.mito = this.mito;
    if (this.review !== "any") x.review = this.review;
    if (this.sortKey !== "prob") x.sortKey = this.sortKey;
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
    ].includes(o.review)
      ? o.review
      : "any";
    if (["prob", "size", "contact", "synapses"].includes(o.sortKey))
      this.sortKey = o.sortKey;
    this.changed.dispatch();
  }
}

/** One candidate as run_twigcapture.py emits it. */
interface TwigCandidate {
  id: string;
  target_segment_id: string;
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
}

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
}

function parseResult(name: string, text: string): TwigResult {
  // run_twigcapture writes NaN for the unused branch, which JSON.parse rejects.
  // Only replace bare NaN tokens in value position, never inside a string.
  const cleaned = text.replace(/:\s*NaN\s*([,}])/g, ": null$1");
  const obj = JSON.parse(cleaned);
  if (!Array.isArray(obj.candidates)) {
    throw new Error(`${name}: no "candidates" array -- not an AMP result file`);
  }
  return {
    name,
    root_id: String(obj.root_id ?? "unknown"),
    timestamp: obj.timestamp,
    dataset: obj.dataset,
    models_used: obj.models_used,
    merge_threshold: obj.merge_threshold,
    connectivity: obj.connectivity,
    candidate_count: obj.candidate_count,
    candidates: obj.candidates as TwigCandidate[],
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
 */
const TWIG_CAPTURE_VERSION = "v1.0";
const TWIG_CAPTURE_DATE = "20260930";

const DEFAULT_CATALOG_URL =
  "https://catmaid2.hms.harvard.edu/twigcapture_aedes/";

function readStoredCatalogUrl(): string {
  try {
    return localStorage.getItem(CATALOG_URL_KEY) ?? DEFAULT_CATALOG_URL;
  } catch {
    return DEFAULT_CATALOG_URL;
  }
}

function storeCatalogUrl(url: string) {
  try {
    if (url) localStorage.setItem(CATALOG_URL_KEY, url);
    else localStorage.removeItem(CATALOG_URL_KEY);
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
  private reviewSelect: HTMLSelectElement | undefined;
  private mergeAllButton = document.createElement("button");
  /** candidate id -> outcome, so a row is not offered twice. */
  private mergeStatus = new Map<
    string,
    "merging" | "done" | "failed" | "already"
  >();
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
  /** Candidates whose decision POST is in flight, so the row can show it. */
  private deciding = new Set<string>();
  /** The target neuron's root TODAY; changes with every merge in this tab. */
  private activeTargetRoot: bigint | undefined;
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
    t.mito = this.mito;
    t.review = this.review;
    t.sortKey = this.sortKey;
    t.changed.dispatch();
  }

  constructor(public layer: SegmentationUserLayer) {
    super();
    // Seed from the layer BEFORE the controls are built, so they render with
    // the shared link's values rather than snapping to them afterwards.
    const saved = layer.twigCapture;
    this.minProb = saved.minProb;
    this.mito = saved.mito;
    this.review = saved.review;
    this.sortKey = saved.sortKey as typeof this.sortKey;
    const { element } = this;
    element.classList.add("neuroglancer-twig-capture-tab");
    element.appendChild(this.makeCatalogRow());
    element.appendChild(this.makeResultSelectRow());
    element.appendChild(this.makeFilterRow());
    this.summary.classList.add("neuroglancer-twig-capture-summary");
    element.appendChild(this.summary);
    const countRow = document.createElement("div");
    countRow.classList.add("neuroglancer-twig-capture-countrow");
    this.countEl.classList.add("neuroglancer-twig-capture-count");
    countRow.appendChild(this.countEl);
    // Disabled until merging exists. It stays visible so the review workflow
    // reads the same as it will once merging is wired up -- and because a
    // bulk graph edit is the one action that most needs to be deliberate.
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
    this.mergeAllButton.classList.add("neuroglancer-twig-capture-merge");
    this.mergeAllButton.textContent = "Merge all";
    this.mergeAllButton.disabled = true;
    this.mergeAllButton.title =
      "Not yet implemented -- this would edit the chunkedgraph for every " +
      "candidate currently listed.";
    countRow.appendChild(this.mergeAllButton);
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
      this.layer.twigCapture.catalogUrl || readStoredCatalogUrl();
    const connect = () => {
      const value = urlInput.value.trim();
      this.catalogUrl = value;
      this.publish();
      storeCatalogUrl(value);
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

    this.catalogStatus.classList.add("neuroglancer-twig-capture-status");

    row.appendChild(urlInput);
    row.appendChild(button);
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
    this.rootInput.addEventListener("change", open);
    this.rootInput.addEventListener("keydown", (event: KeyboardEvent) => {
      if (event.key === "Enter") open();
    });
    search.appendChild(this.rootInput);
    search.appendChild(this.rootList);
    row.appendChild(search);

    if (this.catalogUrl) void this.loadCatalog(this.catalogUrl);
    return row;
  }

  private async loadCatalog(url: string) {
    const base = url.endsWith("/") ? url : url + "/";
    this.catalogStatus.textContent = "loading\u2026";
    try {
      let index: any;
      try {
        const response = await fetch(base + "index.json");
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
      this.catalogStatus.textContent = `${this.catalog.length} file(s)`;
      this.catalogStatus.title = index?.generated
        ? `index generated ${index.generated}`
        : "";
      // A shared link names a FILE. Open it once, here, rather than when the
      // tab was constructed: the catalog has to be in hand first so the file
      // can be resolved to a URL, and a tab opened before the fetch returned
      // would show an empty list under a link that promised a neuron.
      const want = this.layer.twigCapture.file;
      if (want && !this.results.some((r) => r.name === want)) {
        void this.loadUrl(base + want);
      }
      removeChildren(this.rootList);
      for (const entry of this.catalog) {
        const option = document.createElement("option");
        option.value = entry.root_id;
        const bits = [`${entry.candidate_count ?? "?"} cands`];
        if (entry.n_review !== undefined)
          bits.push(`${entry.n_review} to review`);
        if (entry.timestamp) bits.push(entry.timestamp.slice(0, 10));
        option.label = bits.join(", ");
        this.rootList.appendChild(option);
      }
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
    const base = this.catalogUrl.endsWith("/")
      ? this.catalogUrl
      : this.catalogUrl + "/";
    await this.loadUrl(base + entry.file);
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

    addNumber(
      "p ≥",
      this.minProb,
      0.05,
      0,
      1,
      (v) => {
        this.minProb = v;
        this.publish();
      },
      "Minimum merge probability",
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
    // A different neuron means a fresh merge history and a fresh target root.
    this.mergeStatus.clear();
    this.history.clear();
    this.activeTargetRoot = undefined;
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

    // Show both sides of the proposed merge: the neuron under review and the
    // candidate.
    let ids: bigint[];
    try {
      // activeTargetRoot tracks merges made in this session; fall back to the
      // root recorded in the file.
      ids = [
        this.activeTargetRoot ?? BigInt(this.active!.root_id),
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
  private async showSegments(ids: bigint[]) {
    const group = this.layer.displayState.segmentationGroupState.value;
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
    const entry = this.history.get(candidate.id);
    if (server === undefined || entry === undefined) return;
    const redoing = entry.undone;
    this.mergeStatus.set(candidate.id, "merging");
    this.render();
    try {
      const { operationId } = await server.undoOperation(entry.operationId);
      // Act on the operation that now stands: after an undo, that is the undo
      // itself; undoing it again is the redo.
      this.history.set(candidate.id, {
        operationId: operationId ?? entry.operationId,
        undone: !entry.undone,
      });
      if (redoing) {
        this.mergeStatus.set(candidate.id, "done");
      } else {
        // Undone: the candidate stands apart again, so offer Merge afresh.
        this.mergeStatus.delete(candidate.id);
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
        this.history.delete(candidate.id);
      }
      // Every root involved has changed; re-resolve rather than guess.
      await this.refreshAgainstGraph();
    } catch (e) {
      this.mergeStatus.set(candidate.id, "failed");
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
    if (result === undefined) return;
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
      this.activeTargetRoot = targetRoot;
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
            if (root === targetRoot) {
              already++;
              // Do not overwrite a merge this session already recorded.
              if (this.mergeStatus.get(candidate.id) !== "done") {
                this.mergeStatus.set(candidate.id, "already");
              }
            } else if (this.mergeStatus.get(candidate.id) === "already") {
              this.mergeStatus.delete(candidate.id);
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

  /** The decision store for the active result, or undefined with a reason. */
  private store(): DecisionStore | undefined {
    const datastack = this.active?.dataset;
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
   * SUPERVOXEL, which is immutable. activeTargetRoot is preferred when set
   * because it reflects merges made in this session without another round
   * trip. Falls back to the file's root only when there is no graph to ask,
   * which is also the only case where it can still be stale.
   */
  private async currentNeuronRoot(result: TwigResult): Promise<string> {
    if (this.activeTargetRoot !== undefined)
      return String(this.activeTargetRoot);
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
    const store = this.store();
    if (store === undefined) return;
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
    const result = this.active;
    const store = this.store();
    if (result === undefined || store === undefined) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: no CAVE datastack for this layer; decision not saved.",
        5000,
      );
      return false;
    }
    const sink = candidate.merge_sink;
    const source = candidate.merge_source;
    if (!sink || !source) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: this result file has no merge_sink / merge_source, so " +
          "the two sides cannot be bound. Re-run run_twigcapture.py.",
        6000,
      );
      return false;
    }
    // The result file stem, which is what src= has to point at.
    const src = result.name.replace(/\.json$/, "");
    const target: DecisionTarget = {
      sink,
      source,
      coordinate_nm: candidate.coordinate_nm,
      cand: candidate.target_segment_id,
      src,
    };
    this.deciding.add(candidate.target_segment_id);
    this.render();
    try {
      const id = await store.post(target, verdict, op);
      this.decisions.set(candidate.target_segment_id, {
        id,
        tag: verdict,
        cand: candidate.target_segment_id,
        tag2: `src=${src};cand=${candidate.target_segment_id}`,
        created: Date.now(),
      });
      return true;
    } catch (e) {
      StatusMessage.showTemporaryMessage(
        `Twig Capture: decision NOT saved: ${e}`,
        7000,
      );
      return false;
    } finally {
      this.deciding.delete(candidate.target_segment_id);
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
  private async doMerge(candidate: TwigCandidate): Promise<boolean> {
    const graph = this.graphene();
    const server = graph?.graphServer;
    if (server === undefined) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: this layer has no graphene graph to merge on.",
        4000,
      );
      return false;
    }
    const sink = candidate.merge_sink;
    const source = candidate.merge_source;
    if (!sink || !source) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: this result file predates the merge endpoints " +
          "(merge_sink / merge_source). Re-run run_twigcapture.py to add them.",
        6000,
      );
      return false;
    }

    this.mergeStatus.set(candidate.id, "merging");
    this.render();
    try {
      const sinkSv = BigInt(sink.supervoxel_id);
      const sourceSv = BigInt(source.supervoxel_id);
      // Today's roots for both sides.
      const [sinkRoot, sourceRoot] = await Promise.all([
        graph!.getRoot ? graph!.getRoot(sinkSv) : Promise.resolve(sinkSv),
        graph!.getRoot ? graph!.getRoot(sourceSv) : Promise.resolve(sourceSv),
      ]);
      if (sinkRoot === sourceRoot) {
        this.mergeStatus.set(candidate.id, "done");
        StatusMessage.showTemporaryMessage(
          "Already merged: both sides are the same segment today.",
          4000,
        );
        this.render();
        return true;
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
        const ok = window.confirm(
          `Candidate ${candidate.target_segment_id} has been edited since ` +
            `this file was written -- it is now part of ${sourceRoot}.\n\n` +
            "Merging joins THAT segment to this neuron, which may be much " +
            "larger than the twig scored here " +
            `(${candidate.num_l2_nodes} L2 node(s)).\n\n` +
            "Check it in the viewer first. Merge anyway?",
        );
        if (!ok) {
          this.mergeStatus.delete(candidate.id);
          this.render();
          return false;
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
      graph!.state?.replaceSegments(oldValues, newValues);
      const group = this.layer.displayState.segmentationGroupState.value;
      group.selectedSegments.delete(sinkRoot);
      group.selectedSegments.delete(sourceRoot);
      group.selectedSegments.add(newRoot);
      group.visibleSegments.set(newRoot, true);

      // The neuron under review now has a new root; later merges in this
      // session must target it.
      console.log("[twig-capture] merge ok", {
        candidate: candidate.id,
        newRoot: String(newRoot),
        operationId,
      });
      this.activeTargetRoot = newRoot;
      this.mergeStatus.set(candidate.id, "done");
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
        this.history.set(candidate.id, { operationId, undone: false });
      }
      StatusMessage.showTemporaryMessage(
        `Merged ${candidate.target_segment_id} \u2192 new root ${newRoot}`,
        5000,
      );
      this.render();
      return true;
    } catch (e) {
      this.mergeStatus.set(candidate.id, "failed");
      // Full object, not just the message: graphene wraps HttpError and the
      // server's reason is often only in the stack or the response body.
      console.error("[twig-capture] merge failed", candidate.id, e);
      StatusMessage.showTemporaryMessage(
        `Twig Capture: merge failed for ${candidate.target_segment_id}: ${e}`,
        6000,
      );
      this.render();
      return false;
    }
  }

  private filtered(): TwigCandidate[] {
    const result = this.active;
    if (result === undefined) return [];
    const rows = result.candidates.filter((c) => {
      if (c.prob_merge < this.minProb) return false;
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
        const d = this.decisions.get(c.target_segment_id);
        if (this.review === "none" && d !== undefined) return false;
        if (this.review === "done" && d === undefined) return false;
        if (
          ["merge", "no_merge", "unsure"].includes(this.review) &&
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
    const sameFile =
      this.active !== undefined && this.lastRenderedFile === this.active.name;
    const keptScrollTop = this.listEl.scrollTop;
    const restoreScroll = () => {
      // The browser clamps to the new scrollHeight, so a list that shrank
      // under a filter change lands at the bottom rather than out of bounds.
      if (sameFile) this.listEl.scrollTop = keptScrollTop;
      this.lastRenderedFile = this.active?.name;
    };
    // file chooser
    removeChildren(this.fileSelect);
    for (const result of this.results) {
      const option = document.createElement("option");
      option.textContent = `${result.root_id}  (${result.candidates.length})`;
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
      this.countEl.textContent = "";
      restoreScroll();
      return;
    }

    const bits = [`neuron ${result.root_id}`];
    if (result.models_used) bits.push(`model ${result.models_used}`);
    if (result.connectivity !== undefined)
      bits.push(`${result.connectivity === 3 ? 26 : 6}-conn`);
    if (result.merge_threshold !== undefined)
      bits.push(`thr ${result.merge_threshold}`);
    if (result.timestamp) bits.push(result.timestamp.slice(0, 16));
    this.summary.textContent = bits.join("  ·  ");

    const rows = this.filtered();
    this.countEl.textContent = `${rows.length} of ${result.candidates.length} candidates`;
    this.mergeAllButton.textContent = `Merge all (${rows.length})`;

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

    const prob = document.createElement("span");
    prob.classList.add("neuroglancer-twig-capture-prob");
    prob.textContent = fmt(candidate.prob_merge);
    main.appendChild(prob);

    const id = document.createElement("span");
    id.classList.add("neuroglancer-twig-capture-id");
    id.textContent = candidate.target_segment_id;
    main.appendChild(id);

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

    const status = this.mergeStatus.get(candidate.id);
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
    merge.disabled =
      status === "done" || status === "merging" || status === "already";
    merge.title =
      "Writes to the chunkedgraph: adds an edge joining this candidate to the " +
      "neuron, attributed to you and visible to everyone.";
    merge.addEventListener("click", (event: MouseEvent) => {
      event.stopPropagation();
      void this.doMerge(candidate);
    });
    main.appendChild(merge);

    // Undo / Redo, only where this session actually holds an operation id.
    const entry = this.history.get(candidate.id);
    if (entry !== undefined && status !== "merging") {
      const undo = document.createElement("button");
      undo.classList.add("neuroglancer-twig-capture-merge");
      undo.textContent = entry.undone ? "Redo" : "Undo";
      undo.title = entry.undone
        ? `Re-apply operation ${entry.operationId} (a new graph edit).`
        : `Undo operation ${entry.operationId} (itself a new graph edit).`;
      undo.addEventListener("click", (event: MouseEvent) => {
        event.stopPropagation();
        void this.undoOrRedo(candidate);
      });
      main.appendChild(undo);
    }

    // ---- the recorded verdict, and the two buttons that set it -----------
    //
    // There is no "merge" button here on purpose. Merging IS the merge
    // verdict: doMerge records it with the operation id, so a row claiming
    // `merge` is always backed by a real graph edit. A separate button that
    // said "merge" without merging would produce rows nobody could tell
    // apart from executed ones.
    const decided = this.decisions.get(candidate.target_segment_id);
    const pending = this.deciding.has(candidate.target_segment_id);
    if (decided !== undefined) {
      const chip = document.createElement("span");
      chip.classList.add("neuroglancer-twig-capture-tag");
      if (decided.tag === "no_merge") {
        chip.classList.add("neuroglancer-twig-capture-tag-warn");
      }
      chip.textContent = decided.tag;
      chip.title =
        `Recorded in CAVE as "${decided.tag}" (annotation ${decided.id}). ` +
        "Reviewing again adds a new row; the latest wins.";
      main.appendChild(chip);
    }
    // The non-merge verdicts live in a select, not in buttons.
    //
    // Merge stays a button because it is the primary action and it writes to
    // the shared graph -- a destructive edit should not be one mis-click
    // inside a list. The other two are rarer and were costing every one of
    // several hundred rows two permanent controls, which drowned the numbers
    // a reviewer is actually scanning. The select is revealed on hover (see
    // twig_capture.css); the chip above is what shows a recorded verdict at
    // rest, so nothing is hidden, only the means of changing it.
    const allowed = this.decisionsAllowed();
    const verdictWrap = document.createElement("span");
    verdictWrap.classList.add("neuroglancer-twig-capture-verdict");
    const select = document.createElement("select");
    select.classList.add("neuroglancer-twig-capture-merge");
    select.disabled = pending;
    select.title =
      "Record a verdict in twig_capture_decisions (CAVE), attributed to you. " +
      "This does not change the segmentation.";
    for (const [value, label] of [
      ["", pending ? "\u2026" : "Mark\u2026"],
      ["no_merge", "Not a merge"],
      ["unsure", "Unsure"],
    ]) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      if (value === "") {
        // The placeholder is a LABEL for the closed select, not a choice.
        // `disabled` stops it being picked, `hidden` keeps it out of the open
        // list; a hidden option that is selected still shows its text while
        // the select is closed, which is the whole trick. Without both it sat
        // in the menu as a third, meaningless item.
        opt.disabled = true;
        opt.hidden = true;
      }
      select.appendChild(opt);
    }
    // selectedIndex rather than value="": `disabled` does not block a
    // programmatic selection, but going through the index says plainly that
    // we mean "back to the placeholder" and does not depend on that nuance.
    select.selectedIndex = 0;
    select.addEventListener("click", (event: MouseEvent) => {
      // The row's own click handler jumps the viewer; opening the select
      // should not also move the camera.
      event.stopPropagation();
    });
    select.addEventListener("change", (event: Event) => {
      event.stopPropagation();
      const chosen = select.value as Verdict | "";
      // Back to the placeholder straight away: the select is a verb, not a
      // state display. What was recorded is shown by the chip, which is the
      // only thing that reflects what is actually in CAVE.
      select.selectedIndex = 0;
      if (chosen === "") return;
      void this.recordDecision(candidate, chosen);
    });
    verdictWrap.appendChild(select);
    if (allowed) main.appendChild(verdictWrap);

    if (status === "failed") {
      row.classList.add("neuroglancer-twig-capture-row-failed");
    } else if (status === "done" || status === "already") {
      row.classList.add("neuroglancer-twig-capture-row-done");
    }
    row.appendChild(main);

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
    row.appendChild(meta);

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
