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
 * Twig Selector: pick segments, run Twig Capture's merge model on them in
 * this browser, and hand the results to the Twig Capture tab.
 *
 * Everything runs locally: data is read straight from the dataset's public
 * volumes and the chunkedgraph (with the layer's own CAVE credentials), the
 * model runs on the GPU (WebGPU) or CPU (WASM), and results and the model are
 * cached in IndexedDB. Verdicts on these results go to the same CAVE
 * decisions table as any other Twig Capture result (src=browser_<root>_...).
 *
 * The queue and the running job live on TwigSelectorState, owned by the
 * layer, so a run continues while the side panel is closed.
 */

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import { DecisionStore } from "#src/layer/segmentation/twig_capture_decisions.js";
import type { TwigDataset } from "#src/layer/segmentation/twig_inference/datasets.js";
import {
  DEFAULT_MODEL_URL,
  LARGE_MERGE_L2,
  NO_L2_LIMIT,
  TWIG_PARAMS,
  datasetForUrls,
} from "#src/layer/segmentation/twig_inference/datasets.js";
import {
  buildReport,
  copyText,
  errorText,
  logEvent,
} from "#src/layer/segmentation/twig_inference/diagnostics.js";
import type { Backend } from "#src/layer/segmentation/twig_inference/model.js";
import {
  F16_STORAGE,
  MergeModel,
  putLocalModel,
  webgpuAvailable,
} from "#src/layer/segmentation/twig_inference/model.js";
import type {
  GraphAccess,
  Progress,
} from "#src/layer/segmentation/twig_inference/pipeline.js";
import {
  DatasetContext,
  getLeavesL2,
  neuronSize,
  runTwigCapture,
} from "#src/layer/segmentation/twig_inference/pipeline.js";
import { addSynapseCounts } from "#src/layer/segmentation/twig_inference/synapses.js";
import { StatusMessage } from "#src/status.js";
import { removeChildren } from "#src/util/dom.js";
import { NullarySignal } from "#src/util/signal.js";
import { Tab } from "#src/widget/tab_view.js";

const SETTINGS_KEY = "neuroglancer-twig-selector-settings";

export interface SelectorSettings {
  /**
   * Choose max regions, candidate size and sites/pair per neuron from its
   * size and the device (autoPlan); the fields below are then ignored.
   */
  auto: boolean;
  level: number;
  chainThreshold: number;
  maxRegions: number;
  maxSites: number;
  /** Largest candidate proposed, in L2 nodes (default no limit; cluster runner: 20). */
  maxCandidateL2: number;
  /** 2: saved after partners were uncapped (see loadSettings). */
  l2LimitVersion?: number;
  /** contact_gap_voxels: 0 (touching only), 1 (default) or 2. */
  contactGap: number;
  /** The WebGPU network's storage precision: "f32" (default) or "f16". */
  gpuPrecision: "f32" | "f16";

  backend: Backend;
  modelUrl: string;
}

const DEFAULT_SETTINGS: SelectorSettings = {
  auto: true,
  level: 1,
  chainThreshold: 0.75,
  maxRegions: 200,
  maxSites: 8,
  maxCandidateL2: NO_L2_LIMIT,
  l2LimitVersion: 2,
  contactGap: TWIG_PARAMS.contactGap,
  gpuPrecision: "f32",

  backend: "auto",
  modelUrl: DEFAULT_MODEL_URL,
};

function loadSettings(): SelectorSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      // Settings saved before partners were uncapped carry the old default
      // (20); lift it once, or those users would never see large partners.
      if (saved.l2LimitVersion !== 2) {
        saved.maxCandidateL2 = NO_L2_LIMIT;
        saved.l2LimitVersion = 2;
      }
      return { ...DEFAULT_SETTINGS, ...saved };
    }
  } catch {
    // private window / blocked storage
  }
  return { ...DEFAULT_SETTINGS };
}

type ItemStatus = "queued" | "running" | "done" | "failed" | "cancelled";

interface QueueItem {
  root: string;
  status: ItemStatus;
  progress?: Progress;
  error?: string;
  resultName?: string;
  /** Message and stack of a failure, for the copied report. */
  errorDetail?: string;
  /** The neuron's size, measured when queued (see neuronSize). */
  size?: { l2: number; regions: number };
  /** Set when the root was replaced by its latest version before running. */
  updatedFrom?: string;
  /** Share of the old root's L2 nodes now in the new root (below 1: split). */
  updatedShare?: number;
}

/** The options one neuron runs with, and what that costs. */
export interface RunPlan {
  maxRegions: number;
  maxCandidateL2: number;
  maxSites: number;
  /** Rough wall time in minutes on this device. */
  minutes: number;
}

/** Region budget per level when chosen automatically. */
const AUTO_REGIONS_GPU = 300;
const AUTO_REGIONS_CPU = 3;
/** Measured on an Apple M-series laptop (BRIEF.md): seconds per region. */
const SEC_PER_REGION_GPU = 25;
const SEC_PER_REGION_CPU = 600;

/**
 * Options from the neuron's size. Partners are never size-limited: those
 * above LARGE_MERGE_L2 are tagged "large merge" in Twig Capture instead,
 * because the model was trained on smaller ones. Every region is
 * searched up to a budget, above which regions are spread over the neuron.
 * On CPU the budget and sites/pair are cut hard: ~10 min per region.
 */
export function autoPlan(
  size: { l2: number; regions: number },
  gpu: boolean,
): RunPlan {
  const maxRegions = Math.max(
    1,
    Math.min(size.regions, gpu ? AUTO_REGIONS_GPU : AUTO_REGIONS_CPU),
  );
  return {
    maxRegions,
    maxCandidateL2: NO_L2_LIMIT,
    maxSites: gpu ? 8 : 1,
    minutes:
      (maxRegions * (gpu ? SEC_PER_REGION_GPU : SEC_PER_REGION_CPU)) / 60,
  };
}

const contexts = new Map<string, DatasetContext>();

export class TwigSelectorState {
  items: QueueItem[] = [];
  settings = loadSettings();
  changed = new NullarySignal();
  private running = false;
  private abort: AbortController | undefined;
  private model: { key: string; model: MergeModel } | undefined;

  constructor(private layer: SegmentationUserLayer) {}

  private gpuProbe: Promise<boolean> | undefined;
  /** Will the model run on the GPU with the current device setting? */
  willUseGpu(): Promise<boolean> {
    const b = this.settings.backend;
    if (b === "wasm") return Promise.resolve(false);
    if (b === "webgpu" || b === "webgpu-ort") return Promise.resolve(true);
    return (this.gpuProbe ??= webgpuAvailable());
  }
  /** Last known answer of willUseGpu, for drawing the plan synchronously. */
  gpuKnown: boolean | undefined;

  /** The options this neuron will run with (auto from its size, or the fields). */
  planFor(item: QueueItem): RunPlan | undefined {
    const st = this.settings;
    const gpu = this.gpuKnown ?? true;
    if (st.auto) return item.size ? autoPlan(item.size, gpu) : undefined;
    const regions = Math.min(
      item.size?.regions ?? st.maxRegions,
      st.maxRegions,
    );
    return {
      maxRegions: st.maxRegions,
      maxCandidateL2: st.maxCandidateL2 ?? NO_L2_LIMIT,
      maxSites: st.maxSites,
      minutes: (regions * (gpu ? SEC_PER_REGION_GPU : SEC_PER_REGION_CPU)) / 60,
    };
  }

  /**
   * The root's latest version. A root is a version, not an identity: any
   * edit retires it. Unchanged roots cost one is_latest_roots call. For a
   * retired one, an even sample of its L2 nodes (which still resolve) is
   * mapped to today's roots and the most common wins -- the version that
   * carries most of the neuron, also after a split.
   */
  async latestRoot(
    root: string,
  ): Promise<{ root: string; share: number } | undefined> {
    const graph = this.graph();
    if (!graph) return undefined;
    const r = await graph.fetchGraph("/is_latest_roots?int64_as_str=1", {
      method: "POST",
      body: JSON.stringify({ node_ids: [root] }),
    });
    const latest = (await r.json())?.is_latest?.[0];
    if (latest) return { root, share: 1 };
    const leaves = await getLeavesL2(graph, root);
    if (!leaves.length) return undefined;
    const n = Math.min(40, leaves.length);
    const sample = Array.from(
      { length: n },
      (_, i) => leaves[Math.floor((i * leaves.length) / n)],
    );
    const counts = new Map<string, number>();
    await Promise.all(
      sample.map(async (id) => {
        try {
          const rr = await graph.fetchGraph(`/node/${id}/root?int64_as_str=1`);
          const now = String((await rr.json()).root_id);
          counts.set(now, (counts.get(now) ?? 0) + 1);
        } catch {
          // one failed lookup only thins the sample
        }
      }),
    );
    let best: string | undefined;
    let bestN = 0;
    for (const [k, v] of counts) {
      if (v > bestN) {
        best = k;
        bestN = v;
      }
    }
    return best ? { root: best, share: bestN / n } : undefined;
  }

  /** Replace a queued root by its latest version (see latestRoot). */
  private async updateRoot(item: QueueItem) {
    const now = await this.latestRoot(item.root);
    if (!now || now.root === item.root) return;
    logEvent(
      "info",
      `root ${item.root} is outdated; using ${now.root} (${Math.round(now.share * 100)}% of its L2 nodes)`,
    );
    if (this.items.some((i) => i !== item && i.root === now.root)) {
      // Another queued root is already this neuron as it is now.
      if (item.status !== "running") {
        this.items = this.items.filter((i) => i !== item);
        this.changed.dispatch();
      }
      return;
    }
    item.updatedFrom = item.updatedFrom ?? item.root;
    item.updatedShare = now.share;
    item.root = now.root;
    item.size = undefined;
    // A result cached for the new root may already exist.
    const hit = this.cached(item.root);
    if (hit && item.status === "queued") {
      item.status = "done";
      item.resultName = hit.name;
    }
    this.changed.dispatch();
  }

  /** Measure a queued neuron (L2 nodes, regions) in the background. */
  measure(item: QueueItem) {
    const graph = this.graph();
    const ds = this.dataset();
    if (!graph || !ds || item.size) return;
    void this.willUseGpu().then((g) => {
      this.gpuKnown = g;
      this.changed.dispatch();
    });
    this.updateRoot(item)
      .catch(() => undefined)
      .then(() => neuronSize(graph, item.root, ds.supervoxelResolution))
      .then(
        (size) => {
          item.size = size;
          this.changed.dispatch();
        },
        () => {
          // Sizing is advisory: the run measures again and reports failures.
        },
      );
  }

  /** The backend the loaded model actually runs on (after any fallback). */
  get backendUsed() {
    return this.model?.model.backend;
  }

  get isRunning() {
    return this.running;
  }

  dataset(): TwigDataset | undefined {
    return datasetForUrls(this.layer.dataSources.map((d) => d.spec.url));
  }

  /** CAVE server origin of this layer's graphene source. */
  private caveServer(): string | undefined {
    for (const ds of this.layer.dataSources) {
      const m = /https?:\/\/[^/\s]+/.exec(ds.spec?.url ?? "");
      if (m !== null) return m[0];
    }
    return undefined;
  }

  graph(): GraphAccess | undefined {
    const g = this.layer.displayState.segmentationGroupState.value.graph
      .value as any;
    return g && typeof g.fetchGraph === "function"
      ? (g as GraphAccess)
      : undefined;
  }

  saveSettings() {
    // A level change can make a finished item's result too shallow: queue it
    // again (or mark it done if a deep-enough result already exists).
    for (const item of this.items) {
      if (item.status !== "done" && item.status !== "queued") continue;
      const hit = this.cached(item.root);
      item.status = hit ? "done" : "queued";
      item.resultName = hit?.name;
    }
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(this.settings));
    } catch {
      // ignore
    }
    this.changed.dispatch();
  }

  add(roots: string[]) {
    let added = 0;
    for (const root of roots) {
      if (!/^\d+$/.test(root) || root === "0") continue;
      if (this.items.some((i) => i.root === root)) continue;
      // Already computed in this browser at this level or deeper: show it as
      // done rather than queueing work Run would skip anyway.
      const hit = this.cached(root);
      const item: QueueItem = hit
        ? { root, status: "done", resultName: hit.name }
        : { root, status: "queued" };
      this.items.push(item);
      this.measure(item);
      added++;
    }
    this.changed.dispatch();
    return added;
  }

  remove(root: string) {
    if (this.items.find((i) => i.root === root)?.status === "running") return;
    this.items = this.items.filter((i) => i.root !== root);
    this.changed.dispatch();
  }

  clearFinished() {
    this.items = this.items.filter(
      (i) => i.status === "queued" || i.status === "running",
    );
    this.changed.dispatch();
  }

  /**
   * The newest result this browser holds for a root in this dataset, if it
   * went at least as deep as the level now set (a level-1 result does not
   * answer a level-2 request; a level-2 result contains level 1).
   */
  cached(root: string, plan?: RunPlan) {
    const r = this.layer.localTwigResults.forRoot(
      root,
      this.dataset()?.datastack,
    );
    if (r === undefined) return undefined;
    const x = r as any;
    const st = this.settings;
    // In auto mode before the neuron is measured there is no plan: hold the
    // result to the auto budget instead of the (ignored) manual fields.
    const want =
      plan ??
      (st.auto
        ? { maxRegions: AUTO_REGIONS_GPU, maxCandidateL2: NO_L2_LIMIT }
        : {
            maxRegions: st.maxRegions,
            maxCandidateL2: st.maxCandidateL2 ?? NO_L2_LIMIT,
          });
    // The result must have looked at least as far as the settings now ask:
    // as deep a level, as large a partner, and every region unless it was
    // capped at or above today's cap.
    if ((x.level ?? 1) < st.level) return undefined;
    // Made by another model: a new model means new predictions.
    if (x.local_model !== undefined && x.local_model !== st.modelUrl) {
      return undefined;
    }
    if ((x.filters?.max_candidate_l2_nodes ?? 20) < want.maxCandidateL2) {
      return undefined;
    }
    // Scored at another GPU precision.
    if (
      (x.model_precision ?? "f32") !== st.gpuPrecision &&
      x.benchmark?.device === "webgpu"
    )
      return undefined;
    // Searched with another contact gap: a different candidate set.
    if ((x.contact_gap_voxels ?? 0) !== st.contactGap) return undefined;
    if (
      st.contactGap > 0 &&
      (x.contact_gap_mode ?? "all") !== TWIG_PARAMS.contactGapMode
    )
      return undefined;
    const searched = x.binned_regions_searched;
    const total = x.binned_regions_total;
    if (
      typeof searched === "number" &&
      typeof total === "number" &&
      searched < total &&
      searched < want.maxRegions
    ) {
      return undefined;
    }
    return r;
  }

  cachedAnyLevel(root: string) {
    return this.layer.localTwigResults.forRoot(root, this.dataset()?.datastack);
  }

  stop() {
    this.abort?.abort();
  }

  /** Run every queued item; `force` re-runs roots that already have a result. */
  async run(force = false) {
    if (this.running) return;
    const dataset = this.dataset();
    const graph = this.graph();
    if (!dataset || !graph) {
      StatusMessage.showTemporaryMessage(
        "Twig Selector: this layer is not a BANC, Aedes or CRANTb graphene layer.",
        5000,
      );
      return;
    }
    this.running = true;
    this.abort = new AbortController();
    const signal = this.abort.signal;
    this.changed.dispatch();
    try {
      for (const item of this.items) {
        if (signal.aborted) break;
        if (item.status === "running" || item.status === "done") {
          if (!force || item.status === "running") continue;
        }
        if (item.status !== "queued" && !force) continue;
        // This neuron's options: from its size (auto), else the fields.
        let plan: RunPlan;
        try {
          // Always run on the neuron as it is NOW.
          await this.updateRoot(item);
          // Dropped as a duplicate of another queued neuron.
          if (!this.items.includes(item)) continue;
          if (this.settings.auto) {
            const ds = this.dataset()!;
            item.size ??= await neuronSize(
              graph,
              item.root,
              ds.supervoxelResolution,
            );
            const gpu = await this.willUseGpu();
            this.gpuKnown = gpu;
            plan = autoPlan(item.size, gpu);
          } else {
            plan = this.planFor(item)!;
          }
        } catch (e) {
          item.status = "failed";
          item.error = `could not measure the neuron: ${String((e as any)?.message ?? e)}`;
          item.errorDetail = errorText(e);
          logEvent("error", `root ${item.root}: sizing failed`, e);
          this.changed.dispatch();
          continue;
        }
        const cached = this.cached(item.root, plan);
        if (cached && !force) {
          item.status = "done";
          item.resultName = cached.name;
          continue;
        }
        item.status = "running";
        item.error = undefined;
        item.progress = {
          phase: "starting",
          done: 0,
          total: 0,
          patches: 0,
          found: 0,
        };
        this.changed.dispatch();
        try {
          const st = this.settings;
          const key = `${st.modelUrl}|${st.backend}|${st.gpuPrecision}`;
          if (this.model?.key !== key) {
            await this.model?.model.release();
            this.model = undefined;
            const model = await MergeModel.create(
              st.modelUrl,
              st.backend,
              (msg) => {
                item.progress = { ...item.progress!, phase: msg };
                this.changed.dispatch();
              },
              undefined,
              st.gpuPrecision === "f16" ? { precision: F16_STORAGE } : {},
            );
            this.model = { key, model };
          }
          let ctx = contexts.get(dataset.key);
          if (!ctx) {
            ctx = new DatasetContext(dataset);
            contexts.set(dataset.key, ctx);
          }
          const result = await runTwigCapture(
            item.root,
            graph,
            ctx,
            this.model.model,
            {
              level: st.level,
              chainThreshold: st.chainThreshold,
              maxRegions: plan.maxRegions,
              maxSites: plan.maxSites,
              maxCandidateL2: plan.maxCandidateL2,
              contactGap: st.contactGap,

              modelUrl: st.modelUrl,
            },
            (p) => {
              item.progress = p;
              this.changed.dispatch();
            },
            signal,
          );
          // Synapses on each partner, for the "N syn" tag and the synapse
          // sort. Best effort: a result without them is still a result.
          const server = this.caveServer();
          if (server !== undefined && result.candidates.length > 0) {
            item.progress = { ...item.progress!, phase: "counting synapses" };
            this.changed.dispatch();
            try {
              await addSynapseCounts(
                result,
                new DecisionStore(server, dataset.datastack),
                dataset.synapseTable,
                signal,
              );
            } catch (e) {
              if ((e as any)?.name === "AbortError") throw e;
              logEvent("warn", `root ${item.root}: synapse counts failed`, e);
            }
          }
          item.status = "done";
          item.resultName = result.name;
          this.layer.localTwigResults.add(result, false);
        } catch (e) {
          if ((e as any)?.name === "AbortError") {
            item.status = "cancelled";
          } else {
            item.status = "failed";
            item.error = String((e as any)?.message ?? e);
            item.errorDetail = errorText(e);
            logEvent("error", `root ${item.root} failed`, e);
          }
        }
        this.changed.dispatch();
      }
    } finally {
      this.running = false;
      this.abort = undefined;
      this.changed.dispatch();
    }
  }

  /**
   * Alt+T: queue these roots and run. A run already going picks them up (it
   * walks the live queue); roots with a result at this level are shown done.
   */
  async predictFromHotkey(roots: string[]) {
    const ds = this.dataset();
    if (!ds) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: Alt+T works on a BANC, Aedes or CRANTb graphene layer.",
        4000,
      );
      return;
    }
    const wanted = roots.filter((r) => /^\d+$/.test(r) && r !== "0");
    if (wanted.length === 0) {
      StatusMessage.showTemporaryMessage(
        "Twig Capture: point at a neuron (or show some) and press Alt+T.",
        4000,
      );
      return;
    }
    const added = this.add(wanted);
    this.showSelectorTab();
    const allCached = wanted.every((r) => this.cached(r) !== undefined);
    const who = wanted.length === 1 ? wanted[0] : `${wanted.length} neurons`;
    StatusMessage.showTemporaryMessage(
      `Twig Capture: ${who} ` +
        (allCached
          ? "already predicted in this browser: Open in Twig Capture."
          : added
            ? `queued for merge prediction${this.isRunning ? " (running)" : ""}.`
            : "already listed."),
      4000,
    );
    if (!this.isRunning) await this.run(false);
  }

  private showSelectorTab() {
    const panels = this.layer.panels.panels;
    const panel =
      panels.find((p) => p.tabs.includes("twigSelector")) ?? panels[0];
    panel.selectedTab.value = "twigSelector";
  }

  /** Finished roots ticked for "Open ticked together". */
  ticked = new Set<string>();

  /**
   * Show several roots' results in the Twig Capture tab as ONE list, so
   * their candidates can be filtered, staged and merged together.
   */
  openTogether(roots: string[]) {
    const names = roots
      .map((r) => (this.cached(r) ?? this.cachedAnyLevel(r))?.name)
      .filter((n): n is string => n !== undefined);
    if (names.length === 0) return;
    if (names.length === 1) {
      this.open(roots[0]);
      return;
    }
    const store = this.layer.localTwigResults;
    store.pendingCombined = names;
    store.changed.dispatch();
    this.showCaptureTab();
  }

  private showCaptureTab() {
    // The side panel owns which tab is shown: pick the panel that holds Twig
    // Capture (the selected-layer panel unless it was pinned elsewhere).
    const panels = this.layer.panels.panels;
    const panel =
      panels.find((p) => p.tabs.includes("twigCapture")) ?? panels[0];
    panel.selectedTab.value = "twigCapture";
  }

  /** Show a root's result in the Twig Capture tab. */
  open(root: string) {
    const r = this.cached(root) ?? this.cachedAnyLevel(root);
    if (!r) return;
    const store = this.layer.localTwigResults;
    store.pendingOpen = r.name;
    store.changed.dispatch();
    // The side panel owns which tab is shown: pick the panel that holds Twig
    // Capture (the selected-layer panel unless it was pinned elsewhere).
    const panels = this.layer.panels.panels;
    const panel =
      panels.find((p) => p.tabs.includes("twigCapture")) ?? panels[0];
    panel.selectedTab.value = "twigCapture";
  }
}

export class TwigSelectorTab extends Tab {
  private list = document.createElement("div");
  private header = document.createElement("div");
  private runButton = document.createElement("button");
  private rerunButton = document.createElement("button");
  private togetherButton = document.createElement("button");

  constructor(public layer: SegmentationUserLayer) {
    super();
    const state = layer.twigSelector;
    const { element } = this;
    element.classList.add("neuroglancer-twig-capture-tab");
    element.classList.add("neuroglancer-twig-selector-tab");

    this.header.classList.add("neuroglancer-twig-capture-summary");
    element.appendChild(this.header);

    // Add objects.
    const addRow = document.createElement("div");
    addRow.classList.add("neuroglancer-twig-capture-controls");
    const addVisible = document.createElement("button");
    addVisible.classList.add("neuroglancer-twig-capture-merge");
    addVisible.textContent = "Add visible segments";
    addVisible.title =
      "Queue every segment currently shown in this layer. Select segments " +
      "in the 3D or slice view (double-click) or in the Seg. tab first.";
    addVisible.addEventListener("click", () => {
      const group = layer.displayState.segmentationGroupState.value;
      const ids = [...group.visibleSegments.keys()].map(String);
      const n = state.add(ids);
      if (n === 0) {
        StatusMessage.showTemporaryMessage(
          ids.length
            ? "Twig Selector: those segments are already queued."
            : "Twig Selector: no segments are visible in this layer.",
          3000,
        );
      }
    });
    addRow.appendChild(addVisible);
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "root id(s)";
    input.size = 22;
    input.title = "Paste one or more root ids, separated by spaces or commas.";
    const addTyped = () => {
      state.add(input.value.split(/[\s,;]+/).filter(Boolean));
      input.value = "";
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") addTyped();
    });
    addRow.appendChild(input);
    const addButton = document.createElement("button");
    addButton.classList.add("neuroglancer-twig-capture-merge");
    addButton.textContent = "Add";
    addButton.addEventListener("click", addTyped);
    addRow.appendChild(addButton);
    element.appendChild(addRow);

    element.appendChild(this.makeSettings());

    // Run controls.
    const runRow = document.createElement("div");
    runRow.classList.add("neuroglancer-twig-capture-controls");
    this.runButton.classList.add("neuroglancer-twig-capture-merge");
    this.runButton.addEventListener("click", () => {
      if (state.isRunning) state.stop();
      else void state.run(false);
    });
    runRow.appendChild(this.runButton);
    this.rerunButton.classList.add("neuroglancer-twig-capture-merge");
    this.rerunButton.textContent = "Re-run all";
    this.rerunButton.title =
      "Recompute every listed object, replacing the results cached in this browser.";
    this.rerunButton.addEventListener("click", () => {
      for (const i of state.items)
        if (i.status !== "running") i.status = "queued";
      void state.run(true);
    });
    runRow.appendChild(this.rerunButton);
    const clear = document.createElement("button");
    clear.classList.add("neuroglancer-twig-capture-merge");
    clear.textContent = "Clear finished";
    clear.addEventListener("click", () => state.clearFinished());
    runRow.appendChild(clear);
    this.togetherButton.classList.add("neuroglancer-twig-capture-merge");
    this.togetherButton.title =
      "Open the ticked neurons' suggestions in Twig Capture as one list, to " +
      "review, Stage all and Merge staged across all of them at once.";
    this.togetherButton.addEventListener("click", () =>
      state.openTogether(
        state.items.map((i) => i.root).filter((r) => state.ticked.has(r)),
      ),
    );
    runRow.appendChild(this.togetherButton);
    const diag = document.createElement("button");
    diag.classList.add("neuroglancer-twig-capture-merge");
    diag.textContent = "Copy diagnostics";
    diag.title =
      "Copy a plain-text report of this browser's GPU/CPU support, the " +
      "settings and the recent run log, to send to the Twig Capture team " +
      "when something goes wrong. Contains no login tokens.";
    diag.addEventListener("click", () => void this.copyReport("diagnostics"));
    runRow.appendChild(diag);
    element.appendChild(runRow);

    this.list.classList.add("neuroglancer-twig-capture-list");
    element.appendChild(this.list);

    this.registerDisposer(state.changed.add(() => this.render()));
    this.registerDisposer(
      layer.localTwigResults.changed.add(() => this.render()),
    );
    void webgpuAvailable().then((gpu) => {
      this.gpu = gpu;
      if (!this.wasDisposed) this.render();
    });
    this.render();
  }

  private gpu: boolean | undefined;

  private async copyReport(title: string, item?: QueueItem) {
    const state = this.layer.twigSelector;
    const ds = state.dataset();
    const report = await buildReport({
      title,
      error: item?.errorDetail ?? item?.error,
      run: {
        dataset: ds?.datastack,
        root: item?.root,
        progress: item?.progress,
        backendUsed: state.backendUsed,
        settings: state.settings,
        queue: state.items.map((i) => `${i.root}:${i.status}`),
      },
    });
    if (await copyText(report)) {
      StatusMessage.showTemporaryMessage(
        "Twig Selector: report copied. Paste it into a message to the Twig Capture team.",
        5000,
      );
    }
  }

  private makeSettings() {
    const state = this.layer.twigSelector;
    const st = state.settings;
    const row = document.createElement("div");
    row.classList.add("neuroglancer-twig-capture-filters");
    const field = (label: string, el: HTMLElement, title: string) => {
      const wrap = document.createElement("label");
      wrap.title = title;
      wrap.appendChild(document.createTextNode(label + " "));
      wrap.appendChild(el);
      row.appendChild(wrap);
    };
    const num = (
      value: number,
      min: number,
      max: number,
      step: number,
      set: (v: number) => void,
    ) => {
      const el = document.createElement("input");
      el.type = "number";
      el.min = String(min);
      el.max = String(max);
      el.step = String(step);
      el.value = String(value);
      el.style.width = "4.5em";
      el.addEventListener("change", () => {
        const v = Number(el.value);
        if (Number.isFinite(v) && v >= min && v <= max) {
          set(v);
          state.saveSettings();
        } else el.value = String(value);
      });
      return el;
    };
    field(
      "Level",
      num(st.level, 1, 4, 1, (v) => (st.level = Math.round(v))),
      "1: merges onto the selected object. 2: also merges onto the level-1 " +
        "suggestions with p >= the chain threshold (treated as already " +
        "merged), and so on.",
    );
    field(
      "chain p ≥",
      num(st.chainThreshold, 0.05, 1, 0.05, (v) => (st.chainThreshold = v)),
      "Level 2+: suggestions at least this confident are treated as merged " +
        "before searching around them.",
    );
    const auto = document.createElement("input");
    auto.type = "checkbox";
    auto.checked = st.auto;
    field(
      "auto",
      auto,
      "Choose max regions, partner size and sites/pair for each neuron from " +
        "its size and your device: every region up to 300 on GPU (3 on CPU), " +
        "spread over the neuron above that; partners of any size for a " +
        "fragment (<= 50 L2 nodes), so it can find the neuron it belongs to, " +
        "and small twigs (<= 20 L2) for a neuron; 8 sites/pair on GPU, 1 on " +
        "CPU. Each queued neuron shows its plan. Untick to set them by hand.",
    );
    const manual: (HTMLInputElement | HTMLSelectElement)[] = [];
    const applyAuto = () => {
      for (const el of manual) {
        el.disabled = st.auto;
        el.title = st.auto ? "Chosen per neuron (auto is on)." : "";
      }
    };
    auto.addEventListener("change", () => {
      st.auto = auto.checked;
      applyAuto();
      state.saveSettings();
    });
    const regionsEl = num(
      st.maxRegions,
      1,
      100000,
      50,
      (v) => (st.maxRegions = Math.round(v)),
    );
    manual.push(regionsEl);
    field(
      "max regions",
      regionsEl,
      "Per level. One region is a 6.4 x 6.4 um search window around one " +
        "1.1 um bin of the object's L2 graph: a fragment needs a few, most " +
        "neurons 30-200, big ones several hundred. On a laptop GPU a region " +
        "takes ~15-50 s. 200 covers most neurons completely; above the cap, " +
        "regions are spread evenly over the whole neuron. On CPU use 1-3.",
    );
    field(
      "candidate L2 ≤",
      (() => {
        const el = num(
          st.maxCandidateL2 ?? NO_L2_LIMIT,
          1,
          1000000,
          10,
          (v) => (st.maxCandidateL2 = Math.round(v)),
        );
        manual.push(el);
        return el;
      })(),
      "Largest partner proposed, in L2 nodes. No limit by default " +
        `(${NO_L2_LIMIT}). Partners above ${LARGE_MERGE_L2} L2 -- the size ` +
        "the model was trained on -- are tagged 'large merge' in Twig " +
        `Capture: check them before merging. Set ${LARGE_MERGE_L2} to ` +
        "match the cluster runner.",
    );
    const sites = document.createElement("select");
    for (const [v, label] of [
      [8, "8 (as cluster)"],
      [1, "1 (faster)"],
    ] as const) {
      const o = document.createElement("option");
      o.value = String(v);
      o.textContent = label;
      sites.appendChild(o);
    }
    sites.value = String(st.maxSites);
    manual.push(sites);
    sites.addEventListener("change", () => {
      st.maxSites = Number(sites.value);
      state.saveSettings();
    });
    field(
      "sites/pair",
      sites,
      "Contact sites scored per pair (the largest first). The cluster runner " +
        "scores up to 8 and keeps the max; 1 is several times less work, " +
        "useful on CPU, but can miss a merge whose best site is not the largest.",
    );
    const gapSel = document.createElement("select");
    for (const [v, label] of [
      [0, "0 (touching only)"],
      [1, "1 voxel (default)"],
      [2, "2 voxels"],
    ] as const) {
      const o = document.createElement("option");
      o.value = String(v);
      o.textContent = label;
      gapSel.appendChild(o);
    }
    gapSel.value = String(st.contactGap);
    gapSel.addEventListener("change", () => {
      st.contactGap = Number(gapSel.value);
      state.saveSettings();
    });
    field(
      "contact gap",
      gapSel,
      "Pieces up to this many supervoxel voxels apart (32 nm; FANC 34.4 nm) " +
        "count as touching, as the cluster runner's --contact_gap. Merges " +
        'across a thin gap become candidates; they are tagged "gap" in Twig ' +
        "Capture, and are less reliable than touching ones. 0 is the old " +
        "touching-only search.",
    );
    const precSel = document.createElement("select");
    for (const [v, label] of [
      ["f32", "f32 (exact)"],
      ["f16", "f16 (faster)"],
    ] as const) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = label;
      precSel.appendChild(o);
    }
    precSel.value = st.gpuPrecision;
    precSel.addEventListener("change", () => {
      st.gpuPrecision = precSel.value as "f32" | "f16";
      state.saveSettings();
    });
    field(
      "GPU precision",
      precSel,
      "f16 stores the WebGPU network's activations and weights as half floats " +
        "(arithmetic stays f32): ~1.4x faster on an Apple M1 Max, with p within " +
        "~1e-4 of f32. Needs a GPU with shader-f16; otherwise f32 is used.",
    );
    const backend = document.createElement("select");
    for (const [v, label] of [
      ["auto", "auto"],
      ["webgpu", "GPU (WebGPU)"],
      ["webgpu-ort", "GPU (ONNX Runtime)"],
      ["wasm", "CPU"],
    ] as const) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = label;
      backend.appendChild(o);
    }
    backend.value = st.backend;
    backend.addEventListener("change", () => {
      st.backend = backend.value as Backend;
      void state.willUseGpu().then((g) => {
        state.gpuKnown = g;
        state.changed.dispatch();
      });
      state.saveSettings();
    });
    field(
      "device",
      backend,
      "auto uses the GPU when the browser offers WebGPU, else the CPU. CPU " +
        "works everywhere but is much slower. GPU runs Twig Capture's own " +
        "WebGPU network (checked against ONNX Runtime the first time); " +
        "GPU (ONNX Runtime) is the slower previous path.",
    );
    const model = document.createElement("input");
    model.type = "text";
    model.value = st.modelUrl;
    model.size = 28;
    model.addEventListener("change", () => {
      st.modelUrl = model.value.trim() || DEFAULT_MODEL_URL;
      model.value = st.modelUrl;
      state.saveSettings();
    });
    field(
      "model",
      model,
      "URL of the ONNX model (Conv2d rewrite, see twig_capture_inference/). " +
        "Downloaded once and cached in this browser.",
    );
    const file = document.createElement("input");
    file.type = "file";
    file.accept = ".onnx";
    file.style.display = "none";
    file.addEventListener("change", async () => {
      const f = file.files?.[0];
      if (!f) return;
      st.modelUrl = await putLocalModel(f);
      model.value = st.modelUrl;
      state.saveSettings();
    });
    const pick = document.createElement("button");
    pick.classList.add("neuroglancer-twig-capture-merge");
    pick.textContent = "…";
    pick.title =
      "Use a model file from this computer (cached in this browser).";
    pick.addEventListener("click", () => file.click());
    row.appendChild(pick);
    row.appendChild(file);
    applyAuto();
    return row;
  }

  private render() {
    const state = this.layer.twigSelector;
    const ds = state.dataset();
    removeChildren(this.header);
    const line = document.createElement("div");
    line.textContent = ds
      ? `${ds.label} (${ds.datastack}) · ${this.gpu === undefined ? "" : this.gpu ? "GPU available" : "no GPU: runs on CPU"}`
      : "Not a BANC, Aedes or CRANTb graphene layer: in-browser Twig Capture is unavailable.";
    this.header.appendChild(line);
    if (ds?.note) {
      const note = document.createElement("div");
      note.classList.add("neuroglancer-twig-capture-hint");
      note.textContent = ds.note;
      this.header.appendChild(note);
    }
    const queued = state.items.filter((i) => i.status === "queued").length;
    this.runButton.textContent = state.isRunning
      ? "Stop"
      : `Run${queued ? ` (${queued})` : ""}`;
    this.runButton.disabled = !ds || (!state.isRunning && queued === 0);
    this.rerunButton.disabled =
      !ds || state.isRunning || state.items.length === 0;
    const withResult = state.items.filter(
      (i) => state.cached(i.root) ?? state.cachedAnyLevel(i.root),
    );
    for (const r of [...state.ticked]) {
      if (!withResult.some((i) => i.root === r)) state.ticked.delete(r);
    }
    const nTicked = state.ticked.size;
    this.togetherButton.textContent = `Open ticked together (${nTicked})`;
    this.togetherButton.disabled = nTicked < 2;
    this.togetherButton.style.display = withResult.length >= 2 ? "" : "none";

    removeChildren(this.list);
    if (state.items.length === 0) {
      const hint = document.createElement("div");
      hint.classList.add("neuroglancer-twig-capture-hint");
      hint.textContent =
        "Add objects to suggest merges for. Results open in the Twig Capture " +
        "tab and are kept in this browser.";
      this.list.appendChild(hint);
    }
    for (const item of state.items) {
      const row = document.createElement("div");
      row.classList.add("neuroglancer-twig-capture-row");
      const main = document.createElement("div");
      main.classList.add("neuroglancer-twig-capture-main");
      if (state.cached(item.root) ?? state.cachedAnyLevel(item.root)) {
        const tick = document.createElement("input");
        tick.type = "checkbox";
        tick.checked = state.ticked.has(item.root);
        tick.title = 'Include in "Open ticked together"';
        tick.addEventListener("change", () => {
          if (tick.checked) state.ticked.add(item.root);
          else state.ticked.delete(item.root);
          this.render();
        });
        main.appendChild(tick);
      }
      const id = document.createElement("span");
      id.classList.add("neuroglancer-twig-capture-id");
      id.textContent = item.root;
      main.appendChild(id);
      // Any result for this root, even one shallower than the level now set.
      const cached = state.cached(item.root) ?? state.cachedAnyLevel(item.root);
      const status = document.createElement("span");
      status.classList.add("neuroglancer-twig-capture-tag");
      const p = item.progress;
      if (item.status === "running" && p) {
        status.textContent =
          `${p.phase}${p.total ? ` ${p.done}/${p.total}` : ""} · ${p.patches} patches · ${p.found} found` +
          (p.msPerPatch ? ` · model ${p.msPerPatch} ms/patch` : "");
      } else if (item.status === "failed") {
        status.classList.add("neuroglancer-twig-capture-tag-warn");
        status.textContent = "failed";
        status.title = item.error ?? "";
      } else if (cached) {
        const lv = (cached as any).level ?? 1;
        status.textContent =
          `${cached.candidates.length} suggestion(s)` +
          (lv > 1 || item.status === "queued" ? ` · level ${lv}` : "");
        status.title = `${cached.name}\n${cached.timestamp ?? ""}`;
      } else {
        status.textContent = item.status;
      }
      main.appendChild(status);
      row.appendChild(main);
      // Regions that failed even after retries (usually the connection):
      // the result is incomplete, and nothing else on the row says so.
      const failedRegions = (cached as any)?.benchmark?.region_errors ?? 0;
      if (failedRegions > 0) {
        const warn = document.createElement("div");
        warn.classList.add("neuroglancer-twig-capture-hint");
        warn.style.color = "#f88";
        const searched = (cached as any)?.binned_regions_searched;
        warn.textContent =
          `${failedRegions}` +
          (typeof searched === "number" ? ` of ${searched}` : "") +
          " region(s) failed, so this result is incomplete. ";
        warn.title =
          "Usually a dropped connection while reading image chunks. " +
          "Copy report gives the details.";
        if (item.status === "done") {
          const again = document.createElement("button");
          again.textContent = "Re-run";
          again.title =
            "Run this neuron again (queued; it replaces this result).";
          again.addEventListener("click", (e) => {
            e.stopPropagation();
            item.status = "queued";
            state.changed.dispatch();
            if (!state.isRunning) void state.run(false);
          });
          warn.appendChild(again);
        }
        row.appendChild(warn);
      }
      if (item.updatedFrom) {
        const upd = document.createElement("div");
        upd.classList.add("neuroglancer-twig-capture-hint");
        upd.textContent =
          `updated from outdated root ${item.updatedFrom}` +
          (item.updatedShare !== undefined && item.updatedShare < 0.95
            ? ` (split since: this version holds ${Math.round(item.updatedShare * 100)}% of it)`
            : "");
        row.appendChild(upd);
      }
      // What this neuron will be run with, once it has been measured.
      const plan = state.planFor(item);
      if (item.size && plan && item.status !== "done") {
        const line = document.createElement("div");
        line.classList.add("neuroglancer-twig-capture-hint");
        const capped = item.size.regions > plan.maxRegions;
        const gpu = state.gpuKnown ?? true;
        const mins = Math.max(1, Math.round(plan.minutes));
        line.textContent =
          `${item.size.l2} L2 · ${item.size.regions} region(s)` +
          (capped ? `, ${plan.maxRegions} searched (spread)` : "") +
          ` · partners ${plan.maxCandidateL2 >= NO_L2_LIMIT ? "any size" : `≤ ${plan.maxCandidateL2} L2`}` +
          ` · ${plan.maxSites} site(s)/pair` +
          (item.status === "queued"
            ? ` · ~${mins} min (${gpu ? "GPU" : "CPU"})`
            : "");
        line.title =
          (plan.maxCandidateL2 >= NO_L2_LIMIT
            ? `Partners of any size are scored; those above ${LARGE_MERGE_L2} L2 are tagged 'large merge'. `
            : "") + "The time is a rough estimate (dense tissue is slower).";
        row.appendChild(line);
      }
      const actions = document.createElement("div");
      actions.classList.add("neuroglancer-twig-capture-actions");
      if (cached) {
        const open = document.createElement("button");
        open.classList.add("neuroglancer-twig-capture-merge");
        open.textContent = "Open in Twig Capture";
        open.addEventListener("click", () => state.open(item.root));
        actions.appendChild(open);
      }
      if (item.status === "failed") {
        const rep = document.createElement("button");
        rep.classList.add("neuroglancer-twig-capture-merge");
        rep.textContent = "Copy report";
        rep.title =
          "Copy the error, this machine's GPU/CPU details and the run log, " +
          "to paste into a message to the Twig Capture team.";
        rep.addEventListener(
          "click",
          () => void this.copyReport(`root ${item.root} failed`, item),
        );
        actions.appendChild(rep);
      }
      if (item.status !== "running") {
        const rm = document.createElement("button");
        rm.classList.add("neuroglancer-twig-capture-merge");
        rm.textContent = "×";
        rm.title = "Remove from the list (the cached result is kept).";
        rm.addEventListener("click", () => state.remove(item.root));
        actions.appendChild(rm);
      }
      row.appendChild(actions);
      if (item.status === "failed" && item.error) {
        const err = document.createElement("div");
        err.classList.add("neuroglancer-twig-capture-hint");
        err.textContent = item.error;
        row.appendChild(err);
      }
      this.list.appendChild(row);
    }
  }
}
