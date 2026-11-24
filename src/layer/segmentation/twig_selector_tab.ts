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
import type { TwigDataset } from "#src/layer/segmentation/twig_inference/datasets.js";
import {
  DEFAULT_MODEL_URL,
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
  runTwigCapture,
} from "#src/layer/segmentation/twig_inference/pipeline.js";
import { StatusMessage } from "#src/status.js";
import { removeChildren } from "#src/util/dom.js";
import { NullarySignal } from "#src/util/signal.js";
import { Tab } from "#src/widget/tab_view.js";

const SETTINGS_KEY = "neuroglancer-twig-selector-settings";

export interface SelectorSettings {
  level: number;
  chainThreshold: number;
  maxRegions: number;
  maxSites: number;
  /** Largest candidate proposed, in L2 nodes (cluster runner: 20). */
  maxCandidateL2: number;
  backend: Backend;
  modelUrl: string;
}

const DEFAULT_SETTINGS: SelectorSettings = {
  level: 1,
  chainThreshold: 0.75,
  maxRegions: 400,
  maxSites: 8,
  maxCandidateL2: 20,
  backend: "auto",
  modelUrl: DEFAULT_MODEL_URL,
};

function loadSettings(): SelectorSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
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
      this.items.push(
        hit
          ? { root, status: "done", resultName: hit.name }
          : { root, status: "queued" },
      );
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
  cached(root: string) {
    const r = this.layer.localTwigResults.forRoot(
      root,
      this.dataset()?.datastack,
    );
    if (r === undefined) return undefined;
    const x = r as any;
    const st = this.settings;
    // The result must have looked at least as far as the settings now ask:
    // as deep a level, as large a partner, and every region unless it was
    // capped at or above today's cap.
    if ((x.level ?? 1) < st.level) return undefined;
    if ((x.filters?.max_candidate_l2_nodes ?? 20) < (st.maxCandidateL2 ?? 20)) {
      return undefined;
    }
    const searched = x.binned_regions_searched;
    const total = x.binned_regions_total;
    if (
      typeof searched === "number" &&
      typeof total === "number" &&
      searched < total &&
      searched < st.maxRegions
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
        const cached = this.cached(item.root);
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
          const key = `${st.modelUrl}|${st.backend}`;
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
              maxRegions: st.maxRegions,
              maxSites: st.maxSites,
              maxCandidateL2: st.maxCandidateL2 ?? 20,
              modelUrl: st.modelUrl,
            },
            (p) => {
              item.progress = p;
              this.changed.dispatch();
            },
            signal,
          );
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
    field(
      "max regions",
      num(st.maxRegions, 1, 100000, 50, (v) => (st.maxRegions = Math.round(v))),
      "Per level. One region is a 6.4 x 6.4 um search window around one " +
        "1.1 um bin of the object's L2 graph; a fragment needs a few, a " +
        "whole neuron hundreds to thousands.",
    );
    field(
      "candidate L2 ≤",
      num(
        st.maxCandidateL2 ?? 20,
        1,
        1000000,
        10,
        (v) => (st.maxCandidateL2 = Math.round(v)),
      ),
      "Largest partner proposed, in L2 nodes. 20 (as the cluster runner) " +
        "suits a neuron collecting small twigs. To find where a small " +
        "FRAGMENT belongs -- usually onto a big neuron -- raise it (e.g. " +
        "100000) so large partners are scored too.",
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
    const backend = document.createElement("select");
    for (const [v, label] of [
      ["auto", "auto"],
      ["webgpu", "GPU (WebGPU)"],
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
      state.saveSettings();
    });
    field(
      "device",
      backend,
      "auto uses the GPU when the browser offers WebGPU, else the CPU. CPU " +
        "works everywhere but is much slower.",
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
        status.textContent = `${p.phase}${p.total ? ` ${p.done}/${p.total}` : ""} · ${p.patches} patches · ${p.found} found`;
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
