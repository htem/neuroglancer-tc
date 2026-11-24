/**
 * @license
 * Copyright 2026 Alex Bates
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
 * MVB review tab: score candidate multivesicular bodies one at a time.
 *
 * Shown on a segmentation layer whose source is the BANC mitochondria model.
 * It reads a queue JSON over HTTP, jumps to each object, shows that object
 * alone in this layer, and records MVB / Not MVB to the CAVE table
 * multi_vesicular_bodies (mvb_review_decisions.ts). Unsure is kept in this
 * browser only.
 *
 * QUEUE FORMAT (written by the dcv repo's python/mvb_rank.py ngqueue):
 *   { "table": "multi_vesicular_bodies", "voxel_nm": [4, 4, 45],
 *     "items": [ { "code": "R001", "mito_id": "12598170",
 *                  "pt": [x, y, z] }, ... ] }
 * `pt` is a point INSIDE the object, in 4 x 4 x 45 nm voxels. The queue is
 * blind: no score, no host cell, mixed top-ranked and random objects.
 *
 * Keys, while the tab has focus: m = MVB, n = not MVB, u = unsure,
 * left/right arrow = previous/next.
 */

import type { SegmentationUserLayer } from "#src/layer/segmentation/index.js";
import type {
  MvbRow,
  MvbTag,
} from "#src/layer/segmentation/mvb_review_decisions.js";
import {
  MVB_TABLE,
  MVB_VOXEL_NM,
  MvbStore,
  mvbReviewAllowed,
  pointKey,
} from "#src/layer/segmentation/mvb_review_decisions.js";
import { StatusMessage } from "#src/status.js";
import { NullarySignal } from "#src/util/signal.js";
import { Tab } from "#src/widget/tab_view.js";

export const MVB_REVIEW_JSON_KEY = "mvbReview";
const MVB_REVIEW_VERSION = "MVB review v0.1 · 20261113";
const QUEUE_URL_KEY = "neuroglancer-mvb-review-queue-url";
/** Served next to the viewer's index.html unless a URL is given. */
const DEFAULT_QUEUE_URL = "mvb_queue.json";

interface QueueItem {
  code: string;
  mito_id: string;
  pt: number[];
  key: string;
}

/** Shareable state: which queue, which object. */
export class MvbReviewState {
  changed = new NullarySignal();
  queueUrl = "";
  code = "";

  toJSON() {
    const x: { [k: string]: string } = {};
    if (this.queueUrl) x.queueUrl = this.queueUrl;
    if (this.code) x.code = this.code;
    return Object.keys(x).length ? x : undefined;
  }

  /** Tolerates anything: a throwing restoreState takes down the viewer. */
  restoreState(obj: unknown) {
    if (obj === undefined || obj === null || typeof obj !== "object") return;
    const o = obj as { [k: string]: any };
    if (typeof o.queueUrl === "string") this.queueUrl = o.queueUrl;
    if (typeof o.code === "string") this.code = o.code;
    this.changed.dispatch();
  }
}

function parseQueue(text: string): QueueItem[] {
  const d = JSON.parse(text);
  if (d.table !== undefined && d.table !== MVB_TABLE) {
    throw new Error(`queue is for table "${d.table}", not ${MVB_TABLE}`);
  }
  const vox = d.voxel_nm;
  if (
    !Array.isArray(vox) ||
    vox.length !== 3 ||
    vox.some((v: number, i: number) => v !== MVB_VOXEL_NM[i])
  ) {
    throw new Error(`queue voxel_nm must be [${MVB_VOXEL_NM.join(", ")}]`);
  }
  const items: QueueItem[] = [];
  for (const it of d.items ?? []) {
    if (
      typeof it.code !== "string" ||
      !Array.isArray(it.pt) ||
      it.pt.length !== 3 ||
      it.pt.some((v: unknown) => typeof v !== "number" || !Number.isFinite(v))
    ) {
      continue;
    }
    items.push({
      code: it.code,
      // Ids are above 2^53 in some tables: keep them as strings.
      mito_id: String(it.mito_id ?? ""),
      pt: it.pt,
      key: pointKey(it.pt),
    });
  }
  if (items.length === 0) throw new Error("queue has no valid items");
  return items;
}

export class MvbReviewTab extends Tab {
  private items: QueueItem[] = [];
  private index = -1;
  private calls = new Map<string, MvbRow>();
  private unsure = new Set<string>();
  private store = new MvbStore();
  private posting = false;
  private skipScored = true;
  private queueUrl = "";

  private urlInput = document.createElement("input");
  private statusEl = document.createElement("div");
  private codeEl = document.createElement("div");
  private detailEl = document.createElement("div");
  private callEl = document.createElement("div");
  private buttons: HTMLButtonElement[] = [];

  constructor(public layer: SegmentationUserLayer) {
    super();
    const { element } = this;
    element.classList.add("neuroglancer-mvb-review-tab");
    element.tabIndex = 0;

    // Queue URL row.
    const row = document.createElement("div");
    row.classList.add("neuroglancer-mvb-review-row");
    this.urlInput.type = "text";
    this.urlInput.classList.add("neuroglancer-mvb-review-url");
    this.urlInput.placeholder = "queue JSON URL";
    this.urlInput.title =
      "A queue JSON from python/mvb_rank.py ngqueue, served over HTTP. " +
      "Remembered in this browser.";
    this.urlInput.value = this.queueUrl =
      layer.mvbReview.queueUrl ||
      readStored(QUEUE_URL_KEY) ||
      DEFAULT_QUEUE_URL;
    this.urlInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") void this.load();
    });
    row.appendChild(this.urlInput);
    row.appendChild(
      this.button(
        "Load",
        "Read the queue and the table",
        () => void this.load(),
      ),
    );
    row.appendChild(
      this.button(
        "↻",
        "Re-read the table: shows other reviewers' calls",
        () => void this.refresh(),
      ),
    );
    element.appendChild(row);

    const skip = document.createElement("label");
    skip.classList.add("neuroglancer-mvb-review-row");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = this.skipScored;
    box.addEventListener("change", () => {
      this.skipScored = box.checked;
    });
    skip.appendChild(box);
    skip.appendChild(
      document.createTextNode(
        " skip objects already scored (by anyone) or marked unsure by you",
      ),
    );
    element.appendChild(skip);

    this.statusEl.classList.add("neuroglancer-mvb-review-status");
    element.appendChild(this.statusEl);

    this.codeEl.classList.add("neuroglancer-mvb-review-code");
    element.appendChild(this.codeEl);
    this.detailEl.classList.add("neuroglancer-mvb-review-detail");
    element.appendChild(this.detailEl);
    this.callEl.classList.add("neuroglancer-mvb-review-call");
    element.appendChild(this.callEl);

    const verdicts = document.createElement("div");
    verdicts.classList.add("neuroglancer-mvb-review-row");
    verdicts.appendChild(
      this.button(
        "MVB (m)",
        "Record mvb in CAVE",
        () => void this.decide("mvb"),
        "mvb",
      ),
    );
    verdicts.appendChild(
      this.button(
        "Not MVB (n)",
        "Record not_mvb in CAVE",
        () => void this.decide("not_mvb"),
        "not",
      ),
    );
    verdicts.appendChild(
      this.button(
        "Unsure (u)",
        "Skip; remembered in this browser only, nothing written",
        () => this.markUnsure(),
      ),
    );
    element.appendChild(verdicts);

    const nav = document.createElement("div");
    nav.classList.add("neuroglancer-mvb-review-row");
    nav.appendChild(
      this.button("← Back", "Previous object", () => this.step(-1, false)),
    );
    nav.appendChild(
      this.button("Next →", "Next object", () => this.step(1, this.skipScored)),
    );
    nav.appendChild(
      this.button("Go to", "Jump back to the current object", () =>
        this.show(),
      ),
    );
    element.appendChild(nav);

    const help = document.createElement("div");
    help.classList.add("neuroglancer-mvb-review-help");
    help.textContent =
      "Click in this panel, then: m = MVB, n = not MVB, u = unsure, ←/→ = previous/next. " +
      "The model object is shown alone in this layer; judge the EM. " +
      "Calls go to the CAVE table multi_vesicular_bodies under your name; a new call on the same object replaces the old one.";
    element.appendChild(help);

    const version = document.createElement("div");
    version.classList.add("neuroglancer-mvb-review-version");
    version.textContent = MVB_REVIEW_VERSION;
    element.appendChild(version);

    element.addEventListener("keydown", (e) => this.onKey(e));
    this.render();
    if (this.queueUrl) void this.load();
  }

  private button(label: string, title: string, fn: () => void, kind?: string) {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = title;
    if (kind) b.classList.add(`neuroglancer-mvb-review-${kind}`);
    b.addEventListener("click", fn);
    this.buttons.push(b);
    return b;
  }

  private onKey(e: KeyboardEvent) {
    if (e.target instanceof HTMLInputElement && e.target.type === "text")
      return;
    const k = e.key;
    let handled = true;
    if (k === "m") void this.decide("mvb");
    else if (k === "n") void this.decide("not_mvb");
    else if (k === "u") this.markUnsure();
    else if (k === "ArrowRight") this.step(1, this.skipScored);
    else if (k === "ArrowLeft") this.step(-1, false);
    else handled = false;
    if (handled) {
      // Neuroglancer binds arrows (move) and letters on the viewer; while
      // this panel has focus the keys are ours.
      e.preventDefault();
      e.stopPropagation();
    }
  }

  private allowed(): boolean {
    return mvbReviewAllowed(
      this.layer.dataSources.map((ds) => ds.spec?.url ?? ""),
    );
  }

  private unsureKey() {
    return `neuroglancer-mvb-review-unsure:${this.queueUrl}`;
  }

  private async load() {
    const url = this.urlInput.value.trim();
    if (!url) return;
    this.queueUrl = url;
    try {
      localStorage.setItem(QUEUE_URL_KEY, url);
    } catch {
      // storage blocked; the URL still works for this session
    }
    this.statusEl.textContent = "loading queue…";
    try {
      const r = await fetch(url, { cache: "reload" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      this.items = parseQueue(await r.text());
    } catch (err) {
      this.items = [];
      this.index = -1;
      this.statusEl.textContent = `queue not loaded: ${(err as Error).message}`;
      this.render();
      return;
    }
    try {
      this.unsure = new Set(
        JSON.parse(localStorage.getItem(this.unsureKey()) ?? "[]"),
      );
    } catch {
      this.unsure = new Set();
    }
    this.publish();
    await this.refresh();
    // Open on the shared link's object, else the first one left to score.
    const saved = this.items.findIndex(
      (it) => it.code === this.layer.mvbReview.code,
    );
    if (saved >= 0) {
      this.index = saved;
      this.show();
    } else {
      this.index = -1;
      this.step(1, true);
    }
  }

  private async refresh() {
    try {
      this.calls = await this.store.fetchAll();
    } catch (err) {
      StatusMessage.showTemporaryMessage(
        `MVB review: could not read ${MVB_TABLE}: ${(err as Error).message}`,
        6000,
      );
    }
    this.render();
  }

  private isDone(it: QueueItem) {
    return this.calls.has(it.key) || this.unsure.has(it.code);
  }

  /** Move by `dir`, optionally skipping objects already scored. */
  private step(dir: number, skip: boolean) {
    if (this.items.length === 0) return;
    let i = this.index;
    for (;;) {
      i += dir;
      if (i < 0 || i >= this.items.length) {
        if (dir > 0 && skip) {
          StatusMessage.showTemporaryMessage(
            "MVB review: no objects left to score in this queue.",
            4000,
          );
        }
        this.render();
        return;
      }
      if (!skip || !this.isDone(this.items[i])) break;
    }
    this.index = i;
    this.show();
  }

  /** Jump to the current object and show it alone in this layer. */
  private show() {
    const it = this.items[this.index];
    if (it === undefined) return;
    this.layer.mvbReview.code = it.code;
    this.layer.mvbReview.changed.dispatch();
    const space = this.layer.manager.root.coordinateSpace.value;
    const { rank, scales, units } = space;
    const { globalPosition } = this.layer.manager.root;
    if (space.valid && rank >= 3 && units.slice(0, 3).every((u) => u === "m")) {
      const position = Float32Array.from(globalPosition.value);
      for (let i = 0; i < 3; ++i) {
        // queue voxels -> nm -> viewer voxels (scales are in metres). Voxel
        // centre: +0.5.
        position[i] = ((it.pt[i] + 0.5) * MVB_VOXEL_NM[i] * 1e-9) / scales[i];
      }
      globalPosition.value.set(position);
      globalPosition.changed.dispatch();
    } else {
      StatusMessage.showTemporaryMessage(
        "MVB review: viewer coordinate space is not in metres; cannot navigate.",
        4000,
      );
    }
    const group = this.layer.displayState.segmentationGroupState.value;
    group.visibleSegments.clear();
    group.selectedSegments.clear();
    try {
      const id = BigInt(it.mito_id);
      group.selectedSegments.add(id);
      group.visibleSegments.set(id, true);
    } catch {
      // no id: position alone still shows the object
    }
    this.render();
    this.element.focus({ preventScroll: true });
  }

  private async decide(tag: MvbTag) {
    const it = this.items[this.index];
    if (it === undefined || this.posting) return;
    if (!this.allowed()) {
      StatusMessage.showTemporaryMessage(
        "MVB review: calls can only be recorded on the BANC mitochondria-model layer.",
        5000,
      );
      return;
    }
    this.posting = true;
    this.render();
    try {
      const id = await this.store.post(it.pt, tag);
      this.calls.set(it.key, { id, tag, created: Date.now(), key: it.key });
      if (this.unsure.delete(it.code)) this.saveUnsure();
      StatusMessage.showTemporaryMessage(
        `MVB review: ${it.code} recorded as ${tag} (row ${id}).`,
        2500,
      );
    } catch (err) {
      StatusMessage.showTemporaryMessage(
        `MVB review: ${(err as Error).message}`,
        8000,
      );
      this.posting = false;
      this.render();
      return;
    }
    this.posting = false;
    this.step(1, this.skipScored);
  }

  private markUnsure() {
    const it = this.items[this.index];
    if (it === undefined) return;
    this.unsure.add(it.code);
    this.saveUnsure();
    this.step(1, this.skipScored);
  }

  private saveUnsure() {
    try {
      localStorage.setItem(this.unsureKey(), JSON.stringify([...this.unsure]));
    } catch {
      // storage blocked: unsure lasts for this session only
    }
  }

  private publish() {
    this.layer.mvbReview.queueUrl = this.queueUrl;
    this.layer.mvbReview.changed.dispatch();
  }

  private render() {
    const n = this.items.length;
    if (n === 0) {
      if (!this.statusEl.textContent)
        this.statusEl.textContent = "no queue loaded";
      this.codeEl.textContent = "";
      this.detailEl.textContent = "";
      this.callEl.textContent = "";
    } else {
      let mvb = 0;
      let not = 0;
      let uns = 0;
      for (const it of this.items) {
        const c = this.calls.get(it.key);
        if (c?.tag === "mvb") mvb++;
        else if (c?.tag === "not_mvb") not++;
        else if (this.unsure.has(it.code)) uns++;
      }
      this.statusEl.textContent = `${n} objects · MVB ${mvb} · not MVB ${not} · unsure (you) ${uns} · left ${n - mvb - not - uns}`;
      const it = this.items[this.index];
      this.codeEl.textContent = it
        ? `${it.code}  (${this.index + 1} of ${n})`
        : "";
      this.detailEl.textContent = it
        ? `model object ${it.mito_id} · point ${it.pt.join(", ")} (4 × 4 × 45 nm voxels)`
        : "";
      const c = it ? this.calls.get(it.key) : undefined;
      this.callEl.textContent = this.posting
        ? "recording…"
        : c
          ? `already scored: ${c.tag}`
          : it && this.unsure.has(it.code)
            ? "you marked this unsure"
            : "";
    }
    const ok = this.allowed() && n > 0 && !this.posting;
    for (const b of this.buttons) {
      if (
        b.classList.contains("neuroglancer-mvb-review-mvb") ||
        b.classList.contains("neuroglancer-mvb-review-not")
      ) {
        b.disabled = !ok;
      }
    }
  }
}

function readStored(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}
