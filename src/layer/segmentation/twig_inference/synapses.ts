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
 * cand_synapses_total for an in-browser result, from the dataset's CAVE
 * synapse table: the synapse rows whose pre_pt_root_id or post_pt_root_id is
 * the partner (target_segment_id), as of the result's timestamp. A synapse
 * with both ends on the partner counts twice. The same rule as the cluster's
 * scripts/add_cand_synapses_cave.py, so browser and cluster numbers compare.
 *
 * As of the result's timestamp, not now: a reviewer's merge retires the
 * partner root, and a query at "now" would miss exactly the partners
 * already merged.
 */

import type { DecisionStore } from "#src/layer/segmentation/twig_capture_decisions.js";

/** Partner roots per request: they go in the body, but replies grow fast. */
const CHUNK = 100;

/** {root: synapse rows with `col` == root}, for these roots. */
async function countSide(
  store: DecisionStore,
  table: string,
  col: string,
  roots: string[],
  timestamp: string,
  signal?: AbortSignal,
): Promise<Map<string, number>> {
  const out = new Map<string, number>();
  for (let i = 0; i < roots.length; i += CHUNK) {
    const chunk = roots.slice(i, i + CHUNK);
    const rows = await store.liveQuery(
      table,
      timestamp,
      `,"filter_in_dict":{"${table}":{"${col}":[${chunk.join(",")}]}}` +
        // "id" is required: the server joins its materialized and live
        // halves on it.
        `,"select_columns":{"${table}":["id","${col}"]}`,
      signal,
    );
    for (const r of rows) {
      const k = String(r[col]);
      out.set(k, (out.get(k) ?? 0) + 1);
    }
  }
  return out;
}

/**
 * Fill cand_synapses_total on every candidate of `result`, in place, and
 * record where the numbers came from.
 */
export async function addSynapseCounts(
  result: any,
  store: DecisionStore,
  table: string,
  signal?: AbortSignal,
): Promise<void> {
  const cands: any[] = result.candidates ?? [];
  if (cands.length === 0) return;
  const roots = [...new Set(cands.map((c) => String(c.target_segment_id)))];
  const ts = String(result.timestamp);
  const pre = await countSide(
    store,
    table,
    "pre_pt_root_id",
    roots,
    ts,
    signal,
  );
  const post = await countSide(
    store,
    table,
    "post_pt_root_id",
    roots,
    ts,
    signal,
  );
  for (const c of cands) {
    const r = String(c.target_segment_id);
    c.cand_synapses_total = (pre.get(r) ?? 0) + (post.get(r) ?? 0);
  }
  result.cand_synapses_source = {
    table,
    datastack: result.dataset,
    as_of_utc: ts,
    rule: "pre + post rows on the candidate root at the result's timestamp",
    script: "in-browser (twig_inference/synapses.ts)",
  };
}
