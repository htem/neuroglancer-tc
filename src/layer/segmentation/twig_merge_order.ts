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
 * The order in which staged suggestions of one neuron are merged.
 *
 * A level-2 suggestion (in-browser `via_root`, orphan-first `parent_root`)
 * attaches to an earlier suggestion, not to the neuron: until that parent is
 * merged its contact is not on the neuron, and the run skips it. So rows go
 * in order of level (`depth`, else `hop`, else 1), list order within a
 * level, and every row after the row it attaches to whenever that row is in
 * the same batch.
 */

import { LARGE_MERGE_L2 } from "#src/layer/segmentation/twig_inference/datasets.js";

export interface MergeOrderRow {
  target_segment_id: string;
  hop?: number;
  depth?: number;
  via_root?: string;
  parent_root?: string;
}

export function mergeOrder<T extends MergeOrderRow>(rows: readonly T[]): T[] {
  const level = (r: T) => r.depth ?? r.hop ?? 1;
  const sorted = rows
    .map((r, i) => [r, i] as const)
    .sort((a, b) => level(a[0]) - level(b[0]) || a[1] - b[1])
    .map(([r]) => r);
  const byId = new Map<string, T>();
  for (const r of rows)
    if (!byId.has(r.target_segment_id)) byId.set(r.target_segment_id, r);
  const out: T[] = [];
  const done = new Set<T>();
  const visiting = new Set<T>();
  const emit = (r: T) => {
    if (done.has(r) || visiting.has(r)) return;
    visiting.add(r);
    const parentId = r.via_root ?? r.parent_root;
    const parent = parentId === undefined ? undefined : byId.get(parentId);
    if (parent !== undefined && parent !== r) emit(parent);
    visiting.delete(r);
    done.add(r);
    out.push(r);
  };
  for (const r of sorted) emit(r);
  return out;
}

/**
 * A partner above LARGE_MERGE_L2 L2 nodes, or whose size is unknown (the
 * in-browser lookup failed: num_l2_nodes null, l2_unknown): never treated as
 * small.
 */
export function isLargeMerge(c: {
  num_l2_nodes?: number | null;
  l2_unknown?: boolean;
}): boolean {
  if (c.l2_unknown === true || c.num_l2_nodes === null) return true;
  return (c.num_l2_nodes ?? 0) > LARGE_MERGE_L2;
}
