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

import { describe, expect, it } from "vitest";
import { LARGE_MERGE_L2 } from "#src/layer/segmentation/twig_inference/datasets.js";
import {
  isLargeMerge,
  mergeOrder,
} from "#src/layer/segmentation/twig_merge_order.js";

const ids = (rows: { target_segment_id: string }[]) =>
  rows.map((r) => r.target_segment_id);

describe("mergeOrder", () => {
  it("puts a level-2 row after the suggestion it attaches to", () => {
    // List order (by p) had the child first: it was skipped in a staged run.
    const rows = [
      { target_segment_id: "child", hop: 2, via_root: "parent" },
      { target_segment_id: "a", hop: 1 },
      { target_segment_id: "parent", hop: 1 },
    ];
    expect(ids(mergeOrder(rows))).toEqual(["a", "parent", "child"]);
  });
  it("orders orphan-first chains by depth and parent_root", () => {
    const rows = [
      { target_segment_id: "c", depth: 2, parent_root: "b" },
      { target_segment_id: "b", depth: 1, parent_root: "seed" },
      { target_segment_id: "x", depth: 1, parent_root: "seed" },
    ];
    expect(ids(mergeOrder(rows))).toEqual(["b", "x", "c"]);
  });
  it("keeps a child after its parent even when levels disagree", () => {
    const rows = [
      { target_segment_id: "p", hop: 3 },
      { target_segment_id: "k", hop: 2, via_root: "p" },
    ];
    expect(ids(mergeOrder(rows))).toEqual(["p", "k"]);
  });
  it("keeps list order otherwise, and survives a cycle", () => {
    const rows = [
      { target_segment_id: "a", via_root: "b" },
      { target_segment_id: "b", via_root: "a" },
      { target_segment_id: "c" },
    ];
    const out = ids(mergeOrder(rows));
    expect(out.sort()).toEqual(["a", "b", "c"]);
  });
});

describe("isLargeMerge", () => {
  it("never treats an unknown size as small", () => {
    expect(isLargeMerge({ num_l2_nodes: null, l2_unknown: true })).toBe(true);
    expect(isLargeMerge({ num_l2_nodes: null })).toBe(true);
    expect(isLargeMerge({ num_l2_nodes: 0 })).toBe(false);
    expect(isLargeMerge({ num_l2_nodes: LARGE_MERGE_L2 + 1 })).toBe(true);
    expect(isLargeMerge({ num_l2_nodes: LARGE_MERGE_L2 })).toBe(false);
  });
});
