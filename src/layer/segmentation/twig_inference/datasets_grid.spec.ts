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
import {
  TWIG_DATASETS,
  gridOf,
} from "#src/layer/segmentation/twig_inference/datasets.js";
import { binL2 } from "#src/layer/segmentation/twig_inference/pipeline.js";

const byKey = (k: string) => TWIG_DATASETS.find((d) => d.key === k)!;

describe("gridOf", () => {
  it("keeps the 32 nm / 8 nm grid for BANC, Aedes and CRANTb", () => {
    for (const k of ["banc", "aedes", "crantb"]) {
      const g = gridOf(byKey(k));
      expect(g.svNm).toBe(32);
      expect(g.binNm).toBe(16);
      expect(g.emPerSv).toBe(4);
    }
    expect(gridOf(byKey("crantb")).zRes).toBe(42);
  });
  it("uses 34.4 / 8.6 nm for FANC", () => {
    const g = gridOf(byKey("fanc"));
    expect(g.svNm).toBeCloseTo(34.4);
    expect(g.binNm).toBeCloseTo(17.2);
    expect(g.emPerSv).toBe(4);
    expect(g.zRes).toBe(45);
  });
  it("refuses EM that is not 4x finer than the supervoxels", () => {
    expect(() =>
      gridOf({ supervoxelResolution: [32, 32, 45], emResolution: [4, 4, 45] }),
    ).toThrow(/must be 1\/4/);
  });
});

describe("binL2", () => {
  const attrs = {
    a: { rep_coord_nm: [16 * 100 + 5, 16 * 200 + 5, 45 * 30] },
    b: { rep_coord_nm: [17.2 * 100 + 5, 17.2 * 200 + 5, 45 * 30] },
  };
  it("bins BANC coordinates in 16 nm units, as before", () => {
    expect(binL2(["a"], attrs, [32, 32, 45], false)).toEqual([[100, 200, 30]]);
  });
  it("bins FANC coordinates in 17.2 nm units", () => {
    expect(binL2(["b"], attrs, [34.4, 34.4, 45], false)).toEqual([
      [100, 200, 30],
    ]);
  });
});
