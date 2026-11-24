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
import { spreadRegions } from "#src/layer/segmentation/twig_inference/pipeline.js";

describe("spreadRegions", () => {
  // A long neuron: 100 bins in a line along x, one bin (70 units) apart.
  const line = Array.from(
    { length: 100 },
    (_, i) => [i * 70, 0, 0] as [number, number, number],
  );
  it("returns every bin when under the cap", () => {
    expect(spreadRegions(line, 100)).toEqual(line);
    expect(spreadRegions(line, 500)).toEqual(line);
  });
  it("spreads a capped selection over the whole neuron", () => {
    const xs = spreadRegions(line, 5)
      .map((c) => c[0] / 70)
      .sort((a, b) => a - b);
    expect(xs.length).toBe(5);
    // Both ends are covered, and no gap is more than a quarter of the length.
    expect(xs[0]).toBe(0);
    expect(xs[xs.length - 1]).toBe(99);
    for (let i = 1; i < xs.length; ++i)
      expect(xs[i] - xs[i - 1]).toBeLessThanOrEqual(26);
  });
});
