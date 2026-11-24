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
import { autoPlan } from "#src/layer/segmentation/twig_selector_tab.js";

describe("autoPlan", () => {
  it("lets a fragment find a big partner, and searches all of it", () => {
    const p = autoPlan({ l2: 22, regions: 16 }, true);
    expect(p.maxCandidateL2).toBeGreaterThanOrEqual(1_000_000);
    expect(p.maxRegions).toBe(16);
    expect(p.maxSites).toBe(8);
  });
  it("never limits partner size, and caps a big neuron's regions", () => {
    const p = autoPlan({ l2: 1764, regions: 900 }, true);
    expect(p.maxCandidateL2).toBeGreaterThanOrEqual(1_000_000);
    expect(p.maxRegions).toBe(300);
  });
  it("cuts work hard on CPU", () => {
    const p = autoPlan({ l2: 300, regions: 150 }, false);
    expect(p.maxRegions).toBe(3);
    expect(p.maxSites).toBe(1);
  });
});
