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

// Checks the TypeScript port against run_twigcapture_v2.py itself: the
// fixture is produced by twig_capture_inference/scripts/gen_volume_ops_fixture.py,
// which execs the runner's own touching_labels_center, find_contact_sites and
// supervoxel_near (scipy) on random label volumes.

import { describe, expect, it } from "vitest";
import type { Shape } from "#src/layer/segmentation/twig_inference/volume_ops.js";
import {
  findContactSites,
  touchingLabelsCenter,
  voxelNear,
  zoomSourceIndex,
} from "#src/layer/segmentation/twig_inference/volume_ops.js";
import fixture from "#src/layer/segmentation/twig_inference/volume_ops_fixture.json" with { type: "json" };

describe("twig volume ops match run_twigcapture_v2.py", () => {
  (fixture as any[]).forEach((c, n) => {
    const s = c.shape as Shape;
    const lab = Uint32Array.from(c.lab as number[]);
    it(`case ${n}: touching_labels_center`, () => {
      const { keep, rejects } = touchingLabelsCenter(lab, s, c.target, {
        border: 5,
        minContact: 1,
        maxContact: 1500,
        minSegment: 22,
      });
      expect(
        Object.fromEntries([...keep].map(([k, v]) => [String(k), v])),
      ).toEqual(c.keep);
      expect(
        Object.fromEntries([...rejects].map(([k, v]) => [String(k), v])),
      ).toEqual(c.rejects);
    });
    it(`case ${n}: find_contact_sites and supervoxel_near`, () => {
      for (const [k, sites] of Object.entries(
        c.sites as Record<string, any[]>,
      )) {
        const got = findContactSites(lab, s, c.target, Number(k), 10, 1000);
        expect(got.map(([v]) => v)).toEqual(sites.map(([v]) => v));
        // numpy's argsort is not stable beyond 16 elements, so blobs of
        // EQUAL size may come out in another order; compare them as a set.
        const key = (size: number, ctr: number[]) =>
          `${size}:${ctr.map((x) => x.toFixed(6)).join(",")}`;
        expect(got.map(([v, ctr]) => key(v, ctr)).sort()).toEqual(
          sites.map(([v, ctr]: [number, number[]]) => key(v, ctr)).sort(),
        );
        if (sites.length) {
          const pt = voxelNear(lab, s, sites[0][1].map(Math.trunc), c.target);
          expect(pt ?? null).toEqual(c.near[k]);
        }
      }
    });
  });
  it("zoom(order=0) source indices", () => {
    const want = (fixture as any[])[0].mask_x_profile as number[];
    expect(
      Array.from({ length: 128 }, (_, o) => zoomSourceIndex(o, 32, 128)),
    ).toEqual(want);
  });
});
