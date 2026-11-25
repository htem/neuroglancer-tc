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
// supervoxel_near (scipy) on random label volumes, at contact gap 0 and 1
// (the runner's contact_gap_voxels, from its gap-merges branch).

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
    it(`case ${n} (contact gap ${c.gap}): touching_labels_center`, () => {
      const { keep, rejects } = touchingLabelsCenter(lab, s, c.target, {
        border: 5,
        minContact: 1,
        maxContact: 1500,
        minSegment: 22,
        gap: c.gap,
      });
      expect(
        Object.fromEntries([...keep].map(([k, v]) => [String(k), v])),
      ).toEqual(c.keep);
      expect(
        Object.fromEntries([...rejects].map(([k, v]) => [String(k), v])),
      ).toEqual(c.rejects);
    });
    it(`case ${n} (contact gap ${c.gap}): find_contact_sites and supervoxel_near`, () => {
      for (const [k, sites] of Object.entries(
        c.sites as Record<string, any[]>,
      )) {
        const got = findContactSites(
          lab,
          s,
          c.target,
          Number(k),
          10,
          1000,
          c.gap,
        );
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
  // twigcapture scripts/gap/tests/test_contact_gap.py, ported: two slabs a
  // voxel apart are invisible at contact gap 0, found at 1 and flagged
  // gap_contact, while a piece that really touches is not flagged.
  it("contact gap: a piece one voxel away is found at 1 and flagged", () => {
    const TARGET = 1;
    const GAPPED = 2;
    const TOUCHING = 3;
    const s: Shape = [60, 60, 40];
    const lab = new Uint32Array(60 * 60 * 40);
    const fill = (
      x0: number,
      x1: number,
      y0: number,
      y1: number,
      l: number,
    ) => {
      for (let z = 10; z < 30; ++z)
        for (let y = y0; y < y1; ++y)
          for (let x = x0; x < x1; ++x) lab[x + 60 * (y + 60 * z)] = l;
    };
    fill(10, 30, 10, 50, TARGET); // the neuron
    fill(31, 45, 10, 30, GAPPED); // one empty voxel layer away (x = 30)
    fill(10, 30, 50, 58, TOUCHING); // face-adjacent at y = 50
    const run = (gap: number) => {
      const { keep } = touchingLabelsCenter(lab, s, TARGET, {
        border: 2,
        minContact: 1,
        maxContact: 1e6,
        minSegment: 0,
        gap,
      });
      return {
        keep,
        sites: findContactSites(lab, s, TARGET, GAPPED, 1, 1, gap),
      };
    };
    const g0 = run(0);
    expect([...g0.keep.keys()]).toEqual([TOUCHING]);
    expect(g0.sites).toEqual([]);
    expect(g0.keep.get(TOUCHING)!.gap_contact).toBe(false);
    const g1 = run(1);
    expect([...g1.keep.keys()].sort()).toEqual([GAPPED, TOUCHING]);
    expect(g1.keep.get(GAPPED)!.gap_contact).toBe(true);
    expect(g1.keep.get(TOUCHING)!.gap_contact).toBe(false);
    expect(g1.sites.length).toBe(1);
    // The site sits at the gap, between the two pieces (x around 29-31).
    const x = g1.sites[0][1][0];
    expect(x).toBeGreaterThanOrEqual(28);
    expect(x).toBeLessThanOrEqual(32);
  });

  // "nontouching" (the Selector's default): the gapped piece is found and
  // flagged as before, but a touching piece keeps its gap-0 contact count.
  it("contact gap, nontouching: a touching piece is counted as at gap 0", () => {
    const s: Shape = [60, 60, 40];
    const lab = new Uint32Array(60 * 60 * 40);
    const fill = (
      x0: number,
      x1: number,
      y0: number,
      y1: number,
      l: number,
    ) => {
      for (let z = 10; z < 30; ++z)
        for (let y = y0; y < y1; ++y)
          for (let x = x0; x < x1; ++x) lab[x + 60 * (y + 60 * z)] = l;
    };
    fill(10, 30, 10, 50, 1);
    fill(31, 45, 10, 30, 2);
    fill(10, 30, 50, 58, 3);
    const opts = { border: 2, minContact: 1, maxContact: 1e6, minSegment: 0 };
    const g0 = touchingLabelsCenter(lab, s, 1, { ...opts, gap: 0 }).keep;
    const all = touchingLabelsCenter(lab, s, 1, { ...opts, gap: 1 }).keep;
    const nt = touchingLabelsCenter(lab, s, 1, {
      ...opts,
      gap: 1,
      gapMode: "nontouching",
    }).keep;
    expect(all.get(3)!.contact_voxels).toBeGreaterThan(
      g0.get(3)!.contact_voxels,
    );
    expect(nt.get(3)).toEqual(g0.get(3));
    expect(nt.get(2)).toEqual(all.get(2));
    expect(nt.get(2)!.gap_contact).toBe(true);
  });

  it("zoom(order=0) source indices", () => {
    const want = (fixture as any[])[0].mask_x_profile as number[];
    expect(
      Array.from({ length: 128 }, (_, o) => zoomSourceIndex(o, 32, 128)),
    ).toEqual(want);
  });
});
