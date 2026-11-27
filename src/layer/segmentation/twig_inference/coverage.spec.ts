import { describe, expect, it } from "vitest";
import {
  bandHalf,
  COVERAGE_MARGIN,
  coverageCentres,
  keptL2Ids,
  regionCentre,
} from "#src/layer/segmentation/twig_inference/pipeline.js";

describe("coverage of each L2 node's full extent", () => {
  it("search band is +/-80 xy, +/-40 z voxels", () => {
    expect(bandHalf()).toEqual([80, 80, 40]);
  });

  it("region coords (bin units) map to cutout centres", () => {
    expect(regionCentre([201, 400, 33])).toEqual([100, 200, 33]);
  });

  it("adds nothing when every point is inside a band", () => {
    const c: [number, number, number] = [1000, 1000, 500];
    const pts: [number, number, number][] = [
      [1000, 1000, 500],
      [1070, 930, 530],
    ];
    expect(coverageCentres(pts, [c])).toEqual([]);
  });

  it("covers a point 45 sections below the only centre (the Aedes pair)", () => {
    const c: [number, number, number] = [1000, 1000, 500];
    const far: [number, number, number] = [1000, 1000, 545];
    const extra = coverageCentres([far], [c]);
    expect(extra).toEqual([far]);
  });

  it("is greedy and deterministic: one centre covers a cluster", () => {
    const pts: [number, number, number][] = [
      [3010, 0, 0],
      [3000, 0, 0],
      [3005, 5, 1],
    ];
    const a = coverageCentres(pts, []);
    const b = coverageCentres([...pts].reverse(), []);
    expect(a).toEqual([[3000, 0, 0]]);
    expect(b).toEqual(a);
  });

  it("uses a band shrunk by the margin", () => {
    const h = bandHalf()[2] - COVERAGE_MARGIN[2];
    const c: [number, number, number] = [0, 0, 0];
    expect(coverageCentres([[0, 0, h - 1]], [c])).toEqual([]);
    expect(coverageCentres([[0, 0, h]], [c])).toEqual([[0, 0, h]]);
  });

  it("keeps binL2's backbone filter", () => {
    const attrs = {
      a: { rep_coord_nm: [0, 0, 0], max_dt_nm: 50, size_nm3: 1 },
      b: { rep_coord_nm: [0, 0, 0], max_dt_nm: 500, size_nm3: 1 },
      c: { rep_coord_nm: [0, 0, 0], max_dt_nm: 50, size_nm3: 9e8 },
      d: {},
    };
    expect(keptL2Ids(["a", "b", "c", "d"], attrs, true)).toEqual(["a"]);
    expect(keptL2Ids(["a", "b", "c", "d"], attrs, false)).toEqual([
      "a",
      "b",
      "c",
    ]);
  });
});
