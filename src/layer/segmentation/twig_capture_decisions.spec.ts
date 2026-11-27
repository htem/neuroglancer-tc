import { describe, expect, it } from "vitest";
import {
  alignedVolume,
  decisionsAllowed,
} from "#src/layer/segmentation/twig_capture_decisions.js";

describe("twig_capture_decisions", () => {
  it("allows decisions on FANC now that its table exists", () => {
    expect(
      decisionsAllowed([
        "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/mar2021_prod",
      ]),
    ).toBe(true);
  });

  it("refuses a volume that is not on the allowlist", () => {
    expect(
      decisionsAllowed([
        "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/some_other_table",
      ]),
    ).toBe(false);
  });

  it("posts FANC annotations under its aligned volume, fanc_v4", () => {
    expect(alignedVolume("fanc_production_mar2021")).toBe("fanc_v4");
    expect(alignedVolume("brain_and_nerve_cord")).toBe("brain_and_nerve_cord");
    expect(alignedVolume("kronauer_ant")).toBe("kronauer_ant");
  });
});
