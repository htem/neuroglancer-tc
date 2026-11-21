/**
 * neurons.json: the SeaTable annotations behind the tab's neuron filters.
 */
import { describe, expect, it } from "vitest";
import {
  BLANK,
  fieldOptions,
  hasNeuronFilters,
  neuronLabel,
  parseNeuronMeta,
  passesNeuronFilters,
} from "#src/layer/segmentation/twig_capture_neurons.js";

const META = parseNeuronMeta({
  version: 1,
  fields: ["cell_type", "super_class", "cell_class", "status", "flow", "bogus"],
  neurons: {
    "1": {
      super_class: "descending",
      cell_type: "DNg30",
      status: ["BACKBONE_PROOFREAD"],
      flow: "efferent",
    },
    "2": {
      super_class: "descending",
      status: ["PARTIALLY_PROOFREAD", "TRACING_ISSUE"],
    },
    "3": { super_class: "kenyon_cell", cell_class: "clawed_kenyon_cell" },
    "4": { super_class: "kenyon_cell", status: "adequate" },
  },
});

describe("parseNeuronMeta", () => {
  it("keeps known fields only", () => {
    expect(META.fields).toEqual([
      "cell_type",
      "super_class",
      "cell_class",
      "status",
      "flow",
    ]);
  });
  it("reads a single-valued status (Aedes) as a list", () => {
    expect(META.neurons.get("4")!.status).toEqual(["adequate"]);
  });
  it("rejects other versions", () => {
    expect(() => parseNeuronMeta({ version: 2, neurons: {} })).toThrow();
  });
});

describe("passesNeuronFilters", () => {
  const a = (r: string) => META.neurons.get(r);
  it("passes everything with no filters", () => {
    expect(passesNeuronFilters(a("1"), {})).toBe(true);
    expect(passesNeuronFilters(undefined, {})).toBe(true);
  });
  it("matches single fields exactly", () => {
    expect(passesNeuronFilters(a("1"), { super_class: "descending" })).toBe(
      true,
    );
    expect(passesNeuronFilters(a("3"), { super_class: "descending" })).toBe(
      false,
    );
  });
  it("matches any one of a multi-valued status", () => {
    expect(passesNeuronFilters(a("2"), { status: "TRACING_ISSUE" })).toBe(true);
    expect(passesNeuronFilters(a("1"), { status: "TRACING_ISSUE" })).toBe(
      false,
    );
  });
  it("ANDs fields together", () => {
    const f = { super_class: "descending", status: "BACKBONE_PROOFREAD" };
    expect(passesNeuronFilters(a("1"), f)).toBe(true);
    expect(passesNeuronFilters(a("2"), f)).toBe(false);
  });
  it("BLANK selects neurons without the field, and unannotated ones", () => {
    expect(passesNeuronFilters(a("2"), { cell_type: BLANK })).toBe(true);
    expect(passesNeuronFilters(a("1"), { cell_type: BLANK })).toBe(false);
    expect(passesNeuronFilters(undefined, { cell_type: BLANK })).toBe(true);
    expect(passesNeuronFilters(undefined, { cell_type: "DNg30" })).toBe(false);
  });
});

describe("fieldOptions", () => {
  const roots = ["1", "2", "3", "4", "5"];
  it("counts values, most common first, BLANK last", () => {
    expect(fieldOptions(META, "super_class", roots, {})).toEqual([
      ["descending", 2],
      ["kenyon_cell", 2],
      [BLANK, 1],
    ]);
  });
  it("counts under the other fields' filters, not its own", () => {
    const f = { super_class: "descending" };
    expect(fieldOptions(META, "status", roots, f)).toEqual([
      ["BACKBONE_PROOFREAD", 1],
      ["PARTIALLY_PROOFREAD", 1],
      ["TRACING_ISSUE", 1],
    ]);
    expect(fieldOptions(META, "super_class", roots, f).length).toBe(3);
  });
});

describe("helpers", () => {
  it("labels by type, else class, else super class", () => {
    expect(neuronLabel(META.neurons.get("1"))).toBe("DNg30");
    expect(neuronLabel(META.neurons.get("3"))).toBe("clawed_kenyon_cell");
    expect(neuronLabel(META.neurons.get("2"))).toBe("descending");
    expect(neuronLabel(undefined)).toBe("");
  });
  it("knows when any filter is set", () => {
    expect(hasNeuronFilters({})).toBe(false);
    expect(hasNeuronFilters({ flow: "" })).toBe(false);
    expect(hasNeuronFilters({ flow: "efferent" })).toBe(true);
  });
});
