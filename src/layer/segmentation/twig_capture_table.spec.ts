/**
 * The all-neurons candidate table: decoding (gzip or already decompressed),
 * column -> row conversion, and the p >= / p <= filter.
 */
import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  decodeTableBytes,
  passesProb,
  rowsFromTable,
} from "#src/layer/segmentation/twig_capture_table.js";

const TABLE = {
  version: 1,
  generated: "2026-11-21T00:00:00-05:00",
  n: 3,
  files: [
    {
      file: "576460752730258505_20261120_215309.json",
      root_id: "576460752730258505",
      dataset: "kronauer_ant",
    },
    {
      file: "576460752705000070_20261120_220556.json",
      root_id: "576460752705000070",
      dataset: "kronauer_ant",
    },
  ],
  f: [0, 0, 1],
  id: ["cand_1", "cand_2", "cand_1"],
  seg: ["576460752680626465", "576460752797177154", "576460752752532762"],
  p: [0.1032, 0.9857, 0.5],
  l2: [11, 1, 3],
  size: [0.8183, null, 0.2],
  contact: [12, 300, null],
  mito: [0, 1, 0],
  syn: [182, 0, null],
  xyz: [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
  ],
};

describe("candidate table", () => {
  it("decodes gzip bytes", async () => {
    const bytes = new Uint8Array(gzipSync(JSON.stringify(TABLE)));
    expect(await decodeTableBytes(bytes)).toEqual(TABLE);
  });

  it("decodes bytes the server already decompressed", async () => {
    const bytes = new TextEncoder().encode(JSON.stringify(TABLE));
    expect(await decodeTableBytes(bytes)).toEqual(TABLE);
  });

  it("turns columns into rows tagged with their file and neuron", () => {
    const { files, rows } = rowsFromTable(TABLE);
    expect(files).toHaveLength(2);
    expect(rows).toHaveLength(3);
    expect(rows[0]).toMatchObject({
      id: "cand_1",
      target_segment_id: "576460752680626465",
      prob_merge: 0.1032,
      num_l2_nodes: 11,
      cand_synapses_total: 182,
      likely_mito: false,
      coordinate_nm: [1, 2, 3],
      _file: "576460752730258505_20261120_215309.json",
      _root: "576460752730258505",
      _lite: true,
    });
    expect(rows[1].likely_mito).toBe(true);
    expect(rows[2]._root).toBe("576460752705000070");
    expect(rows[2].cand_synapses_total).toBeNull();
    // ids above 2^53 stay exact
    expect(rows[1].target_segment_id).toBe("576460752797177154");
  });

  it("rejects a table of another version or with short columns", () => {
    expect(() => rowsFromTable({ ...TABLE, version: 2 })).toThrow();
    expect(() => rowsFromTable({ ...TABLE, p: [0.1] })).toThrow();
    expect(() => rowsFromTable(null)).toThrow();
  });

  it("filters p >= and p <= inclusively", () => {
    expect(passesProb(0.5, ">=", 0.5)).toBe(true);
    expect(passesProb(0.49, ">=", 0.5)).toBe(false);
    expect(passesProb(0.5, "<=", 0.5)).toBe(true);
    expect(passesProb(0.51, "<=", 0.5)).toBe(false);
  });
});
