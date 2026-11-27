import { describe, expect, it } from "vitest";
import {
  IdTable,
  LargeMap,
} from "#src/layer/segmentation/twig_inference/precomputed_reader.js";

describe("LargeMap", () => {
  it("spills into new shards and still finds, updates and clears", () => {
    const m = new LargeMap<bigint, number>(4);
    for (let i = 0; i < 10; ++i) m.set(BigInt(i), i);
    expect(m.size).toBe(10);
    for (let i = 0; i < 10; ++i) expect(m.get(BigInt(i))).toBe(i);
    expect(m.has(11n)).toBe(false);
    expect(m.get(11n)).toBeUndefined();
    m.set(2n, 20); // an entry in an older shard is updated, not duplicated
    expect(m.get(2n)).toBe(20);
    expect(m.size).toBe(10);
    m.clear();
    expect(m.size).toBe(0);
    expect(m.has(2n)).toBe(false);
  });

  it("IdTable keeps index 0 for id 0 and numbers new ids in order", () => {
    const t = new IdTable();
    expect(t.get(0n)).toBe(0);
    expect(t.get(5n)).toBe(1);
    expect(t.get(7n)).toBe(2);
    expect(t.get(5n)).toBe(1);
    expect(t.ids).toEqual([0n, 5n, 7n]);
  });
});
