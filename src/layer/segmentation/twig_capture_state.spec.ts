/**
 * TwigCaptureState: links made before the filter facets must keep working.
 *
 * mito replaced a pair of hideMito/mitoOnly booleans on 2026-10-06. Links
 * carrying the old keys are already pasted into chats and tickets, and a
 * state restore that silently drops them would change what a shared link
 * shows without saying so -- the reader would be looking at a different set
 * of candidates from the sender and neither would know.
 */
import { describe, expect, it } from "vitest";
import { TwigCaptureState } from "#src/layer/segmentation/twig_capture_tab.js";

function restored(obj: unknown) {
  const s = new TwigCaptureState();
  s.restoreState(obj);
  return s;
}

describe("TwigCaptureState legacy links", () => {
  it("maps hideMito to mito=hide", () => {
    expect(restored({ hideMito: true }).mito).toBe("hide");
  });

  it("maps mitoOnly to mito=only", () => {
    expect(restored({ mitoOnly: true }).mito).toBe("only");
  });

  it("leaves mito=any when neither is set", () => {
    expect(restored({ minProb: 0.9 }).mito).toBe("any");
  });

  it("keeps the other legacy fields", () => {
    const s = restored({
      catalogUrl: "https://example.org/x/",
      file: "648518347601362272_20261002_164317.json",
      candidate: "648518347517868507",
      minProb: 0.5,
      hideMito: true,
      sortKey: "synapses",
    });
    expect(s.catalogUrl).toBe("https://example.org/x/");
    expect(s.file).toBe("648518347601362272_20261002_164317.json");
    expect(s.candidate).toBe("648518347517868507");
    expect(s.minProb).toBe(0.5);
    expect(s.sortKey).toBe("synapses");
    expect(s.mito).toBe("hide");
  });

  it("defaults review to any for a link that predates it", () => {
    expect(restored({ hideMito: true }).review).toBe("any");
  });

  it("prefers the new key when both somehow appear", () => {
    expect(restored({ hideMito: true, mito: "only" }).mito).toBe("only");
  });

  it("ignores junk rather than throwing", () => {
    // A restoreState that throws takes the whole viewer down, not one tab.
    expect(restored({ mito: "nonsense", review: 7 }).mito).toBe("any");
    expect(restored({ mito: "nonsense", review: 7 }).review).toBe("any");
    expect(() => restored(null)).not.toThrow();
    expect(() => restored("a string")).not.toThrow();
  });

  it("round-trips the new facets through toJSON", () => {
    const s = new TwigCaptureState();
    s.mito = "only";
    s.review = "unsure";
    const json = s.toJSON()!;
    expect(json.mito).toBe("only");
    expect(json.review).toBe("unsure");
    expect(restored(json).mito).toBe("only");
    expect(restored(json).review).toBe("unsure");
  });

  it("emits nothing for an untouched tab", () => {
    expect(new TwigCaptureState().toJSON()).toBeUndefined();
  });
});
