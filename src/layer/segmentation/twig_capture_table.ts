/**
 * @license
 * Copyright 2026 Kevin Delgado
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

/**
 * The catalog-wide candidate table behind the tab's ALL-NEURONS view.
 *
 * A catalog can hold thousands of result files and gigabytes of JSON (BANC:
 * 3,745 files, 9 GB, 1.56M candidates), which no browser can load to list
 * "every candidate above p". scripts/make_results_index.py therefore writes
 * candidates.json.gz beside index.json: one compact row per candidate of the
 * LATEST run of each root, with only what the list shows, filters and sorts
 * on. Merge endpoints and synapse windows are not in it -- the tab fetches a
 * neuron's own result file when one of its rows is acted on.
 *
 * Format (version 1), columnar so it gzips well:
 *   { version, generated, n, files: [{file, root_id, timestamp, dataset}],
 *     f, id, seg, p, l2, size, contact, mito, syn, xyz }
 * where every array has length n and `f` indexes `files`. Ids are strings:
 * they exceed 2^53.
 */

export const CANDIDATE_TABLE_FILE = "candidates.json.gz";

export interface TableFile {
  file: string;
  root_id: string;
  timestamp?: string;
  dataset?: string;
}

/**
 * One candidate from the table. Field names match the result-file candidate,
 * so a row can be listed by the same code; `_file` and `_root` say which
 * neuron's file it came from.
 */
export interface TableCandidate {
  id: string;
  target_segment_id: string;
  prob_merge: number;
  num_l2_nodes: number;
  cand_size_um3: number | null;
  contact_voxels: number | null;
  likely_mito: boolean;
  cand_synapses_total: number | null;
  coordinate_nm: [number, number, number];
  _file: string;
  _root: string;
  /** Not a full result-file record: act on it via the neuron's file. */
  _lite: true;
}

export interface CandidateTable {
  generated?: string;
  files: TableFile[];
  rows: TableCandidate[];
}

/** gzip magic: 1f 8b. */
function isGzip(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

/**
 * Bytes -> parsed JSON, decompressing if they are still gzip.
 *
 * Checked by content, not by headers: a server that sends the .gz as an
 * opaque file (radagast: application/octet-stream) leaves it compressed, while
 * one that adds Content-Encoding: gzip has the browser decompress it already.
 */
export async function decodeTableBytes(bytes: Uint8Array): Promise<unknown> {
  let text: string;
  if (isGzip(bytes)) {
    // Response rather than Blob.stream(): both work in browsers, and this one
    // also in the test environment (jsdom's Blob has no stream()).
    const stream = new Response(bytes as BodyInit).body!.pipeThrough(
      new DecompressionStream("gzip"),
    );
    text = await new Response(stream).text();
  } else {
    text = new TextDecoder().decode(bytes);
  }
  return JSON.parse(text);
}

/** The table's parallel arrays -> one object per candidate. */
export function rowsFromTable(obj: unknown): CandidateTable {
  const t = obj as { [k: string]: any };
  if (t === null || typeof t !== "object" || !Array.isArray(t.files)) {
    throw new Error("not a candidates table (no files array)");
  }
  if (t.version !== 1) {
    throw new Error(`unsupported candidates table version ${t.version}`);
  }
  const files: TableFile[] = t.files.map((f: any) => ({
    file: String(f.file),
    root_id: String(f.root_id),
    timestamp: f.timestamp ?? undefined,
    dataset: f.dataset ?? undefined,
  }));
  const n = Array.isArray(t.f) ? t.f.length : 0;
  for (const k of ["id", "seg", "p", "l2", "xyz"]) {
    if (!Array.isArray(t[k]) || t[k].length !== n) {
      throw new Error(`candidates table column ${k} is missing or short`);
    }
  }
  const col = (k: string, i: number) =>
    Array.isArray(t[k]) ? (t[k][i] ?? null) : null;
  const rows: TableCandidate[] = new Array(n);
  for (let i = 0; i < n; ++i) {
    const file = files[t.f[i]];
    const xyz = t.xyz[i];
    rows[i] = {
      id: String(t.id[i]),
      target_segment_id: String(t.seg[i]),
      prob_merge: Number(t.p[i] ?? 0),
      num_l2_nodes: Number(t.l2[i] ?? 0),
      cand_size_um3: col("size", i),
      contact_voxels: col("contact", i),
      likely_mito: col("mito", i) === 1,
      cand_synapses_total: col("syn", i),
      coordinate_nm: Array.isArray(xyz) ? [xyz[0], xyz[1], xyz[2]] : [0, 0, 0],
      _file: file?.file ?? "",
      _root: file?.root_id ?? "",
      _lite: true,
    };
  }
  return { generated: t.generated, files, rows };
}

/**
 * Fetch a catalog's candidate table, or undefined when it has none (an older
 * catalog, or one indexed with --no-candidates).
 */
export async function fetchCandidateTable(
  base: string,
  force = false,
): Promise<CandidateTable | undefined> {
  const response = await fetch(
    base + CANDIDATE_TABLE_FILE,
    force ? { cache: "reload" } : {},
  );
  if (response.status === 404) return undefined;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  return rowsFromTable(await decodeTableBytes(bytes));
}

/** The probability filter: `p >= threshold` or `p <= threshold`. */
export function passesProb(
  p: number,
  op: ">=" | "<=",
  threshold: number,
): boolean {
  return op === "<=" ? p <= threshold : p >= threshold;
}
