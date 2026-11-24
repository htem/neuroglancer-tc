/**
 * @license
 * Copyright 2026 Alex Bates
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
 * MVB review: write multivesicular-body calls to a CAVE annotation table.
 *
 * Modelled on twig_capture_decisions.ts (same sign-in, same error handling,
 * same append-only rule).
 *
 * THE TABLE
 * ---------
 * multi_vesicular_bodies on BANC (datastack brain_and_nerve_cord), schema
 * `bound_tag`, voxel_resolution [4, 4, 45] nm:
 *
 *   pt   bound point INSIDE the scored object, in 4 x 4 x 45 nm voxels
 *   tag  mvb | not_mvb
 *
 * "Unsure" is NOT written (ASB 2026-11-13); the tab remembers it in this
 * browser only. The point binds to the BANC segmentation, so pt_root_id is
 * the neuron the object sits in, not the mitochondria-model object.
 *
 * MATCHING. The table has no field naming the model object, so a row is
 * matched back to a queue item by its position. The queue fixes one point per
 * object and the tab writes exactly that point, so the match is exact.
 *
 * APPEND-ONLY. An object scored twice has two rows; the later `created` wins,
 * ties broken by `id`. Whoever builds ground truth from the table must apply
 * the same rule.
 */

export const MVB_TABLE = "multi_vesicular_bodies";
export const MVB_SERVER = "https://cave.fanc-fly.com";
export const MVB_DATASTACK = "brain_and_nerve_cord";
/** The table's voxel size; queue points and posted positions use it. */
export const MVB_VOXEL_NM: readonly number[] = [4, 4, 45];

/**
 * Layers on which calls may be written: the BANC mitochondria model, whose
 * objects the queue lists. Checked by the tab (no buttons) AND by the writer
 * (refuses), as in Twig Capture: a call recorded while looking at another
 * volume would be a false label nothing in the row reveals.
 */
export const MVB_ALLOWLIST: readonly string[] = [
  "zetta_lee_fly_cns_001_mito/250423_mito/seg",
];

export function mvbReviewAllowed(urls: readonly string[]): boolean {
  return urls.some((url) => MVB_ALLOWLIST.some((a) => url.includes(a)));
}

export type MvbTag = "mvb" | "not_mvb";

export interface MvbRow {
  id: number;
  tag: string;
  created: number;
  /** pt position, 4 x 4 x 45 nm voxels, as "x,y,z". */
  key: string;
}

export function pointKey(pt: readonly number[]): string {
  return pt.map((v) => Math.round(v)).join(",");
}

/** The useful sentence out of a CAVE error body (see twig_capture_decisions). */
async function caveError(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const parsed = JSON.parse(text);
    if (typeof parsed?.message === "string") return parsed.message;
  } catch {
    // Not JSON; fall through to the raw text.
  }
  return text.slice(0, 200);
}

export class MvbStore {
  private token: { tokenType: string; accessToken: string } | undefined;

  constructor(
    public server = MVB_SERVER,
    public datastack = MVB_DATASTACK,
  ) {}

  /**
   * The middleauth token this page already holds (loading the BANC graphene
   * segmentation layer signs the reviewer in). Never opens a login popup.
   */
  private async auth(): Promise<{ tokenType: string; accessToken: string }> {
    if (this.token !== undefined) return this.token;
    const info = await fetch(`${this.server}/auth_info`).then((r) => r.json());
    const raw = localStorage.getItem(`auth_token_v2_${info.login_url}`);
    if (raw === null) {
      throw new Error(
        `no CAVE credentials for ${this.server} in this browser. Load the ` +
          "BANC segmentation layer first, which signs you in.",
      );
    }
    const t = JSON.parse(raw);
    if (Array.isArray(t.appUrls) && !t.appUrls.includes(this.server)) {
      throw new Error(
        `your CAVE token is not valid for ${this.server} (not in appUrls).`,
      );
    }
    this.token = { tokenType: t.tokenType, accessToken: t.accessToken };
    return this.token;
  }

  private async headers(): Promise<Record<string, string>> {
    const t = await this.auth();
    return {
      Authorization: `${t.tokenType} ${t.accessToken}`,
      "Content-Type": "application/json",
    };
  }

  /** Post one call. `pt` is in 4 x 4 x 45 nm voxels. Returns the row id. */
  async post(pt: readonly number[], tag: MvbTag): Promise<number> {
    if (tag !== "mvb" && tag !== "not_mvb") {
      throw new Error(`refusing to write tag "${tag}"`);
    }
    const body = {
      annotations: [{ pt: { position: pt.map((v) => Math.round(v)) }, tag }],
    };
    const url =
      `${this.server}/annotation/api/v2/aligned_volume/${this.datastack}` +
      `/table/${MVB_TABLE}/annotations`;
    const response = await fetch(url, {
      method: "POST",
      headers: await this.headers(),
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(
        `CAVE refused the call (${response.status}): ` +
          `${await caveError(response)}`,
      );
    }
    const out = await response.json();
    return Array.isArray(out) ? Number(out[0]) : Number(out);
  }

  /**
   * Every call in the table, keyed by point, latest row winning. Live query
   * with allow_missing_lookups, so another reviewer's call a minute ago is
   * visible (see twig_capture_decisions.fetchForNeuron).
   */
  async fetchAll(): Promise<Map<string, MvbRow>> {
    const url =
      `${this.server}/materialize/api/v3/datastack/${this.datastack}/query` +
      "?return_pyarrow=false&arrow_format=false&merge_reference=false" +
      "&allow_missing_lookups=true&direct_sql_pandas=true";
    const body = JSON.stringify({
      table: MVB_TABLE,
      timestamp: new Date().toISOString(),
      select_columns: { [MVB_TABLE]: ["id", "created", "tag", "pt_position"] },
    });
    const response = await fetch(url, {
      method: "POST",
      headers: await this.headers(),
      body,
    });
    if (!response.ok) {
      const detail = await caveError(response);
      // A table nobody has written to yet has no segmentation table, so the
      // live query fails with "relation ... does not exist": no calls yet.
      if (/does not exist/i.test(detail)) return new Map();
      throw new Error(`CAVE query failed (${response.status}): ${detail}`);
    }
    const rows = await response.json();
    const out = new Map<string, MvbRow>();
    for (const r of rows as any[]) {
      // The live query returns the point either as one array or split into
      // pt_position_x/_y/_z (its SQL casts each coordinate separately).
      const pos = Array.isArray(r.pt_position)
        ? r.pt_position
        : [r.pt_position_x, r.pt_position_y, r.pt_position_z];
      if (
        pos.length !== 3 ||
        pos.some((v: unknown) => !Number.isFinite(Number(v)))
      )
        continue;
      const row: MvbRow = {
        id: Number(r.id),
        tag: String(r.tag ?? ""),
        created: Number(new Date(r.created ?? 0)),
        key: pointKey(pos.map(Number)),
      };
      const prev = out.get(row.key);
      const newer =
        prev === undefined ||
        row.created > prev.created ||
        (row.created === prev.created && row.id > prev.id);
      if (newer) out.set(row.key, row);
    }
    return out;
  }
}
