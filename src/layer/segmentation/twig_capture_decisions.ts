/**
 * Twig Capture: persist review decisions to a CAVE annotation table.
 *
 * WHY THIS EXISTS
 * ---------------
 * Verdicts used to live in Neuroglancer state JSONs and were scraped
 * afterwards into a parquet. That works for one person reviewing then handing
 * off; with ten reviewers it gives no way to split work, no duplicate
 * detection, and a manual harvest at the end. Writing straight to CAVE gives
 * concurrent writes, rows that survive segmentation edits, and a queryable
 * store.
 *
 * THE TABLE
 * ---------
 * twig_capture_decisions, schema `bound_2tag`, voxel_resolution [1,1,1]:
 *
 *   pt    bound point on the NEURON    position in NANOMETRES
 *   pt2   bound point on the ORPHAN
 *   tag   merge | no_merge | unsure
 *   tag2  src=<result file stem>;cand=<target_segment_id>;ctr=<x,y,z>[;op=<id>]
 *
 * `ctr` is in 8 x 8 x 45 nm VOXELS while pt/pt2 are nm. Different units, on
 * purpose, and the table description says so.
 *
 * POSITION IS THE LOAD-BEARING FIELD. CAVE derives the supervoxel by looking
 * up the segmentation at `position`; a supplied supervoxel_id is not what
 * ends up bound. Verified 2026-10-06: a row posted from this browser with
 * supervoxel ids mangled by float64 still bound the correct supervoxels,
 * because the positions were right.
 *
 * TWO BIGINT TRAPS, both real on this data
 * ----------------------------------------
 * 1. Root ids exceed 2^53, so JSON.parse mangles pt_root_id / pt2_root_id in
 *    query responses. Never match on them. Match on the `cand=` field inside
 *    tag2, which is a string and exact.
 * 2. The same applies on the way out, so a filter on pt_root_id is built by
 *    STRING CONCATENATION into the request body rather than by putting a
 *    number through JSON.stringify.
 *
 * APPEND-ONLY. A candidate reviewed twice gets two rows; the later `created`
 * wins. That is deliberate -- an audit trail of what a reviewer thought and
 * when is worth more than an in-place edit, and it avoids depending on the
 * update endpoint.
 */

const TABLE = "twig_capture_decisions";

/**
 * The ONLY segmentation sources whose decisions may be written to CAVE.
 *
 * Decisions are ground truth. A row posted against the wrong volume is not
 * merely useless, it is a false label in a table someone will later train on,
 * and nothing in the row itself would reveal the mistake -- the supervoxels
 * would resolve, the positions would look plausible, and the datastack name
 * comes from the result file rather than from the layer.
 *
 * So the rule is an allowlist, not a denylist: a volume nobody has vouched
 * for gets no verdict control and is refused by the writer. Add an entry
 * deliberately, after checking that twig_capture_decisions exists on that
 * datastack.
 *
 * Matched as a substring of the layer's data source URL, which for a graphene
 * layer looks like
 *   graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/<table>
 */
export const DECISION_ALLOWLIST: readonly string[] = [
  // BANC -- twig_capture_decisions created 2026-10-06
  "https://cave.fanc-fly.com/segmentation/table/wclee_fly_cns_001",
  // Aedes -- twig_capture_decisions created 2026-10-06
  "https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain",
];

/**
 * May decisions be recorded for a layer with these data source URLs?
 *
 * Checked in TWO places on purpose: the tab hides the verdict control, and
 * the writer refuses the post. Hiding a button is a courtesy, not a
 * safeguard -- a stale render, a keyboard path or a future caller would walk
 * straight past it.
 */
export function decisionsAllowed(urls: readonly string[]): boolean {
  for (const url of urls) {
    for (const allowed of DECISION_ALLOWLIST) {
      if (url.includes(allowed)) return true;
    }
  }
  return false;
}

export type Verdict = "merge" | "no_merge" | "unsure";

export interface DecisionRow {
  id: number;
  tag: string;
  /** The candidate's target_segment_id, parsed out of tag2. */
  cand: string;
  tag2: string;
  created: number;
}

export interface DecisionTarget {
  /** nm position + supervoxel of the neuron side (merge_sink). */
  sink: { supervoxel_id: string; coordinate_nm: readonly number[] };
  /** nm position + supervoxel of the orphan side (merge_source). */
  source: { supervoxel_id: string; coordinate_nm: readonly number[] };
  /** The contact centroid in nm; stored in tag2 as 8x8x45 nm voxels. */
  coordinate_nm: readonly number[];
  /** target_segment_id. */
  cand: string;
  /** The result file stem, <root_id>_<timestamp>. */
  src: string;
}

/** The voxel grid `ctr` is written in -- what you paste into Neuroglancer. */
const CTR_VOXEL_NM = [8, 8, 45];

function ctrVoxels(nm: readonly number[]): string {
  return nm.map((v, i) => Math.floor(v / CTR_VOXEL_NM[i])).join(",");
}

/** Pull `cand=...` out of a tag2 string. Empty when absent. */
export function candOf(tag2: string): string {
  const m = /(?:^|;)cand=([^;]*)/.exec(tag2);
  return m === null ? "" : m[1];
}

/**
 * The useful sentence out of a CAVE error body.
 *
 * These responses carry the server's own Python traceback in a JSON array,
 * so the raw text is thousands of characters of stack frames with the actual
 * complaint at the front. A toast showing the first 200 characters showed
 * mostly `{"code": 500, "traceback": ["Traceback (most recent call last)...`
 * and cut off before anything a reader could act on.
 */
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

export class DecisionStore {
  private token: { tokenType: string; accessToken: string } | undefined;

  /**
   * @param server    CAVE server origin, e.g. https://cave.fanc-fly.com
   * @param datastack e.g. brain_and_nerve_cord
   */
  constructor(
    public server: string,
    public datastack: string,
  ) {}

  /**
   * The middleauth token this page already holds.
   *
   * Mirrors MiddleAuthAppCredentialsProvider: ask the server which auth
   * server it uses, then read the token the credentials provider cached in
   * localStorage under `auth_token_v2_<login_url>`. Note login_url carries a
   * path (".../sticky_auth"), so the origin alone is the wrong key.
   *
   * Deliberately read-only: it never triggers a login popup. If there is no
   * token the caller reports it rather than interrupting a review session --
   * loading any graphene layer will have obtained one already.
   */
  private async auth(): Promise<{ tokenType: string; accessToken: string }> {
    if (this.token !== undefined) return this.token;
    const info = await fetch(`${this.server}/auth_info`).then((r) => r.json());
    const raw = localStorage.getItem(`auth_token_v2_${info.login_url}`);
    if (raw === null) {
      throw new Error(
        `no CAVE credentials for ${this.server} in this browser. Load the ` +
          "segmentation layer first, which signs you in.",
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

  /**
   * Post one decision. Returns the new annotation id.
   *
   * `op` is the chunkedgraph operation id and is set only when the merge was
   * actually executed.
   */
  async post(
    target: DecisionTarget,
    verdict: Verdict,
    op?: string,
  ): Promise<number> {
    const parts = [
      `src=${target.src}`,
      `cand=${target.cand}`,
      `ctr=${ctrVoxels(target.coordinate_nm)}`,
    ];
    if (op !== undefined && op !== "") parts.push(`op=${op}`);

    const body = {
      annotations: [
        {
          pt: { position: Array.from(target.sink.coordinate_nm) },
          pt2: { position: Array.from(target.source.coordinate_nm) },
          tag: verdict,
          tag2: parts.join(";"),
        },
      ],
    };
    const url =
      `${this.server}/annotation/api/v2/aligned_volume/${this.datastack}` +
      `/table/${TABLE}/annotations`;
    const response = await fetch(url, {
      method: "POST",
      headers: await this.headers(),
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(
        `CAVE refused the decision (${response.status}): ` +
          `${await caveError(response)}`,
      );
    }
    const out = await response.json();
    return Array.isArray(out) ? Number(out[0]) : Number(out);
  }

  /**
   * Every decision recorded against one neuron, keyed by candidate id.
   *
   * Uses the live query endpoint with allow_missing_lookups, which reads the
   * annotation database directly rather than a materialized view -- so a
   * verdict another reviewer posted a minute ago is visible. A freshly
   * created table is not in any materialized version at all, and without
   * allow_missing_lookups the segmentation join 500s.
   *
   * When a candidate has been reviewed more than once the LATEST row wins.
   */
  async fetchForNeuron(rootId: string): Promise<Map<string, DecisionRow>> {
    const url =
      `${this.server}/materialize/api/v3/datastack/${this.datastack}/query` +
      "?return_pyarrow=false&arrow_format=false&merge_reference=false" +
      "&allow_missing_lookups=true&direct_sql_pandas=true";

    // Built by hand, not JSON.stringify: rootId is above 2^53 and would be
    // rounded if it ever became a JS number.
    const body =
      `{"table":"${TABLE}",` +
      `"timestamp":"${new Date().toISOString()}",` +
      `"filter_equal_dict":{"${TABLE}":{"pt_root_id":${rootId}}}}`;

    const response = await fetch(url, {
      method: "POST",
      headers: await this.headers(),
      body,
    });
    if (!response.ok) {
      const detail = await caveError(response);
      // A table nobody has written to yet has no SEGMENTATION table --
      // `<table>__<datastack>`, which carries the resolved supervoxel and
      // root columns. CAVE creates it lazily on the first post, so the live
      // query's join fails with `relation ... does not exist` until then.
      //
      // That is not an error worth shouting about: it means "no decisions
      // yet", which is exactly what a fresh datastack should report. Shown
      // as a red toast it looked like the feature was broken on Aedes when
      // it was simply unused.
      if (/does not exist/i.test(detail)) return new Map();
      throw new Error(`CAVE query failed (${response.status}): ${detail}`);
    }
    const rows = await response.json();
    const out = new Map<string, DecisionRow>();
    for (const r of rows as any[]) {
      const tag2 = String(r.tag2 ?? "");
      const cand = candOf(tag2);
      if (cand === "") continue;
      const row: DecisionRow = {
        id: Number(r.id),
        tag: String(r.tag ?? ""),
        cand,
        tag2,
        created: Number(r.created ?? 0),
      };
      // LATEST WINS, and the tie-break is `id`, not row order.
      //
      // `created` is the SERVER's millisecond clock, so it is comparable
      // across reviewers -- a skewed browser clock cannot win a race. But two
      // rows can share a millisecond (a double click, or two reviewers at
      // once), and the order rows come back in is not guaranteed, so
      // comparing on `created` alone would resolve those nondeterministically.
      // `id` is server-assigned and monotonic.
      //
      // WHOEVER BUILDS GROUND TRUTH FROM THIS TABLE MUST APPLY THE SAME RULE.
      // A candidate reviewed twice has two rows, and taking both yields
      // contradictory labels.
      const prev = out.get(cand);
      const newer =
        prev === undefined ||
        row.created > prev.created ||
        (row.created === prev.created && row.id > prev.id);
      if (newer) out.set(cand, row);
    }
    return out;
  }
}
