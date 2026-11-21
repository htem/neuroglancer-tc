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
 * Neuron annotations for the catalog, so a reviewer can pick neurons by what
 * they are (cell type, class, proofreading status ...) rather than by root id.
 *
 * They come from each project's SeaTable, but never from the browser: a
 * SeaTable token in the page would be a token handed to every reviewer.
 * scripts/make_seatable_meta.py reads the table on the cluster and writes
 * neurons.json beside index.json, holding only these fields and only for the
 * catalog's neurons, keyed by the catalog's root ids:
 *
 *   { version: 1, generated, dataset, source, fields: [...],
 *     neurons: { <root>: { cell_type?, super_class?, cell_class?,
 *                          status?: [..], flow? } } }
 *
 * The file is optional. Without it the tab has no neuron filters, exactly as
 * before.
 */

export const NEURON_META_FILE = "neurons.json";

/** Filterable fields, in the order the filter row shows them. */
export const NEURON_FIELDS = [
  "super_class",
  "cell_class",
  "cell_type",
  "status",
  "flow",
] as const;
export type NeuronField = (typeof NEURON_FIELDS)[number];

export const NEURON_FIELD_LABELS: Record<NeuronField, string> = {
  super_class: "super class",
  cell_class: "class",
  cell_type: "type",
  status: "status",
  flow: "flow",
};

/** The filter value that selects neurons with nothing in that field. */
export const BLANK = "(blank)";

export interface NeuronAnnotation {
  cell_type?: string;
  super_class?: string;
  cell_class?: string;
  /** Multi-valued in SeaTable (CRANTb, BANC); a single value for Aedes. */
  status?: string[];
  flow?: string;
}

export interface NeuronMeta {
  generated?: string;
  source?: string;
  /** Which of NEURON_FIELDS this dataset's table has. */
  fields: NeuronField[];
  neurons: Map<string, NeuronAnnotation>;
}

/** Field -> chosen value. An absent or empty field is "any". */
export type NeuronFilters = Partial<Record<NeuronField, string>>;

export function parseNeuronMeta(obj: unknown): NeuronMeta {
  const t = obj as { [k: string]: any };
  if (t === null || typeof t !== "object" || t.version !== 1) {
    throw new Error("not a version-1 neurons.json");
  }
  const fields = (Array.isArray(t.fields) ? t.fields : []).filter(
    (f: unknown): f is NeuronField =>
      (NEURON_FIELDS as readonly unknown[]).includes(f),
  );
  const neurons = new Map<string, NeuronAnnotation>();
  const raw = t.neurons && typeof t.neurons === "object" ? t.neurons : {};
  for (const [root, a] of Object.entries(raw as { [k: string]: any })) {
    if (a === null || typeof a !== "object") continue;
    const rec: NeuronAnnotation = {};
    for (const f of fields) {
      const v = a[f];
      if (f === "status") {
        const list = Array.isArray(v) ? v : typeof v === "string" ? [v] : [];
        rec.status = list.map(String).filter((s: string) => s !== "");
      } else if (typeof v === "string" && v !== "") {
        rec[f] = v;
      }
    }
    neurons.set(String(root), rec);
  }
  return { generated: t.generated, source: t.source, fields, neurons };
}

/** The catalog's neurons.json, or undefined when it has none. */
export async function fetchNeuronMeta(
  base: string,
  force = false,
): Promise<NeuronMeta | undefined> {
  const response = await fetch(
    base + NEURON_META_FILE,
    force ? { cache: "reload" } : {},
  );
  if (response.status === 404) return undefined;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return parseNeuronMeta(await response.json());
}

/** A neuron's values for one field: [] when it has none. */
function valuesOf(a: NeuronAnnotation | undefined, f: NeuronField): string[] {
  if (a === undefined) return [];
  if (f === "status") return a.status ?? [];
  const v = a[f];
  return v === undefined ? [] : [v];
}

/**
 * Does a neuron pass every chosen filter? A neuron with no annotation at all
 * (not in the table yet) fails any filter except BLANK.
 */
export function passesNeuronFilters(
  a: NeuronAnnotation | undefined,
  filters: NeuronFilters,
): boolean {
  for (const f of NEURON_FIELDS) {
    const want = filters[f];
    if (!want) continue;
    const have = valuesOf(a, f);
    if (want === BLANK ? have.length > 0 : !have.includes(want)) return false;
  }
  return true;
}

export function hasNeuronFilters(filters: NeuronFilters): boolean {
  return NEURON_FIELDS.some((f) => !!filters[f]);
}

/**
 * The choices for one field among `roots`, most common first, with how many
 * neurons have each. Counted under the OTHER fields' filters, so every option
 * offered leads to at least one neuron. BLANK is offered when some have none.
 */
export function fieldOptions(
  meta: NeuronMeta,
  field: NeuronField,
  roots: Iterable<string>,
  filters: NeuronFilters,
): [string, number][] {
  const others: NeuronFilters = { ...filters, [field]: undefined };
  const counts = new Map<string, number>();
  let blank = 0;
  for (const root of roots) {
    const a = meta.neurons.get(root);
    if (!passesNeuronFilters(a, others)) continue;
    const vals = valuesOf(a, field);
    if (vals.length === 0) blank++;
    for (const v of vals) counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const out = [...counts.entries()].sort(
    (x, y) => y[1] - x[1] || x[0].localeCompare(y[0]),
  );
  if (blank > 0) out.push([BLANK, blank]);
  return out;
}

/** A short label for a neuron: its type, else class, else super class. */
export function neuronLabel(a: NeuronAnnotation | undefined): string {
  return a?.cell_type ?? a?.cell_class ?? a?.super_class ?? "";
}

/** Every annotation of a neuron, one per line, for a tooltip. */
export function neuronTooltip(a: NeuronAnnotation | undefined): string {
  if (a === undefined) return "not in SeaTable (or not yet matched)";
  return NEURON_FIELDS.map((f) => {
    const v = valuesOf(a, f);
    return v.length ? `${NEURON_FIELD_LABELS[f]}: ${v.join(", ")}` : "";
  })
    .filter((s) => s !== "")
    .join("\n");
}
