/**
 * @license
 * Copyright 2026 Google Inc.
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
 * Voxel operations of twigcapture's run_twigcapture_v2.py, ported 1:1.
 *
 * Volumes are flat arrays in (x, y, z) order, x fastest:
 * index = x + sx * (y + sy * z). Labels are small integers (indices into a
 * table of ids built per cutout), 0 = background. All structuring elements
 * are 26-connected (CONNECTIVITY = 3 in the Python) unless stated.
 */

export type Shape = readonly [number, number, number];

const idx = (s: Shape, x: number, y: number, z: number) =>
  x + s[0] * (y + s[1] * z);

/** scipy.ndimage.binary_dilation with the full 3x3x3 cube (separable max). */
export function dilate26(mask: Uint8Array, s: Shape): Uint8Array {
  const [sx, sy, sz] = s;
  const a = new Uint8Array(mask.length);
  // x pass
  for (let z = 0; z < sz; ++z)
    for (let y = 0; y < sy; ++y) {
      const row = sx * (y + sy * z);
      for (let x = 0; x < sx; ++x) {
        const i = row + x;
        a[i] =
          mask[i] | (x > 0 ? mask[i - 1] : 0) | (x < sx - 1 ? mask[i + 1] : 0);
      }
    }
  const b = new Uint8Array(mask.length);
  for (let z = 0; z < sz; ++z)
    for (let y = 0; y < sy; ++y) {
      const row = sx * (y + sy * z);
      for (let x = 0; x < sx; ++x) {
        const i = row + x;
        b[i] = a[i] | (y > 0 ? a[i - sx] : 0) | (y < sy - 1 ? a[i + sx] : 0);
      }
    }
  const plane = sx * sy;
  for (let z = 0; z < sz; ++z)
    for (let i0 = 0; i0 < plane; ++i0) {
      const i = i0 + plane * z;
      a[i] =
        b[i] | (z > 0 ? b[i - plane] : 0) | (z < sz - 1 ? b[i + plane] : 0);
    }
  return a;
}

/** binary_dilation with the 6-connected cross (border value 0). */
export function dilate6(mask: Uint8Array, s: Shape): Uint8Array {
  const [sx, sy, sz] = s;
  const out = new Uint8Array(mask.length);
  const plane = sx * sy;
  for (let z = 0; z < sz; ++z)
    for (let y = 0; y < sy; ++y)
      for (let x = 0; x < sx; ++x) {
        const i = idx(s, x, y, z);
        out[i] =
          mask[i] |
          (x > 0 ? mask[i - 1] : 0) |
          (x < sx - 1 ? mask[i + 1] : 0) |
          (y > 0 ? mask[i - sx] : 0) |
          (y < sy - 1 ? mask[i + sx] : 0) |
          (z > 0 ? mask[i - plane] : 0) |
          (z < sz - 1 ? mask[i + plane] : 0);
      }
  return out;
}

/** binary_erosion with the 6-connected cross (scipy's default border_value=0). */
export function erode6(mask: Uint8Array, s: Shape): Uint8Array {
  const [sx, sy, sz] = s;
  const out = new Uint8Array(mask.length);
  const plane = sx * sy;
  for (let z = 0; z < sz; ++z)
    for (let y = 0; y < sy; ++y)
      for (let x = 0; x < sx; ++x) {
        const i = idx(s, x, y, z);
        out[i] =
          mask[i] &
          (x > 0 ? mask[i - 1] : 0) &
          (x < sx - 1 ? mask[i + 1] : 0) &
          (y > 0 ? mask[i - sx] : 0) &
          (y < sy - 1 ? mask[i + sx] : 0) &
          (z > 0 ? mask[i - plane] : 0) &
          (z < sz - 1 ? mask[i + plane] : 0);
      }
  return out;
}

/** ndimage.label with 26-connectivity. Returns labels (1..n) and n. */
export function label26(
  mask: Uint8Array,
  s: Shape,
): { labels: Int32Array; count: number } {
  const [sx, sy, sz] = s;
  const labels = new Int32Array(mask.length);
  const stack: number[] = [];
  let count = 0;
  for (let start = 0; start < mask.length; ++start) {
    if (!mask[start] || labels[start]) continue;
    ++count;
    labels[start] = count;
    stack.push(start);
    while (stack.length) {
      const i = stack.pop()!;
      const x = i % sx;
      const y = Math.floor(i / sx) % sy;
      const z = Math.floor(i / (sx * sy));
      for (let dz = -1; dz <= 1; ++dz) {
        const zz = z + dz;
        if (zz < 0 || zz >= sz) continue;
        for (let dy = -1; dy <= 1; ++dy) {
          const yy = y + dy;
          if (yy < 0 || yy >= sy) continue;
          for (let dx = -1; dx <= 1; ++dx) {
            const xx = x + dx;
            if (xx < 0 || xx >= sx) continue;
            const j = idx(s, xx, yy, zz);
            if (mask[j] && !labels[j]) {
              labels[j] = count;
              stack.push(j);
            }
          }
        }
      }
    }
  }
  return { labels, count };
}

export interface NeighbourStats {
  contact_voxels: number;
  cand_voxels: number;
  /**
   * Found only because `gap` widened the search: the piece does not touch the
   * target at supervoxel resolution (the runner's gap_contact).
   */
  gap_contact: boolean;
}

/**
 * dilate26 `times` times: scipy's binary_dilation(structure 26-connected,
 * iterations=times). Nothing outside the volume is ever set, as in scipy.
 */
export function dilate26Times(
  mask: Uint8Array,
  s: Shape,
  times: number,
): Uint8Array {
  let out = mask;
  for (let i = 0; i < times; ++i) out = dilate26(out, s);
  return out === mask ? mask.slice() : out;
}
export type RejectReason =
  | "contact_below_min"
  | "contact_above_max"
  | "segment_too_small";

/** touching_labels_center: neighbours of `target` in the centre box. */
export function touchingLabelsCenter(
  lab: Uint32Array,
  s: Shape,
  target: number,
  opts: {
    border: number;
    minContact: number;
    maxContact: number;
    minSegment: number;
    /**
     * contact_gap_voxels (runner `CONTACT_GAP`): pieces up to this many
     * supervoxel voxels apart count as touching. 0 is the old search.
     */
    gap?: number;
    /**
     * "all" (the runner's rule): every neighbour's contact is counted on the
     * widened ring. "nontouching": neighbours that really touch keep the
     * one-voxel ring (so their counts, and with them the contact cap, are
     * exactly as at gap 0) and only the others are counted on the wide one.
     */
    gapMode?: "all" | "nontouching";
  },
): {
  keep: Map<number, NeighbourStats>;
  rejects: Map<number, [RejectReason, number, number]>;
} {
  const keep = new Map<number, NeighbourStats>();
  const rejects = new Map<number, [RejectReason, number, number]>();
  const [sx, sy, sz] = s;
  const xb = Math.min(opts.border, sx >> 1);
  const yb = Math.min(opts.border, sy >> 1);
  const zb = Math.min(opts.border, sz >> 1);
  const inCentre = (x: number, y: number, z: number) =>
    x >= xb && x < sx - xb && y >= yb && y < sy - yb && z >= zb && z < sz - zb;
  const maskCentre = new Uint8Array(lab.length);
  let any = false;
  for (let z = zb; z < sz - zb; ++z)
    for (let y = yb; y < sy - yb; ++y)
      for (let x = xb; x < sx - xb; ++x) {
        const i = idx(s, x, y, z);
        if (lab[i] === target) {
          maskCentre[i] = 1;
          any = true;
        }
      }
  if (!any) return { keep, rejects };
  const gap = opts.gap ?? 0;
  const dil = dilate26Times(maskCentre, s, 1 + gap);
  // Which neighbours really touch (one-voxel dilation), to flag the rest.
  let really: Set<number> | undefined;
  const contact1 = new Map<number, number>();
  if (gap > 0) {
    really = new Set();
    const touch1 = dilate26(maskCentre, s);
    for (let z = zb; z < sz - zb; ++z)
      for (let y = yb; y < sy - yb; ++y)
        for (let x = xb; x < sx - xb; ++x) {
          const i = idx(s, x, y, z);
          if (touch1[i] && !maskCentre[i]) {
            const l = lab[i];
            really.add(l);
            contact1.set(l, (contact1.get(l) ?? 0) + 1);
          }
        }
  }
  const contact = new Map<number, number>();
  for (let z = zb; z < sz - zb; ++z)
    for (let y = yb; y < sy - yb; ++y)
      for (let x = xb; x < sx - xb; ++x) {
        const i = idx(s, x, y, z);
        if (!dil[i] || maskCentre[i] || !inCentre(x, y, z)) continue;
        const l = lab[i];
        if (l === 0 || l === target) continue;
        contact.set(l, (contact.get(l) ?? 0) + 1);
      }
  if (contact.size === 0) return { keep, rejects };
  if (really && opts.gapMode === "nontouching")
    for (const l of contact.keys())
      if (really.has(l)) contact.set(l, contact1.get(l)!);
  const sizes = new Map<number, number>();
  for (let i = 0; i < lab.length; ++i) {
    const l = lab[i];
    if (contact.has(l)) sizes.set(l, (sizes.get(l) ?? 0) + 1);
  }
  for (const [l, ct] of [...contact].sort((a, b) => a[0] - b[0])) {
    const sz2 = sizes.get(l) ?? 0;
    if (ct < opts.minContact) rejects.set(l, ["contact_below_min", ct, sz2]);
    else if (ct > opts.maxContact)
      rejects.set(l, ["contact_above_max", ct, sz2]);
    else if (opts.minSegment > 0 && sz2 < opts.minSegment)
      rejects.set(l, ["segment_too_small", ct, sz2]);
    else
      keep.set(l, {
        contact_voxels: ct,
        cand_voxels: sz2,
        gap_contact: really !== undefined && !really.has(l),
      });
  }
  return { keep, rejects };
}

/**
 * find_contact_sites: contact blobs between id1 and id2, biggest first, as
 * (voxels, centroid in cutout coordinates). Computed on id2's bbox
 * +-(2 + gap), with both dilations widened by `gap` (create_contact_mask).
 */
export function findContactSites(
  lab: Uint32Array,
  s: Shape,
  id1: number,
  id2: number,
  minVoxels: number,
  maxSites: number,
  gap = 0,
): Array<[number, [number, number, number]]> {
  const [sx, sy, sz] = s;
  const lo = [sx, sy, sz];
  const hi = [-1, -1, -1];
  for (let z = 0; z < sz; ++z)
    for (let y = 0; y < sy; ++y)
      for (let x = 0; x < sx; ++x) {
        if (lab[idx(s, x, y, z)] !== id2) continue;
        if (x < lo[0]) lo[0] = x;
        if (y < lo[1]) lo[1] = y;
        if (z < lo[2]) lo[2] = z;
        if (x > hi[0]) hi[0] = x;
        if (y > hi[1]) hi[1] = y;
        if (z > hi[2]) hi[2] = z;
      }
  if (hi[0] < 0) return [];
  const margin = 2 + gap;
  for (let d = 0; d < 3; ++d) {
    lo[d] = Math.max(0, lo[d] - margin);
    hi[d] = Math.min(s[d], hi[d] + margin + 1);
  }
  const cs: Shape = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const n = cs[0] * cs[1] * cs[2];
  const m1 = new Uint8Array(n);
  const m2 = new Uint8Array(n);
  for (let z = 0; z < cs[2]; ++z)
    for (let y = 0; y < cs[1]; ++y)
      for (let x = 0; x < cs[0]; ++x) {
        const v = lab[idx(s, x + lo[0], y + lo[1], z + lo[2])];
        const j = idx(cs, x, y, z);
        if (v === id1) m1[j] = 1;
        else if (v === id2) m2[j] = 1;
      }
  const d1 = dilate26Times(m1, cs, 1 + gap);
  const d2 = dilate26Times(m2, cs, 1 + gap);
  const contact = new Uint8Array(n);
  for (let j = 0; j < n; ++j) contact[j] = (d1[j] & m2[j]) | (d2[j] & m1[j]);
  const { labels, count } = label26(contact, cs);
  if (count === 0) return [];
  const size = new Float64Array(count + 1);
  const sumx = new Float64Array(count + 1);
  const sumy = new Float64Array(count + 1);
  const sumz = new Float64Array(count + 1);
  for (let z = 0; z < cs[2]; ++z)
    for (let y = 0; y < cs[1]; ++y)
      for (let x = 0; x < cs[0]; ++x) {
        const l = labels[idx(cs, x, y, z)];
        if (!l) continue;
        size[l]++;
        sumx[l] += x;
        sumy[l] += y;
        sumz[l] += z;
      }
  // np.argsort(sizes)[::-1]: descending; numpy's default quicksort is not
  // stable, so equal-size ties may order differently from Python. Ties are
  // broken here by the higher label first, which is what [::-1] of a stable
  // ascending sort would give.
  const order = Array.from({ length: count }, (_, i) => i + 1).sort(
    (a, b) => size[b] - size[a] || b - a,
  );
  const sites: Array<[number, [number, number, number]]> = [];
  for (const l of order) {
    if (sites.length && size[l] < minVoxels) break;
    sites.push([
      size[l],
      [
        sumx[l] / size[l] + lo[0],
        sumy[l] / size[l] + lo[1],
        sumz[l] / size[l] + lo[2],
      ],
    ]);
    if (sites.length >= maxSites) break;
  }
  return sites;
}

/**
 * supervoxel_near: the voxel of `target` nearest `centroid` within
 * +-radius, as its index; undefined if none.
 */
export function voxelNear(
  lab: Uint32Array,
  s: Shape,
  centroid: readonly number[],
  target: number,
  radius: readonly number[] = [10, 10, 5],
): [number, number, number] | undefined {
  const c = centroid.map((v) => Math.trunc(v));
  const lo = c.map((v, d) => Math.max(0, v - radius[d]));
  const hi = c.map((v, d) => Math.min(s[d], v + radius[d] + 1));
  let best: [number, number, number] | undefined;
  let bestD = Infinity;
  // np.argwhere order is C order over (x, y, z): x slowest. argmin keeps the
  // first minimum, so iterate in that order and use strict <.
  for (let x = lo[0]; x < hi[0]; ++x)
    for (let y = lo[1]; y < hi[1]; ++y)
      for (let z = lo[2]; z < hi[2]; ++z) {
        if (lab[idx(s, x, y, z)] !== target) continue;
        const d =
          (x - centroid[0]) ** 2 +
          (y - centroid[1]) ** 2 +
          (z - centroid[2]) ** 2;
        if (d < bestD) {
          bestD = d;
          best = [x, y, z];
        }
      }
  return best;
}

/** scipy.ndimage.zoom(order=0, grid_mode=False) source index for output o. */
export function zoomSourceIndex(o: number, nIn: number, nOut: number) {
  return Math.floor((o * (nIn - 1)) / (nOut - 1) + 0.5);
}

export const PATCH_Z = 18;
export const PATCH_XY = 128;
export const PATCH_VOXELS = PATCH_Z * PATCH_XY * PATCH_XY;

/**
 * Writes the two mask channels of one patch into `out` (float32, length
 * 2*PATCH_VOXELS, (z, y, x) order) from the 32x32x18 crop of `lab` at
 * centroid c: 1 where the label is `a` (neuron), the second where it is `b`.
 * Returns false if the crop leaves the cutout (local_overflow).
 */
export function writeMasks(
  lab: Uint32Array,
  s: Shape,
  c: readonly number[],
  a: number,
  b: number,
  out: Float32Array,
  offset: number,
): boolean {
  const x0 = c[0] - 16;
  const y0 = c[1] - 16;
  const z0 = c[2] - 9;
  if (
    x0 < 0 ||
    y0 < 0 ||
    z0 < 0 ||
    c[0] + 16 > s[0] ||
    c[1] + 16 > s[1] ||
    c[2] + 9 > s[2]
  )
    return false;
  const src = new Int32Array(PATCH_XY);
  for (let o = 0; o < PATCH_XY; ++o) src[o] = zoomSourceIndex(o, 32, PATCH_XY);
  for (let z = 0; z < PATCH_Z; ++z)
    for (let y = 0; y < PATCH_XY; ++y) {
      const yy = y0 + src[y];
      for (let x = 0; x < PATCH_XY; ++x) {
        const v = lab[idx(s, x0 + src[x], yy, z0 + z)];
        const j = offset + x + PATCH_XY * (y + PATCH_XY * z);
        out[j] = v === a ? 1 : 0;
        out[j + PATCH_VOXELS] = v === b ? 1 : 0;
      }
    }
  return true;
}

/**
 * organelle_features on one patch: enclosure of the candidate by the target,
 * mean nonzero raw EM under the candidate, and the likely-mito flag.
 * em: uint8 raw (z, y, x); masks: float32 (z, y, x).
 */
export function organelleFeatures(
  em: Uint8Array,
  maskTarget: Float32Array,
  maskCand: Float32Array,
): {
  enclosure: number | null;
  cand_em_mean: number | null;
  likely_mito: boolean;
} {
  const s: Shape = [PATCH_XY, PATCH_XY, PATCH_Z];
  const n = PATCH_VOXELS;
  const c = new Uint8Array(n);
  const t = new Uint8Array(n);
  let nc = 0;
  for (let i = 0; i < n; ++i) {
    c[i] = maskCand[i] > 0 ? 1 : 0;
    t[i] = maskTarget[i] > 0 ? 1 : 0;
    nc += c[i];
  }
  if (nc < 8)
    return { enclosure: null, cand_em_mean: null, likely_mito: false };
  const er = erode6(c, s);
  const td = dilate6(t, s);
  let surf = 0;
  let touch = 0;
  let emSum = 0;
  let emN = 0;
  for (let i = 0; i < n; ++i) {
    if (c[i] && !er[i]) {
      surf++;
      if (td[i]) touch++;
    }
    if (c[i] && em[i] > 0) {
      emSum += em[i];
      emN++;
    }
  }
  const enc = touch / Math.max(surf, 1);
  const mean = emN > 30 ? emSum / emN : null;
  return {
    enclosure: enc,
    cand_em_mean: mean,
    likely_mito: enc > 0.6 && mean !== null && mean < 100,
  };
}
