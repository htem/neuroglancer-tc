/**
 * @license
 * Ported from Coda by Philipp Schlegel (https://github.com/navis-org/coda,
 * src/packs/seam/gpuAssessor), with changes for the Twig Capture model.
 *
 * MIT License
 *
 * Copyright (c) 2026 NAVis
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * WGSL for the hand-written Twig Capture network, generated per step with
 * every size a constant. Plain functions of a step's numbers, so they are
 * testable without a GPU; `r3d_network.ts` compiles and runs them.
 *
 * Coda's shaders.ts (Philipp Schlegel) for SEAM's 3D ResNet-18, generalised
 * from cubes to the (depth, height, width) boxes and kernels of this model
 * (a (3, 7, 7) stem with stride (1, 2, 2) on an 18 x 128 x 128 patch). Coda's
 * sparse stem and background skipping are not ported: they exploit
 * mask-only input, and our first channel is dense EM.
 *
 * ## The convolution: an implicit matrix multiply
 *
 * A convolution is a (Cout x Cin*taps) by (Cin*taps x voxels) product. The
 * right-hand matrix is never built: each tile of it is read from the input,
 * at the tap's offset, straight into workgroup memory. Tile: 64 output
 * channels x 64 voxels a workgroup, 8 x 4 a thread, 16 reduction steps a
 * tile. The reduction index is tap-major, `k = tap * Cin + ci`; when Cin is a
 * multiple of 16 one tile is one tap, so a thread works out its input
 * address once per tile (the stem, Cin = 3, takes the general path).
 *
 * ## Splitting the reduction
 *
 * The late stages are small in space and long in the reduction (layer4: 192
 * voxels against 13,824 terms), so they split the reduction across
 * workgroups, each writing a partial sum, and a second pass (`reduceShader`)
 * adds them and applies the epilogue. WGSL has no float atomics.
 *
 * ## Two forms for two shader translators
 *
 * Firefox's translator (on a Mac, where Coda measured it) charges for every
 * loop iteration; Chrome's does not. So the 16-step inner loop can be
 * written out (`ConvForm.unrolled`, Coda §18), chosen by `frugalShaders`.
 * Both forms add the same terms in the same order, so the logits are equal
 * to the bit. Coda's other Firefox change (a packed stem) belongs to its
 * sparse stem, which this network does not have.
 *
 * ## The epilogue
 *
 * Batch norm is folded into each conv's bias by the export, so after a conv
 * comes: bias; then ReLU (a block's first conv, the stem), or the residual
 * add and ReLU (a block's second conv), or nothing (a downsample).
 */

import type {
  ConvStep,
  HeadStep,
} from "#src/layer/segmentation/twig_inference/gpu/r3d_plan.js";

/**
 * A convolution's tile: output channels (m) and voxels (n) a workgroup
 * computes, per thread (perM x perN), and reduction terms a tile holds (k).
 */
export interface Tile {
  readonly m: number;
  readonly n: number;
  readonly perM: number;
  readonly perN: number;
  readonly k: number;
}
/** Coda's tile, the fastest of five it swept on an M3 Max. */
export const DEFAULT_TILE: Tile = { m: 64, n: 64, perM: 8, perN: 4, k: 16 };
/** Reduction terms a tile holds, in the default tile. */
export const TILE_K = DEFAULT_TILE.k;

/** Whether this generator can write a tile (whole loads per thread). */
export function tileValid(t: Tile): boolean {
  const threads = (t.m / t.perM) * (t.n / t.perN);
  return (
    Number.isInteger(threads) &&
    threads <= 256 &&
    (t.k * t.n) % threads === 0 &&
    (t.k * t.m) % threads === 0 &&
    threads % t.n === 0 &&
    64 % t.m === 0
  );
}

/**
 * Storage and arithmetic types. "f16" needs the adapter's shader-f16
 * feature. The patch, partial sums, biases and the head stay f32.
 */
export interface Precision {
  readonly store: "f32" | "f16";
  readonly math: "f32" | "f16";
}
export const F32: Precision = { store: "f32", math: "f32" };

/** f32 -> f16 bits, rounding to nearest even (IEEE 754 binary16). */
export function toHalfBits(f: number): number {
  const x = new Uint32Array(new Float32Array([f]).buffer)[0];
  const sign = (x >>> 16) & 0x8000;
  const e = (x >>> 23) & 0xff;
  let m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  let he = e - 127 + 15;
  if (he >= 0x1f) return sign | 0x7c00;
  if (he <= 0) {
    if (he < -10) return sign;
    m |= 0x800000;
    const shift = 14 - he;
    let hm = m >>> shift;
    const rem = m & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && hm & 1)) hm++;
    return sign | hm;
  }
  let hm = m >>> 13;
  const rem = m & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && hm & 1)) {
    hm++;
    if (hm === 0x400) {
      hm = 0;
      he++;
      if (he >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (he << 10) | hm;
}

/** Workgroup size of the one-thread-per-element passes. */
export const FLAT = 256;

/**
 * Enough splits to put about this many workgroups on the GPU, never fewer
 * than eight tiles a split (Coda swept 80-2,560 on an M3 Max; 160-1,280 timed
 * alike).
 */
const SPLIT_TARGET = 320;

export interface ConvLayout {
  /** Reduction length, `Cin * taps`. */
  readonly k: number;
  /** Tiles along the reduction; weights are zero-padded to `chunks * TILE_K` rows. */
  readonly chunks: number;
  readonly splits: number;
  /** Workgroups along voxels, channels and splits. */
  readonly groups: readonly [number, number, number];
  readonly tile: Tile;
}

export const voxelsOf = (s: readonly number[]) => s[0] * s[1] * s[2];
const tapsOf = (step: ConvStep) =>
  step.kernel[0] * step.kernel[1] * step.kernel[2];

export function convLayout(
  step: ConvStep,
  splitTarget = SPLIT_TARGET,
  tile: Tile = DEFAULT_TILE,
): ConvLayout {
  if (step.cout % tile.m)
    throw new Error(
      `${step.name}: ${step.cout} channels is not a multiple of ${tile.m}`,
    );
  const k = step.cin * tapsOf(step);
  const chunks = Math.ceil(k / tile.k);
  const across = Math.ceil(voxelsOf(step.outShape) / tile.n);
  const tiles = across * (step.cout / tile.m);
  const splits = Math.max(
    1,
    Math.min(Math.floor(chunks / 8), Math.round(splitTarget / tiles)),
  );
  return {
    k,
    chunks,
    splits,
    groups: [across, step.cout / tile.m, splits],
    tile,
  };
}

/** Floats in a step's laid-out weights: `[k][Cout]`, zero-padded to whole tiles. */
export function convWeightFloats(step: ConvStep, layout: ConvLayout): number {
  return layout.chunks * layout.tile.k * step.cout;
}

/**
 * The weights, (Cout, Cin, taps), rearranged to the shader's `[k][Cout]` with
 * `k = tap * Cin + ci`, into `out` (a buffer's mapped range, so the weights
 * are written once where the GPU reads them). `out` must start zeroed: the
 * padding rows are not written. Output channels go in blocks of 16 innermost,
 * which Coda measured at half the time of the plain order.
 */
export function convWeightLayout(
  step: ConvStep,
  layout: ConvLayout,
  weight: Float32Array<ArrayBufferLike> = step.weight(),
  out: Float32Array<ArrayBufferLike> = new Float32Array(
    convWeightFloats(step, layout),
  ),
): Float32Array<ArrayBufferLike> {
  const taps = tapsOf(step);
  const BLOCK = 16;
  for (let c0 = 0; c0 < step.cout; c0 += BLOCK) {
    const c1 = Math.min(c0 + BLOCK, step.cout);
    for (let ci = 0; ci < step.cin; ci++) {
      for (let tap = 0; tap < taps; tap++) {
        const row = (tap * step.cin + ci) * step.cout;
        for (let co = c0; co < c1; co++) {
          out[row + co] = weight[(co * step.cin + ci) * taps + tap];
        }
      }
    }
  }
  return out;
}

const u = (n: number) => `${n}u`;

/** The epilogue as WGSL, on `v` for channel `m` at voxel `n`. */
function epilogue(step: ConvStep): string {
  const lines = [`v = v + ep[m];`];
  if (step.residual)
    lines.push(
      `v = max(v + f32(res[m * ${u(voxelsOf(step.outShape))} + n]), 0.0);`,
    );
  else if (step.relu) lines.push(`v = max(v, 0.0);`);
  return lines.join("\n    ");
}

/** How a convolution is written; neither form changes a sum. */
export interface ConvForm {
  /** The inner loop's 16 steps written out (Firefox on a Mac). */
  readonly unrolled?: boolean;
  /** Storage and arithmetic types (default f32). */
  readonly precision?: Precision;
}

/**
 * Firefox on a Mac, by name (Coda's `frugalShaders`): its translator
 * charges for loop iterations, measured there; elsewhere the plain form.
 */
export function frugalShaders(
  agent = globalThis.navigator?.userAgent,
): boolean {
  return (
    typeof agent === "string" &&
    agent.includes("Firefox/") &&
    agent.includes("Macintosh")
  );
}

/**
 * The convolution. Bindings: 0 input, 1 weights, 2 output (or partial sums
 * when split), 3 the bias, and 4 the residual, both only when not split.
 */
export function convShader(
  step: ConvStep,
  layout: ConvLayout,
  form: ConvForm = {},
): string {
  const { k, chunks, splits, tile } = layout;
  const TILE_M = tile.m;
  const TILE_N = tile.n;
  const TILE_K = tile.k;
  const PER_M = tile.perM;
  const PER_N = tile.perN;
  const THREADS = (TILE_M / PER_M) * (TILE_N / PER_N);
  const prec = form.precision ?? F32;
  const ST = prec.store;
  const MT = prec.math;
  const IT = step.input === "input" ? "f32" : ST;
  const f16 = ST === "f16" || MT === "f16";
  const [D, H, W] = step.inShape;
  const [OD, OH, OW] = step.outShape;
  const [, KH, KW] = step.kernel;
  const [SD, SH, SW] = step.stride;
  const [PD, PH, PW] = step.pad;
  const vi = D * H * W;
  const vo = OD * OH * OW;
  const cin = step.cin;
  const perSplit = Math.ceil(chunks / splits);
  const fast = cin % TILE_K === 0;
  const MB = TILE_M / PER_M;
  const NB = TILE_N / PER_N;
  const loadsB = (TILE_K * TILE_N) / THREADS;
  const loadsA = (TILE_K * TILE_M) / THREADS;
  const rowStep = THREADS / TILE_N;
  const rows = (fn: (r: number) => string) =>
    Array.from({ length: PER_M }, (_, r) => fn(r)).join("\n");
  const cols = (fn: (s: number) => string) =>
    Array.from({ length: PER_N }, (_, s) => fn(s)).join("\n");
  const both = (fn: (r: number, s: number) => string) =>
    rows((r) => cols((s) => fn(r, s)));

  // The input voxel at tap `tap` from this thread's window corner, or 0
  // outside the volume: zero padding, done by not reading.
  const address = (tap: string) => `
      let t0 = i32(${tap} / ${u(KH * KW)});
      let t1 = i32((${tap} / ${u(KW)}) % ${u(KH)});
      let t2 = i32(${tap} % ${u(KW)});
      let zi = cd + t0;
      let yi = ch + t1;
      let xi = cw + t2;
      let inb = valid && zi >= 0 && zi < ${D} && yi >= 0 && yi < ${H} && xi >= 0 && xi < ${W};
      let base = select(0u, (u32(zi) * ${u(H)} + u32(yi)) * ${u(W)} + u32(xi), inb);`;

  const loadB = fast
    ? `
    {
      let tap = q / ${u(cin / TILE_K)};
      let c0 = (q % ${u(cin / TILE_K)}) * ${u(TILE_K)};${address("tap")}
      for (var i = 0u; i < ${u(loadsB)}; i++) {
        let r = kb + i * ${u(rowStep)};
        Bs[r * ${u(TILE_N)} + nn] = select(${ST}(0), ${ST}(inp[(c0 + r) * ${u(vi)} + base]), inb);
      }
    }`
    : `
    for (var i = 0u; i < ${u(loadsB)}; i++) {
      let r = kb + i * ${u(rowStep)};
      let kidx = q * ${u(TILE_K)} + r;
      let ci = kidx % ${u(cin)};
      let tap = kidx / ${u(cin)};${address("tap")}
      Bs[r * ${u(TILE_N)} + nn] = select(${ST}(0), ${ST}(inp[ci * ${u(vi)} + base]), inb && kidx < ${u(k)});
    }`;

  const reductionStep = (
    r: string,
  ) => `${rows((i) => `      let a${i} = ${MT}(As[${r} * ${u(TILE_M)} + tm + ${u(i * MB)}]);`)}
${cols((j) => `      let b${j} = ${MT}(Bs[${r} * ${u(TILE_N)} + tn + ${u(j * NB)}]);`)}
${both((i, j) => `      acc${i}_${j} = fma(a${i}, b${j}, acc${i}_${j});`)}`;
  const inner = form.unrolled
    ? Array.from(
        { length: TILE_K },
        (_, r) => `    {\n${reductionStep(u(r))}\n    }`,
      ).join("\n")
    : `    for (var r = 0u; r < ${u(TILE_K)}; r++) {
${reductionStep("r")}
    }`;

  const write = (r: number, s: number) => {
    const m = `(m0 + tm + ${u(r * MB)})`;
    const n = `(n0 + tn + ${u(s * NB)})`;
    return splits === 1
      ? `  if (${n} < ${u(vo)}) {
    let m = ${m};
    let n = ${n};
        var v = f32(acc${r}_${s});
    ${epilogue(step)}
    outp[m * ${u(vo)} + n] = ${ST}(v);
  }`
      : `  if (${n} < ${u(vo)}) { outp[(wg.z * ${u(step.cout)} + ${m}) * ${u(vo)} + ${n}] = f32(acc${r}_${s}); }`;
  };

  return /* wgsl */ `${f16 ? "enable f16;" : ""}
@group(0) @binding(0) var<storage, read> inp: array<${IT}>;
@group(0) @binding(1) var<storage, read> wt: array<${ST}>;
@group(0) @binding(2) var<storage, read_write> outp: array<${splits === 1 ? ST : "f32"}>;
${splits === 1 ? "@group(0) @binding(3) var<storage, read> ep: array<f32>;" : ""}
${splits === 1 && step.residual ? `@group(0) @binding(4) var<storage, read> res: array<${ST}>;` : ""}
var<workgroup> As: array<${ST}, ${TILE_K * TILE_M}>;
var<workgroup> Bs: array<${ST}, ${TILE_K * TILE_N}>;

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let n0 = wg.x * ${u(TILE_N)};
  let m0 = wg.y * ${u(TILE_M)};
  let tm = li / ${u(NB)};
  let tn = li % ${u(NB)};
${both((r, s) => `  var acc${r}_${s} = ${MT}(0);`)}

  // The output voxel this thread gathers input for, fixed for the whole
  // reduction, and the corner of its window in the input.
  let nn = li % ${u(TILE_N)};
  let kb = li / ${u(TILE_N)};
  let gn = n0 + nn;
  let valid = gn < ${u(vo)};
  let cd = i32(gn / ${u(OH * OW)}) * ${SD} - ${PD};
  let ch = i32((gn / ${u(OW)}) % ${u(OH)}) * ${SH} - ${PH};
  let cw = i32(gn % ${u(OW)}) * ${SW} - ${PW};

  let q0 = wg.z * ${u(perSplit)};
  let q1 = min(q0 + ${u(perSplit)}, ${u(chunks)});
  for (var q = q0; q < q1; q++) {${loadB}
    let a0 = q * ${u(TILE_K * step.cout)} + m0;
    for (var i = 0u; i < ${u(loadsA)}; i++) {
      let e = li + i * ${u(THREADS)};
      As[e] = wt[a0 + (e / ${u(TILE_M)}) * ${u(step.cout)} + e % ${u(TILE_M)}];
    }
    workgroupBarrier();
${inner}
    workgroupBarrier();
  }
${both(write)}
}`;
}

/**
 * Add a split step's partial sums and apply its epilogue. Bindings: 0
 * partials, 1 output, 2 bias, 3 the residual if any. A thread per element.
 */
export function reduceShader(
  step: ConvStep,
  splits: number,
  precision: Precision = F32,
): string {
  const vo = voxelsOf(step.outShape);
  const total = step.cout * vo;
  const ST = precision.store;
  return /* wgsl */ `${ST === "f16" ? "enable f16;" : ""}
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<${ST}>;
@group(0) @binding(2) var<storage, read> ep: array<f32>;
${step.residual ? `@group(0) @binding(3) var<storage, read> res: array<${ST}>;` : ""}

@compute @workgroup_size(${FLAT})
fn main(@builtin(global_invocation_id) id: vec3u) {
  let e = id.x;
  if (e >= ${u(total)}) { return; }
  let m = e / ${u(vo)};
  let n = e % ${u(vo)};
  var v = 0.0;
  for (var s = 0u; s < ${u(splits)}; s++) { v = v + part[s * ${u(total)} + e]; }
    ${epilogue(step)}
  outp[e] = ${ST}(v);
}`;
}

/**
 * Global average pool, fc.0 + ReLU, fc.3: one workgroup, two logits out.
 * Bindings: 0 the last block's output, 1 the head's weights packed as
 * [w1 (hidden x channels), b1, w2 (2 x hidden), b2], 2 the two logits.
 */
export function headShader(head: HeadStep, precision: Precision = F32): string {
  const ST = precision.store;
  const { channels: C, voxels: V, hidden: Hd } = head;
  const perC = Math.ceil(C / FLAT);
  const perH = Math.ceil(Hd / FLAT);
  const oB1 = Hd * C;
  const oW2 = oB1 + Hd;
  const oB2 = oW2 + 2 * Hd;
  return /* wgsl */ `${ST === "f16" ? "enable f16;" : ""}
@group(0) @binding(0) var<storage, read> inp: array<${ST}>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<f32>;
var<workgroup> feat: array<f32, ${C}>;
var<workgroup> hid: array<f32, ${Hd}>;
var<workgroup> part: array<f32, ${FLAT}>;

@compute @workgroup_size(${FLAT})
fn main(@builtin(local_invocation_index) li: u32) {
  for (var i = 0u; i < ${u(perC)}; i++) {
    let c = li + i * ${u(FLAT)};
    if (c < ${u(C)}) {
      var t = 0.0;
      for (var n = 0u; n < ${u(V)}; n++) { t = t + f32(inp[c * ${u(V)} + n]); }
      feat[c] = t / ${V}.0;
    }
  }
  workgroupBarrier();
  for (var i = 0u; i < ${u(perH)}; i++) {
    let h = li + i * ${u(FLAT)};
    if (h < ${u(Hd)}) {
      var t = w[${u(oB1)} + h];
      for (var c = 0u; c < ${u(C)}; c++) { t = fma(w[h * ${u(C)} + c], feat[c], t); }
      hid[h] = max(t, 0.0);
    }
  }
  workgroupBarrier();
  for (var o = 0u; o < 2u; o++) {
    var t = 0.0;
    for (var h = li; h < ${u(Hd)}; h = h + ${u(FLAT)}) { t = fma(w[${u(oW2)} + o * ${u(Hd)} + h], hid[h], t); }
    part[li] = t;
    workgroupBarrier();
    for (var width = ${u(FLAT / 2)}; width > 0u; width = width / 2u) {
      if (li < width) { part[li] = part[li] + part[li + width]; }
      workgroupBarrier();
    }
    if (li == 0u) { outp[o] = part[0] + w[${u(oB2)} + o]; }
    workgroupBarrier();
  }
}`;
}

/** The head's weights in `headShader`'s packed order. */
export function headWeights(head: HeadStep): Float32Array {
  const out = new Float32Array(
    head.w1.length + head.b1.length + head.w2.length + head.b2.length,
  );
  let o = 0;
  for (const a of [head.w1, head.b1, head.w2, head.b2]) {
    out.set(a, o);
    o += a.length;
  }
  return out;
}
