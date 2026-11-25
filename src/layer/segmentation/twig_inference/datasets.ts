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
 * Per-dataset settings for in-browser Twig Capture, transcribed from
 * twigcapture configs/{banc,aedes,crantb,fanc}.yaml (branch banc-fragments).
 *
 * The pipeline is the same for every dataset; only data sources and the
 * resolutions differ. BANC, Aedes and CRANTb cut supervoxels at 32 nm and
 * read EM at 8 nm; FANC has no 32 nm mip and uses 34.4 / 8.6 nm. Everything
 * the pipeline computes in nm is derived from these (see gridOf), and the EM
 * must always be exactly 4x finer than the supervoxels in xy, as the model's
 * inputs are 4 EM voxels per mask voxel. The dataset is picked by matching
 * the segmentation layer's graphene table.
 */

export interface TwigDataset {
  key: string;
  label: string;
  /** CAVE datastack: written as the result's `dataset` (verdicts need it). */
  datastack: string;
  /** Graphene table name, matched against the layer's source URL. */
  table: string;
  /** Watershed supervoxels under the graphene layer (precomputed). */
  supervoxelUrl: string;
  /** Supervoxel scale used for cutouts: [32, 32, z] (FANC [34.4, 34.4, 45]). */
  supervoxelResolution: [number, number, number];
  /** EM, read at 1/4 of the supervoxel xy resolution (8 nm; FANC 8.6 nm). */
  emUrl: string;
  emResolution: [number, number, number];
  /**
   * Results catalog (cluster runs, index.json) the Twig Capture tab opens by
   * default: the newest model's results, served beside each viewer as
   * twigcapture_<dataset>/results/ (see deploy/deploy.sh).
   */
  catalogUrl: string;
  /**
   * CAVE synapse table that in-browser results take cand_synapses_total
   * from (pre + post rows on the partner). The same tables as the cluster's
   * counts: BANC and Aedes synapses_v2 (Aedes v3 is not served by CAVE),
   * CRANTb synapses_v3.
   */
  synapseTable: string;
  /** Caveats shown in the selector. */
  note?: string;
}

/**
 * Default model: Kevin's combined Aedes+BANC classifier, fine-tuned on
 * reviewer decisions (joint_finetune_20261123_105038 ep8), exported to ONNX
 * and rewritten so every Conv3d runs as Conv2d (conv3d_as_conv2d.py concat).
 */
export const DEFAULT_MODEL_URL =
  "https://radagast.hms.harvard.edu/twigcapture_dev/local_fp32c2d.onnx";

export const TWIG_DATASETS: TwigDataset[] = [
  {
    key: "banc",
    label: "BANC",
    datastack: "brain_and_nerve_cord",
    table: "wclee_fly_cns_001",
    catalogUrl: "https://radagast.hms.harvard.edu/twigcapture_banc/results/",
    synapseTable: "synapses_v2",
    supervoxelUrl:
      "precomputed://gs://zetta_lee_fly_cns_001_segmentation/v1_sharded",
    supervoxelResolution: [32, 32, 45],
    emUrl:
      "precomputed://https://bossdb-open-data.s3.amazonaws.com/bates_phelps_kim_yang2025/banc/cns_001_alignment_aligned_v0",
    emResolution: [8, 8, 45],
    note:
      "EM is the public 8 nm mip (downsampled); the cluster runner " +
      "subsamples the 4 nm zarr, so p can differ slightly from batch results.",
  },
  {
    key: "aedes",
    label: "Aedes",
    datastack: "wclee_aedes_brain",
    table: "wclee_aedes_brain",
    catalogUrl: "https://radagast.hms.harvard.edu/twigcapture_aedes/results/",
    synapseTable: "synapses_v2",
    supervoxelUrl: "precomputed://gs://fanc-fly_segmentation/wclee_aedes_brain",
    supervoxelResolution: [32, 32, 45],
    emUrl:
      "precomputed://gs://zetta_lee_mosquito_001_alignment_central/img/v2_sharded",
    emResolution: [8, 8, 45],
  },
  {
    key: "crantb",
    label: "CRANTb",
    datastack: "kronauer_ant",
    table: "kronauer_ant_x1",
    catalogUrl: "https://radagast.hms.harvard.edu/twigcapture_crantb/results/",
    synapseTable: "synapses_v3",
    supervoxelUrl:
      "precomputed://gs://zetta_ws/dkronauer-ant-001-240904-finetune-v3.2-0.27",
    supervoxelResolution: [32, 32, 42],
    emUrl: "precomputed://gs://dkronauer-ant-001-alignment-final/aligned",
    emResolution: [8, 8, 42],
  },
  {
    key: "fanc",
    label: "FANC",
    datastack: "fanc_production_mar2021",
    table: "mar2021_prod",
    catalogUrl: "https://radagast.hms.harvard.edu/twigcapture_fanc/results/",
    synapseTable: "synapses_nov2022",
    supervoxelUrl: "precomputed://gs://fanc-fly_segmentation/mar2021/ws",
    supervoxelResolution: [34.4, 34.4, 45],
    emUrl:
      "precomputed://gs://lee-lab_female-adult-nerve-cord/alignmentV4/em/rechunked",
    emResolution: [8.6, 8.6, 45],
    note:
      "FANC has no 32 nm supervoxel mip: cutouts are at 34.4 nm and EM at " +
      "8.6 nm, so the model sees ~7.5% more tissue per patch than it was " +
      "trained on (no FANC training data yet). There is no " +
      "twig_capture_decisions table on FANC yet, so verdicts cannot be saved.",
  },
];

/**
 * A dataset's grids, in nm. Region bins are half the supervoxel xy voxel
 * (16 nm for BANC/Aedes/CRANTb, 17.2 nm for FANC) so that a supervoxel
 * cutout centre is bin / 2; the EM is `emPerSv` times finer in xy.
 */
export function gridOf(
  ds: Pick<TwigDataset, "supervoxelResolution" | "emResolution">,
) {
  const [svNm, , zRes] = ds.supervoxelResolution;
  const emPerSv = svNm / ds.emResolution[0];
  if (Math.abs(emPerSv - 4) > 1e-6 || ds.emResolution[2] !== zRes)
    throw new Error(
      `EM ${ds.emResolution.join("x")} nm must be 1/4 of supervoxels ` +
        `${ds.supervoxelResolution.join("x")} nm in xy, same z`,
    );
  return { svNm, binNm: svNm / 2, zRes, emPerSv: 4 };
}

/** The dataset whose graphene table appears in any of the layer's URLs. */
export function datasetForUrls(urls: string[]): TwigDataset | undefined {
  for (const d of TWIG_DATASETS) {
    if (urls.some((u) => new RegExp(`/table/${d.table}(/|$)`).test(u)))
      return d;
  }
  return undefined;
}

/**
 * Pipeline parameters (configs/*.yaml `filtering`, identical for every
 * dataset). The synapse-density gate (min_synapses_within) is not applied:
 * it needs a precomputed per-dataset grid that only exists on the cluster, and
 * the orphan-first runner already runs without it.
 */
/**
 * Partner size, in L2 nodes, above which a suggestion is a LARGE MERGE. The
 * model was trained only on partners up to this size (the GT pipeline's and
 * the cluster runner's candidate cap), and it sees ~1 um around the contact,
 * not the partner's size: a large orphan wrongly merged does the most damage.
 * The browser no longer caps partners at this size; it tags them instead.
 */
export const LARGE_MERGE_L2 = 20;

/** "No limit" for the Twig Selector's partner-size field. */
export const NO_L2_LIMIT = 1_000_000;

export const TWIG_PARAMS = {
  cutout: [200, 200, 120] as [number, number, number],
  border: 20,
  minContactVoxels: 1,
  maxContactVoxels: 1500,
  minSegmentVoxels: 22,
  maxCandidateL2: 20,
  maxContactSites: 8,
  minContactBlob: 10,
  /**
   * Region bins, in bin units (half a supervoxel: 16 nm, FANC 17.2 nm) in
   * xy and z voxels: 1120 x 1120 nm x 28 slices (FANC 1204 nm).
   */
  binXY: 70,
  binZ: 28,
  /**
   * Extra-region passes so every searched L2 node's full extent is searched,
   * not only the band around its rep_coord (pipeline.ts, "Coverage of each
   * L2 node's full extent"). As the cluster runner's filtering.coverage_passes;
   * 0 reproduces the old rep_coord-only search.
   */
  coveragePasses: 3,
  /**
   * contact_gap_voxels, as the cluster runner's --contact_gap: pieces up to
   * this many supervoxel voxels apart count as touching, so a merge across a
   * thin gap (a neck lost at 32 nm) becomes a candidate. Rows found only
   * this way carry gap_contact: true, which Twig Capture tags "gap". 0
   * reproduces the old touching-only search. The Selector can override it.
   */
  contactGap: 1,
  /**
   * How the gap applies (volume_ops touchingLabelsCenter): "nontouching"
   * widens the search only for pieces that do not touch, so every touching
   * pair keeps exactly its gap-0 contact count, sites and score, and the gap
   * only adds rows. "all" is the cluster runner's rule, which widens it for
   * every pair: on Aedes 648518347555976042 that moved 180 of 432 rows' p
   * and pushed 14 touching candidates (up to p 0.94) past the contact cap.
   */
  contactGapMode: "nontouching" as "all" | "nontouching",
  maxEmZeroFrac: 0.05,
  l2MaxDtNm: 180,
  l2MaxSizeNm3: 4.0e8,
  /** Rows with p above this are reported (production --threshold 0.05). */
  reportThreshold: 0.05,
};
