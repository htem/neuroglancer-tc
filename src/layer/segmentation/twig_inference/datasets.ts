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
 * twigcapture configs/{banc,aedes,crantb}.yaml (branch banc-fragments).
 *
 * The pipeline is the same for every dataset; only data sources and the z
 * resolution differ. The dataset is picked by matching the segmentation
 * layer's graphene table.
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
  /** Supervoxel scale used for cutouts: [32, 32, z]. */
  supervoxelResolution: [number, number, number];
  /** EM read at 8 nm in xy. */
  emUrl: string;
  emResolution: [number, number, number];
  /**
   * Results catalog (cluster runs, index.json) the Twig Capture tab opens by
   * default: the newest model's results, served beside each viewer as
   * twigcapture_<dataset>/results/ (see deploy/deploy.sh).
   */
  catalogUrl: string;
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
    supervoxelUrl:
      "precomputed://gs://zetta_ws/dkronauer-ant-001-240904-finetune-v3.2-0.27",
    supervoxelResolution: [32, 32, 42],
    emUrl: "precomputed://gs://dkronauer-ant-001-alignment-final/aligned",
    emResolution: [8, 8, 42],
  },
];

/** The dataset whose graphene table appears in any of the layer's URLs. */
export function datasetForUrls(urls: string[]): TwigDataset | undefined {
  for (const d of TWIG_DATASETS) {
    if (urls.some((u) => new RegExp(`/table/${d.table}(/|$)`).test(u)))
      return d;
  }
  return undefined;
}

/**
 * Pipeline parameters (configs/*.yaml `filtering`, identical for the three
 * datasets). The synapse-density gate (min_synapses_within) is not applied:
 * it needs a precomputed per-dataset grid that only exists on the cluster, and
 * the orphan-first runner already runs without it.
 */
export const TWIG_PARAMS = {
  cutout: [200, 200, 120] as [number, number, number],
  border: 20,
  minContactVoxels: 1,
  maxContactVoxels: 1500,
  minSegmentVoxels: 22,
  maxCandidateL2: 20,
  maxContactSites: 8,
  minContactBlob: 10,
  /** Region bins, in 16 nm (xy) and z-voxel units: 1120 x 1120 x 28 slices. */
  binXY: 70,
  binZ: 28,
  maxEmZeroFrac: 0.05,
  l2MaxDtNm: 180,
  l2MaxSizeNm3: 4.0e8,
  /** Rows with p above this are reported (production --threshold 0.05). */
  reportThreshold: 0.05,
};
