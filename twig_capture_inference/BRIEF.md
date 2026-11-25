# In-browser Twig Capture: brief

Branch `browser-inference` (off `crantb`), started 2026-11-23.

## Goal

Run the Twig Capture merge model inside the viewer, on the reviewer's own
GPU or CPU, so anyone can ask "suggest merges for this fragment" with nothing
to install. The model and the image data are both fetched over the web: the
model as an ONNX file (radagast now, GCS later), and EM and segmentation from
the same sources neuroglancer already reads for BANC, Aedes and CRANTb.

The plan is to split the pipeline into one core that runs both in batch on
O2 and in the browser:

| part | batch (O2) | browser |
|---|---|---|
| fetch EM, segmentation, L2 graph | Python (cloudvolume, caveclient) | neuroglancer's chunk fetching, with the user's CAVE token |
| candidate finding and patch building | one I/O-free numpy module | the same module, run under Pyodide |
| model | ONNX on onnxruntime-gpu | ONNX on ONNX Runtime Web (WebGPU, else WASM) |

Steps:

1. Export the model to ONNX and check it against PyTorch. **Done.**
2. Score real-sized patches in the browser: correctness and speed. **Done
   (2026-11-23): exact on WebGPU at 83 ms/patch on an Apple M-series GPU,
   about 2 min per 1,500-candidate fragment** (the Conv2d rewrite, below).
3. Separate candidate finding and patch building (the `run_twigcapture_v2.py`
   region loop) from all I/O. Run it under Pyodide on neuroglancer's cutouts.
4. Move the batch runner onto the same module and ONNX model, so batch and
   browser cannot drift apart.

## The model

`LocalOnlyClassifier`: a torchvision `r3d_18` with a 3-channel stem and a
512→256→2 head, defined in twigcapture `scripts/model_utils.py`.

- **Input** `x_local`, float32 `[N, 3, 18, 128, 128]` in (z, y, x) order. The
  channels are:
  1. EM at 8 nm, scaled to [-1, 1];
  2. the neuron-side mask;
  3. the candidate-side mask.

  The masks come from 32 nm segmentation, upsampled ×4 in xy. This is how
  `run_twigcapture_v2.py` builds a patch (around line 1830).
- **Output** `logits [N, 2]`; p(merge) = softmax(logits)[:, 1].
- **Checkpoint:** `joint_finetune_20261123_105038` ep8, Kevin's combined
  Aedes+BANC model, fine-tuned on reviewer decisions.

### Our own WebGPU network (2026-11-25, adapted from Coda)

ONNX Runtime Web runs a 3D Conv on WebGPU through a naive per-voxel shader
(`Conv3DNaive`). Our Conv2d rewrite (below) sidestepped most of that, at
82 ms/patch, about 1.5 TFLOP/s for the model's ~124 GFLOP. Philipp
Schlegel's Coda (https://github.com/navis-org/coda,
`src/packs/seam/gpuAssessor`, MIT) went further for SEAM's 3D ResNet-18: no
ONNX Runtime at all, with every convolution an implicit matrix multiply in
WGSL. `src/layer/segmentation/twig_inference/gpu/` is that approach, ported
and generalised to our anisotropic patch:

- `onnx_proto.ts`: Coda's ONNX reader (initializers and nodes only).
- `r3d_plan.ts`: our r3d_18 as a fixed plan. Every weight is checked by
  shape. It reads either the Conv2d rewrite we serve
  (`local_fp32c2d.onnx`, with the depth taps unstacked again) or the plain
  3D export, so no new model file is needed. BN is folded into the conv
  biases by the export.
- `r3d_shaders.ts`: a 64 × 64 output tile per workgroup, 8 × 4 per thread,
  16 reduction terms per tile. Layers 3 and 4 split the reduction across
  workgroups and add the partial sums in a second pass. Bias, ReLU and the
  residual are folded into each conv. The head (pool, fc, ReLU, fc) is one
  workgroup.
- `r3d_network.ts`: buffers are allocated once and reused when dead, with
  one submit and an 8-byte read back per patch. Up to 12 patches are in
  flight.

**Measured (Apple M1 Max, 32-core GPU, Chrome, the published
`local_fp32c2d.onnx`):**

| | ms/patch | |
|---|---|---|
| ONNX Runtime 1.22, WebGPU (c2d) | 80.4 | |
| **our WebGPU network** | **40.7** | 3.0 TFLOP/s |

- **Parity:** max \|dp\| against ONNX Runtime is 1.2e-7 over 24 patches.
  ORT itself matches PyTorch exactly on the reference patches
  (p 0.8288575, 0.5244967).
- **Per step** (`gpu.profile`): stem 2.5 ms; each layer1 conv 5.1 ms
  (3.2 TFLOP/s, 82 ms of 41); layer2 convs 2.9; layer3 1.6; layer4 1.0.
  Every stage runs at 2.6–3.2 TFLOP/s, so what is left is the kernel's
  own speed (plain f32, no subgroups), not the plan. Coda's figures were
  measured on an M3 Max.
- **The page checks it the first time.** When a model is first opened on
  an adapter, two test patches are scored by both runtimes. If they differ
  by more than 1e-4, or the network cannot be built, ONNX Runtime is used.
  The verdict is kept in IndexedDB per model URL and adapter.
- **Device setting:** auto and **GPU (WebGPU)** use our network.
  **GPU (ONNX Runtime)** is the previous path.

**End to end** (Aedes `648518347555976042`: 36 regions, 2,477 patches,
432 suggestions, M1 Max). Busy seconds per phase (`benchmark.phase_seconds`):

| | total | model | em_cutout | sites (CPU) | sv_cutout | cand_l2 | sv_roots |
|---|---|---|---|---|---|---|---|
| before: ORT, one region at a time | 512 s | 205 | 111 | 105 | 34 | 34 | 18 |
| after: our network + overlap | 247–263 s | 164* | 90–101 | 90 | 26–32 | 33 | 32–39 |

\* Overlapped with everything else. Busy time includes waiting while the
page's main thread is busy with the next region, so it reads about
66 ms/patch, against 41 alone.

The suggestions are identical: the same 432 rows in the same order with
the same prob_merge, and the same rejects. The next bottleneck is the
page's own work: the CPU site building (~90 s) and the EM cutouts (~95 s,
JPEG decode included). These now set the pace, not the GPU.

**Measuring:** from the console, `m = await twigCaptureModel.open(url,
"webgpu")`, then `await m.scorePatches([...])`, or
`await m.gpu.profile(patch, 30)` for per-step times. `"webgpu-ort"` gives
ONNX Runtime for comparison.

## What has been measured

ONNX against PyTorch on the 2 reference patches:

| model file | size | largest difference in p | ORT CPU, 8 O2 cores |
|---|---|---|---|
| `local_fp32.onnx` | 133 MB | 6e-8 | 1.26 s/patch |
| `local_fp16.onnx` (weights fp16, fp32 inputs and outputs) | 67 MB | 5e-5 | |
| `local_fp16pad.onnx` (WebGPU-compatible graph, below) | 67 MB | 5e-5 | |

The reference p values are 0.8289 and 0.5245.

At 1.26 s/patch, CPU alone takes about 30 minutes for one fragment
(~1,500 candidates). **WebGPU is required.**

### WebGPU on ONNX Runtime Web (1.20.1 and 1.22.0, Apple M-series / Metal)

1. The original graph fails on the first 3D Conv with
   `Unsupported padding parameter: 1,3,3,1,3,3`.
2. `scripts/explicit_pad.py` rewrites every padded Conv as
   `Pad(0) -> Conv(pads=0)`. The session must then be created with
   `graphOptimizationLevel: "disabled"`, otherwise ORT fuses the Pad back
   into the Conv. With that, **all 3D Convs run on WebGPU.**
3. Next it failed on `GlobalAveragePool` ("kernelShape.length > 2 is not
   supported for NHWC"). `explicit_pad.py` now replaces it with
   `ReduceMean(axes=[2,3,4], keepdims=1)`. Verified on CPU.
4. (2026-11-23, headless Chromium 1243 on O2, SwiftShader adapter, ORT
   1.22.0) `local_fp16pad.onnx`: the session now builds on WebGPU, so
   `ReduceMean` is accepted. The run fails only because SwiftShader lacks
   `shader-f16` ("'f16' type used without 'f16' extension enabled" in the
   Cast kernel), so every p comes out 0.5000. This is a SwiftShader limit;
   Metal and NVIDIA have f16. The padded graph now has only Cast, Pad, Conv,
   Relu, Add, ReduceMean, Flatten and Gemm, all in ORT Web's WebGPU op set.
   Still to run: fp16pad on a real GPU (Mac click, or the gpu_requeue job
   below), and `local_fp32pad.onnx` (new, 133 MB) on SwiftShader.

### The fix: every Conv3d as Conv2d (2026-11-23, Apple M-series, Chrome, Metal-3)

`local_fp32pad.onnx` was exact on WebGPU but took 3.7 s/patch at every batch
size, slower than O2's CPU. ORT Web 1.22 runs 3D Conv on WebGPU with
`Conv3DNaive`. Its 2D Conv uses tuned matmul kernels.

`scripts/conv3d_as_conv2d.py` rewrites the graph so activations stay in
(N, D, C, H, W) layout. Each Conv3d becomes: zero-pad depth only, slice the
kd depth taps, then either

- **concat** (`c2d`): stack the taps along channels and run one Conv2d with
  the weight reordered to (Co, kd·Ci, kh, kw), 20 Conv2d nodes in all; or
- **sum** (`s2d`): one Conv2d per tap, then add, 54 Conv2d nodes.

H and W padding stays inside the Conv2d, and GlobalAveragePool becomes
ReduceMean over axes (1, 3, 4). No Pad feeds a Conv directly, so these models
load with full graph optimisation. `scripts/check_onnx.py` checks any model
against PyTorch on CPU.

| model | WebGPU ms/patch (batch 1 / 4 / 8) | max \|dp\| on WebGPU | 1,500 candidates |
|---|---|---|---|
| `local_fp32pad.onnx` (3D Conv) | 3704 / 3689 / – | 1e-7 | 93 min |
| `local_fp32s2d.onnx` | 335 / 251 / 232 | 1e-7 | 6 min |
| **`local_fp32c2d.onnx`** | **83** / 126 / 131 | **9e-8** | **2.1 min** |
| `local_fp16c2d.onnx` | 67 / 67 / 70 | 3.8e-3 | 1.7 min |
| `local_fp16pad.onnx` | 3 / 11 / 2 | 0.28: p is 0.5520 for every input | – |

- **Use `local_fp32c2d.onnx` at batch 1.** It is exact, and fp16 saves only
  about 20%.
- The rewrite also halves CPU time with onnxruntime: 370 ms/patch against
  716 ms on the Mac.
- fp16pad's constant output is a bug in the fp16 3D/Pad path on WebGPU: the
  fp16 Conv2d model is fine. It has not been chased further, because Conv2d
  replaces that path.
- The batch 4/8 slowdown for c2d comes from only 2 timing reps, so it may be
  noise. Re-check when real patches are scored.

Build the models (any machine with onnx and onnxruntime; fp16 also needs
onnxconverter-common):

```
python scripts/conv3d_as_conv2d.py <dev>/local_fp32.onnx <dev>/local_fp32c2d.onnx concat
python scripts/conv3d_as_conv2d.py <dev>/local_fp32.onnx <dev>/local_fp32s2d.onnx sum
python -c "import onnx; from onnxconverter_common import float16 as f; onnx.save(f.convert_float_to_float16(onnx.load('<dev>/local_fp32c2d.onnx'), keep_io_types=True), '<dev>/local_fp16c2d.onnx')"
python scripts/check_onnx.py <dev> <dev>/local_fp32c2d.onnx
```

## In the viewer: Twig Selector (2026-11-23)

A **Twig Selector** tab on graphene segmentation layers (BANC, Aedes,
CRANTb) queues objects ("Add visible segments", or paste root ids) and runs
the whole pipeline in the browser. Results open in the **Twig Capture** tab,
listed as `[this browser]`. Nothing runs on a server:

- **Data:** read straight from the public volumes (supervoxels at 32 nm, EM
  at 8 nm), plus the chunkedgraph and L2 cache using the layer's own CAVE
  login.
- **Model:** runs on the GPU (WebGPU) or, without one, on the CPU (WASM).
- **Cache:** the model bytes and every result are kept in IndexedDB.
- **Verdicts:** go to the same CAVE decisions tables. Results carry
  `dataset` = the datastack, and `src=browser_<root>_<stamp>`.

Code: `src/layer/segmentation/twig_inference/`, with `twig_selector_tab.ts`
and `twig_local_results.ts` beside it.

| file | what it does |
|---|---|
| `datasets.ts` | the per-dataset sources (from twigcapture `configs/*.yaml`) and filter values |
| `precomputed_reader.ts` | exact cutouts from sharded or unsharded precomputed volumes (compressed_segmentation, jpeg, raw). gs:// is read through the GCS JSON API, for CORS |
| `volume_ops.ts` | 1:1 port of `touching_labels_center`, `find_contact_sites`, `supervoxel_near`, the zoom(order=0) mask upsampling and `organelle_features` |
| `pipeline.ts` | `run_twigcapture_v2.py`'s region loop; writes the runner's result schema |
| `model.ts` | the runtimes: our WebGPU network (checked against ONNX Runtime once per model and adapter), else ONNX Runtime Web (from jsDelivr, pinned) on WebGPU or WASM; model cache |
| `gpu/` | the WebGPU r3d_18, adapted from Coda (Philipp Schlegel, MIT): ONNX reader, plan, WGSL, buffers and submits |

**Level.** Level 1 suggests merges onto the object. Level N+1 treats the
level-N suggestions with p ≥ the chain threshold (default 0.75) as already
merged, by relabelling them as the object, as the orphan-first runner does.
It then searches around them. Those rows carry `hop` and `via_root` (the
suggestion they attach to; merge that one first). The tab shows a "level N"
chip, and the sink side is labelled "via".

**Checks:**

- `volume_ops.spec.ts` compares against the runner's own Python
  (`scripts/gen_volume_ops_fixture.py` execs its functions under scipy). On
  1,363 contact blobs everything matches exactly: the neighbours, the
  rejections, the blob sizes and centroids, the supervoxel_near voxels and
  the zoom indices.
  - numpy's argsort is not stable above 16 elements, so blobs of equal size
    can be listed in another order. That only matters at the 8-site cutoff.
- The BANC cutouts are identical to CloudVolume's: all 79,660 distinct
  supervoxels in a 200×200×120 region, the sampled voxels, and the EM mean
  and zero fraction (so JPEG decoding is exact).
- In the viewer, the model gives exactly PyTorch's p on the reference patches:
  - WebGPU (Apple M-series): 82 ms/patch with ONNX Runtime, 41 ms/patch with
    our own WebGPU network (above, max |dp| 1.2e-7);
  - CPU (WASM): 3.1 s/patch.

**Differences from the cluster runner:**

- No synapse-density gate (it needs a grid that exists only on the
  cluster), so every non-backbone L2 bin is searched. Use "max regions" to
  cap it.
- BANC EM is the public 8 nm mip, which is downsampled. The runner instead
  takes every second voxel of the 4 nm zarr.
- No synapse counts.

**CPU.** Runs single-threaded, because the page is not cross-origin isolated:
COOP/COEP would break CAVE's login popup. The UI stalls during each patch,
because ORT's worker proxy cannot start from the CDN script. Bundling
onnxruntime-web would allow both. On CPU, set sites/pair to 1 and keep
max regions small.

**End-to-end runs (2026-11-24, Apple M-series, Chrome, WebGPU):**

| dataset | root | regions | patches | time | result |
|---|---|---|---|---|---|
| BANC | 720575941598365041 | 15 (capped) | ~1,080 | 3.5 min | 108 suggestions; opens in Twig Capture; verdict buttons live |
| BANC, level 2 | 720575941598365041 | 6 + 4 (capped) | 859 | 146 s | level 2 searched around the level-1 suggestions (see note) |
| Aedes | 648518347555976042 | 36 (all) | 1,915 | 354 s | 427 suggestions, `dataset` wclee_aedes_brain |
| CRANTb | (needs the zetta login) | | | | EM and supervoxel reads identical to CloudVolume |

- Results are cached in IndexedDB and survive a reload. Re-adding a root
  that has a result at this level or deeper shows it as done.
- **Decisions:** the BANC `twig_capture_decisions` read works with the
  layer's token (HTTP 200). A verdict on a browser row goes through the same
  `DecisionStore.post`, with `src=browser_<root>_<stamp>.json`. No test row
  was written.
- **Against the cluster (Aedes 648518347555976042):** the cluster searched
  2 regions (5 L2 nodes, after its synapse gate), the browser 36 (46 L2
  nodes, no gate). Of the cluster's 25 candidates, 18 are in the browser
  result. Their p agree only loosely, because the checkpoints differ: the
  cluster run used `20260920_143447_aedes/ckpt_0016`, the browser model is
  the joint Aedes+BANC fine-tune. The other 7 were not rejected. The browser
  now logs pairs scored at or below the report threshold as
  `below_threshold`, so whether they were scored low can be checked.
- **Level 2, honestly labelled:** a row's `hop` now says what its sink
  touches. 1 means the neuron, even when the row was found while searching a
  later level (that level's windows overlap the neuron wherever the level-1
  region cap stopped short). Higher means a suggestion, and then `via_root`
  names it. On the test neuron all accepted level-1 suggestions were tiny
  pieces lying against the neuron, so no true second-order rows came up.
- **Site claims are released** when a site's EM is unusable (em_overflow,
  em_zero_frac), as the runner does, so another region can still score it.

**Several neurons at once.** The Selector queues and predicts any number of
roots in turn. Tick finished ones and press **Open ticked together**: Twig
Capture lists all their candidates as one list (`members` on the combined
result). The graph check, decisions and **Stage all → Merge staged** then
work per neuron. Each neuron's root is established from its own level-1
contacts, and merges run neuron by neuron, each against its own advancing
root.

**Bug reports.** **Copy diagnostics** (always shown), and **Copy report** on
a failed item, put a plain-text block on the clipboard. It holds the
browser, OS, CPU threads and memory, the WebGPU adapter (vendor,
architecture, f16, buffer limits, fallback adapter), cross-origin isolation,
the ORT version, the backend actually used, the settings, the error with its
stack, and the last 60 run events (model start, fallbacks, per-region
failures with stacks). No tokens and no URL hash. If the clipboard is
refused, the report opens in a text box instead. In "auto" mode a WebGPU
session that fails to start falls back to the CPU and logs why.

**Deploying (radagast).** One build serves three folders, each with its own
`default_state.json` (in `deploy/<dataset>/`). A link with no state opens
that folder's dataset, with the Twig Capture catalog of its newest model's
results (`twigcapture_<dataset>/results/`, see README "Results folders"). See `deploy/deploy.sh`. The default
and remembered catalog are now per dataset, so a BANC layer no longer opens
the Aedes catalog. CRANTb's cluster results (`twig_capture/crantb_v2.2`)
are not served on catmaid2 yet; `catmaid2/twigcapture_crantb/` currently
falls through to WEBKNOSSOS.

## How to test (browser)

Page: https://radagast.hms.harvard.edu/twigcapture_dev/bench.html
(the copy in this folder is `bench.html`).

The radagast folder `/twigcapture_dev/` is
`/n/data3_vast/hms/neurobio/htem2/twig_capture/dev/` on O2. It also holds:

- the `.onnx` files (fp32, fp16, fp16pad, fp32pad);
- `ref_input_2x3x18x128x128_f32.bin` (2 patches, float32);
- `ref_output.json` (PyTorch's p for those patches).

The model files are not in git (too large); rebuild them with the scripts
below.

The page has one button per backend and model: WebGPU fp16, WebGPU fp16
explicit padding, WebGPU fp32, WebGPU fp32 explicit padding, WASM fp16,
WASM fp32. Graph optimisation is off for every `*pad` model. Each run logs:

- the GPU adapter;
- the model download time and the session start time;
- p for both reference patches against PyTorch (green if |dp| < 1e-3);
- ms/patch at batch sizes 1, 4 and 8, converted to minutes per
  1,500-candidate fragment.

`?ort=<version>` picks the ONNX Runtime Web release (default 1.22.0). Use a
hard reload (Cmd+Shift+R) after a model file changes.

Not set up yet:

- `crossOriginIsolated` is false on radagast, so WASM runs single-threaded.
  Multithreaded WASM needs COOP/COEP headers. Try it only if WebGPU fails,
  because COEP can block cross-origin bucket loads.

## Rebuilding the model files (O2)

Use the venv `/n/data3_vast/hms/neurobio/htem2/users/ab714/envs/tc_onnx`
(system site packages from the Twig Capture env, plus onnx, onnxruntime and
onnxconverter-common). Run on a compute node: the login node caps memory at
2 GiB.

```
python scripts/export_local_onnx.py <ckpt.pth> <out_fp32.onnx>       # export + torch check
python scripts/make_bench_assets.py <fp32.onnx> <ckpt.pth> <dev dir> # fp16 + reference patches
python scripts/explicit_pad.py <dev>/local_fp16.onnx <dev>/local_fp16pad.onnx
```

The scripts import `model_utils` from the twigcapture checkout
(`/n/data3_vast/hms/neurobio/htem2/users/ab714/twigcapture/scripts`).

### Headless, on O2 (`headless/`)

`headless/bench.mjs` drives the same page with Playwright and Chromium and
prints its log. **Never run it on a login node**: RC kills Chrome there.
Use `srun -p interactive` or `sbatch headless/bench.sh`, which asks
gpu_requeue (account wilson) for one GPU and starts Chromium with Vulkan
WebGPU flags. `bench.sh` runs from `$BENCH_DIR` (default
`~/jobs/browser_bench`), which must hold `bench.mjs` and
`node_modules/playwright-core` (`npm i playwright-core`). The Chromium
build is `~/.cache/ms-playwright/chromium-1243`.

- The first argument is the list of buttons to click, as `ep:model` pairs,
  e.g. `webgpu:fp16pad,wasm:fp16`. `bench.sh` takes it from `$WHICH`.
- `STOP=<regex>` ends each run early, e.g. `STOP='FAILED|patch 1:'` skips the
  timing loop, which is very slow on SwiftShader.
- Without a GPU, Chromium falls back to SwiftShader, which has no f16, so it
  can only check fp32 models.

## Next steps

1. Copy `local_fp32c2d.onnx` to the radagast dev folder (`DEFAULT_MODEL_URL`),
   then deploy with `deploy/deploy.sh` and check the three links.
2. Serve CRANTb's results (`twig_capture/crantb_v2.2`) as
   `catmaid2/twigcapture_crantb/` with an `index.json`, and run CRANTb end to
   end after a zetta login.
3. Optional synapse gate in the browser: serve each dataset's 500 nm density
   grid (`data_files/<ds>/synapse_counts_500nm.npy`) and apply
   `min_synapses_within` as the runner does. That makes runs faster and
   region-for-region comparable with the cluster.
4. Bundle onnxruntime-web, so the CPU path can use ORT's worker proxy (the UI
   stays responsive) and, with COOP/COEP, threads.

## Also on this branch (viewer changes since `crantb` fed5bf0)

- **Two new Twig Capture verdicts:**
  - `new_neuron`: that cell is a whole neuron of its own. Counts as no_merge
    in training.
  - `split_needed`: that cell contains a false merge. Left out of training.

  Each names one side of the pair: `tag2 ...;cell=<current root>;side=sink|source`.
  Each row gets a line per cell (the cell's role and id, then its buttons):
  in neuron-first results the roles are neuron and candidate; in orphan-first
  results they are seed and candidate. `cell=` is resolved from the side's
  supervoxel when you click, so it is the current root, not the root at
  inference.
- **Orphan-first result files** (`orph_<seed>_*.json`) now open: the root is
  read from `seed_id`. Their All-neurons rows recognise the seed.
- **Graphene segmentation layers get Spelunker's shortcuts:** M merge,
  C multicut, F find path. A key the link or another layer has already bound
  is left alone.

Companion work in `htem/twigcapture` (branch `banc-fragments`):

- the orphan-first BANC runner;
- `make_orph_index.py` writes `candidates.json.gz`;
- new_neuron is mapped to no_merge in the training scripts.
