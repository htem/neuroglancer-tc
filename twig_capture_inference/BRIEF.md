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
2. Score real-sized patches in the browser: correctness and speed. **In progress.**
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

1. Run WebGPU on `local_fp16pad.onnx`. If it passes, record ms/patch, then
   try fp32-pad if fp16 numerics drift.
2. If WebGPU fails on another operator, rewrite it in `explicit_pad.py` the
   same way. If 3D Conv is too slow or unsupported, express each Conv3d as a
   sum of Conv2d over kernel depth.
3. Load the model in the viewer from a URL and score one real candidate.
   Build its patch from neuroglancer's own EM and segmentation chunks, check
   p against that candidate's `prob_merge` in a result JSON, then move on to
   step 3 of the plan.

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
