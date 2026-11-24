#!/bin/bash
# Build the in-browser model from a twigcapture checkpoint, check it, and
# publish it on radagast under its own versioned URL.
#
#   cd <neuroglancer-tc>/twig_capture_inference      # on O2
#   sbatch scripts/build_browser_model.sh <ckpt.pth> [name]
#
# e.g. Kevin's newest joint model:
#   sbatch scripts/build_browser_model.sh \
#     /n/data3_vast/hms/neurobio/htem2/users/kd193/automergeproof/models/models_local/joint_finetune_20261123_105038/local_classifier_ckpt_0008.pth
#
# name defaults to <run dir>_ep<N> (joint_finetune_20261123_105038_ep8).
# Publishes  twig_capture/dev/models/<name>/local_fp32c2d.onnx  served as
#   https://radagast.hms.harvard.edu/twigcapture_dev/models/<name>/local_fp32c2d.onnx
# plus MODEL.txt (checkpoint, sha256, the torch-vs-ONNX check). It never
# overwrites a published model: the browser caches models by URL, so a new
# model must have a new URL or browsers keep scoring with the old one.
#
# Steps (see BRIEF.md "The model" for why each exists):
#   1. export_local_onnx.py (twigcapture): checkpoint -> fp32 ONNX, checked
#      against PyTorch on random inputs;
#   2. make_bench_assets.py: two reference patches and PyTorch's p on them;
#   3. conv3d_as_conv2d.py concat: every Conv3d as Conv2d (WebGPU is ~40x
#      faster on it, and exact);
#   4. check_onnx.py: the rewritten model against PyTorch on the reference
#      patches. Fails the job if max|dp| > 1e-4.
#SBATCH -p short
#SBATCH -t 0-01:00
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -J tc_browser_model
#SBATCH -o tc_browser_model_%j.out
set -euo pipefail

CKPT=${1:?usage: sbatch scripts/build_browser_model.sh <ckpt.pth> [name]}
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }
EP=$(basename "$CKPT" .pth | sed -E 's/.*_0*([0-9]+)$/ep\1/')
NAME=${2:-$(basename "$(dirname "$CKPT")")_$EP}

HTEM=/n/data3_vast/hms/neurobio/htem2
PY=${PY:-$HTEM/users/ab714/envs/tc_onnx/bin/python}
TWIGCAPTURE=${TWIGCAPTURE:-$HTEM/users/ab714/twigcapture}
HERE=${SLURM_SUBMIT_DIR:-$(pwd)}/scripts      # sbatch runs a copy of this file
PUBLISH=${PUBLISH:-$HTEM/twig_capture/dev/models}
OUT=$PUBLISH/$NAME
URL=https://radagast.hms.harvard.edu/twigcapture_dev/models/$NAME/local_fp32c2d.onnx

[ -f "$HERE/conv3d_as_conv2d.py" ] || {
  echo "run from twig_capture_inference/ (no $HERE/conv3d_as_conv2d.py)"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists; pick another name (URLs are never reused)"; exit 1; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/tc_model_XXXX")
trap 'rm -rf "$WORK"' EXIT
echo "== $NAME from $CKPT  ($(date))"

"$PY" "$TWIGCAPTURE/scripts/export_local_onnx.py" "$CKPT" "$WORK/local_fp32.onnx"
"$PY" "$HERE/make_bench_assets.py" "$WORK/local_fp32.onnx" "$CKPT" "$WORK" >/dev/null
"$PY" "$HERE/conv3d_as_conv2d.py" "$WORK/local_fp32.onnx" "$WORK/local_fp32c2d.onnx" concat
CHECK=$("$PY" "$HERE/check_onnx.py" "$WORK" "$WORK/local_fp32c2d.onnx")
echo "$CHECK"
DP=$(sed -nE 's/.*max\|dp\|=([0-9.e+-]+).*/\1/p' <<<"$CHECK")
"$PY" -c "import sys; sys.exit(0 if float('$DP') <= 1e-4 else 1)" || {
  echo "FAILED: rewritten model differs from PyTorch (max|dp| $DP)"; exit 1; }

mkdir -p "$OUT"
cp "$WORK/local_fp32c2d.onnx" "$OUT/"
cat >"$OUT/MODEL.txt" <<EOF
name        $NAME
checkpoint  $CKPT
built       $(date -u +%Y-%m-%dT%H:%M:%SZ) by $USER
sha256      $(sha256sum "$OUT/local_fp32c2d.onnx" | cut -d' ' -f1)
check       $CHECK
url         $URL
EOF
chmod -R a+rX "$OUT"
echo
echo "published: $URL"
echo "Try it: paste the URL into the Twig Selector's model field."
echo "Make it the default: set DEFAULT_MODEL_URL in"
echo "  src/layer/segmentation/twig_inference/datasets.ts and redeploy."
