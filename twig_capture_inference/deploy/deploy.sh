#!/usr/bin/env bash
# Deploy the Twig Capture viewer (this branch's build) to the dataset
# folders served by radagast:
#
#   https://radagast.hms.harvard.edu/twigcapture_banc/
#   https://radagast.hms.harvard.edu/twigcapture_aedes/
#   https://radagast.hms.harvard.edu/twigcapture_crantb/
#   https://radagast.hms.harvard.edu/twigcapture_fanc/
#
# FANC's cluster results are results_fanc -> twig_capture/fanc_v1 (297
# proofread neurons, joint model); like the others it opens on Twig Capture.
#
# One build, copied into each folder with that folder's default_state.json
# (a link with no state opens the folder's dataset; see
# twigCaptureDefaultState in src/ui/default_viewer_setup.ts).
#
# Usage (from the repo root, on a machine that can ssh to O2):
#   twig_capture_inference/deploy/deploy.sh <user@host> <remote_base>
#
# remote_base: the O2 directory whose twigcapture_<name>/ subfolders radagast
# serves as https://radagast.hms.harvard.edu/twigcapture_<name>/ (the parent
# of the folder that holds today's twigcapture_crantb build). The previous
# build in each folder is kept as twigcapture_<name>.prev for rollback.
set -euo pipefail

HOST="${1:?usage: deploy.sh <user@host> <remote_base>}"
BASE="${2:?usage: deploy.sh <user@host> <remote_base>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$ROOT/dist/twigcapture"

cd "$ROOT"
rm -rf "$OUT"
node --experimental-strip-types ./build_tools/cli.ts build --output "$OUT"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
for k in banc aedes crantb fanc; do
  mkdir -p "$STAGE/twigcapture_$k"
  cp -R "$OUT/." "$STAGE/twigcapture_$k/"
  cp "$HERE/$k/default_state.json" "$STAGE/twigcapture_$k/"
done

# One ssh session (one Duo prompt): upload a tarball, then swap folders.
tar -C "$STAGE" -czf "$STAGE/site.tgz" twigcapture_banc twigcapture_aedes twigcapture_crantb twigcapture_fanc
scp "$STAGE/site.tgz" "$HOST:/tmp/twigcapture_site.$$.tgz"
ssh "$HOST" bash -s -- "$BASE" "/tmp/twigcapture_site.$$.tgz" <<'EOF'
set -euo pipefail
BASE="$1"; TGZ="$2"
mkdir -p "$BASE/.incoming"
tar -C "$BASE/.incoming" -xzf "$TGZ"
for k in banc aedes crantb fanc; do
  d="$BASE/twigcapture_$k"
  rm -rf "$d.prev"
  [ -d "$d" ] && mv "$d" "$d.prev"
  mv "$BASE/.incoming/twigcapture_$k" "$d"
  chmod -R a+rX "$d"
  # The newest model's results for this dataset, served at
  # twigcapture_<k>/results/ (index.json, candidates.json.gz, neurons.json
  # and links to the result files; see README "Results folders").
  [ -e "$BASE/results_$k" ] && ln -sfn "$BASE/results_$k" "$d/results"
done
rm -rf "$BASE/.incoming" "$TGZ"
ls -ld "$BASE"/twigcapture_*
EOF
echo "Deployed. Check:"
for k in banc aedes crantb fanc; do echo "  https://radagast.hms.harvard.edu/twigcapture_$k/"; done
