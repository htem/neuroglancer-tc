import json, sys, numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
sys.path.insert(0, sys.argv[1])
# import the runner's own functions without running its main
import importlib.util
spec = importlib.util.spec_from_file_location("v2", sys.argv[1] + "/run_twigcapture_v2.py")
src = open(sys.argv[1] + "/run_twigcapture_v2.py").read()
# exec only the pure functions we need
# CONTACT_GAP is the runner's contact_gap_voxels (gap-merges branch): each
# volume is run at 0 (the old touching-only search) and again at 1.
ns = {"np": np, "ndimage": ndimage, "binary_dilation": ndimage.binary_dilation, "CONNECTIVITY": 3, "CONTACT_GAP": 0, "zoom": zoom}
import re, textwrap
for name in ["touching_labels_center", "candidate_bbox", "create_contact_mask", "find_contact_sites", "supervoxel_near"]:
    m = re.search(r"^def %s\(.*?(?=^def |^class |\Z)" % name, src, re.S | re.M)
    exec(m.group(0), ns)
rng = np.random.default_rng(0)
cases = []
vols = []
for t in range(4):
    shape = (40, 36, 30)
    # blobby labels: smooth noise -> watershed-ish regions
    seeds = rng.integers(1, 12, size=(8, 8, 6))
    lab = np.kron(seeds, np.ones((5, 5, 5), dtype=int))[:shape[0], :shape[1], :shape[2]]
    noise = rng.random(shape) < 0.08
    lab[noise] = rng.integers(0, 12, size=noise.sum())
    lab = lab.astype(np.uint64)
    vols.append(lab)
for gap, lab in [(g, v) for g in (0, 1) for v in vols]:
    ns["CONTACT_GAP"] = gap
    shape = lab.shape
    target = int(lab[20, 18, 15]) or 1
    keep, _, stats, rejects = ns["touching_labels_center"](lab, target, border=5, min_contact_voxels=1, max_contact_voxels=1500, min_segment_voxels=22, verbose=False)
    sites = {}
    near = {}
    for k in keep.tolist():
        ss = ns["find_contact_sites"](lab, target, int(k), min_voxels=10, max_sites=1000)
        sites[int(k)] = [[int(v), [float(x) for x in c]] for v, c in ss]
        if ss:
            c = np.array(ss[0][1]).astype(int)
            sv, pt = ns["supervoxel_near"](lab, lab, c, target)
            near[int(k)] = None if pt is None else [int(x) for x in pt]
    # masks: zoom of a 32x32x18 crop
    crop = lab[2:34, 2:34, 6:24]
    m = np.zeros(crop.shape, np.int8); m[crop == target] = 1
    up = zoom(m, (4.0, 4.0, 1.0), order=0)  # (x,y,z) 128x128x18
    cases.append({
        "gap": gap, "shape": list(shape), "lab": lab.transpose(2, 1, 0).ravel().tolist(), "target": target,
        "keep": {str(k): stats[int(k)] for k in keep.tolist()},
        "rejects": {str(k): list(v) for k, v in rejects.items()},
        "sites": {str(k): v for k, v in sites.items()},
        "near": {str(k): v for k, v in near.items()},
        "mask_x_profile": up[:, 0, 0].astype(int).tolist() if False else [int(i) for i in np.rint(zoom(np.arange(32, dtype=float), 4.0, order=0))],
    })
json.dump(cases, open(sys.argv[2], "w"))
print("cases", len(cases), [len(c["keep"]) for c in cases], [sum(len(v) for v in c["sites"].values()) for c in cases])
