"""Score the reference patches with onnxruntime (CPU) and compare p with PyTorch.

Usage: python check_onnx.py <dev dir> <model.onnx> [<model.onnx> ...]
The dev dir holds ref_input_2x3x18x128x128_f32.bin and ref_output.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

dev = Path(sys.argv[1])
x = np.fromfile(dev / "ref_input_2x3x18x128x128_f32.bin", np.float32).reshape(2, 3, 18, 128, 128)
p_torch = np.array(json.loads((dev / "ref_output.json").read_text())["p_torch"])
for path in sys.argv[2:]:
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    s.run(None, {"x_local": x})
    t = time.perf_counter()
    logits = s.run(None, {"x_local": x})[0]
    ms = (time.perf_counter() - t) * 1000 / 2
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e[:, 1] / e.sum(1)
    print(f"{Path(path).name}: p={np.round(p, 4)} max|dp|={np.abs(p - p_torch).max():.1e} cpu {ms:.0f} ms/patch")
