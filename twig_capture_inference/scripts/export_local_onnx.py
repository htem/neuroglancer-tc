"""Export a LocalOnlyClassifier checkpoint to ONNX, for in-browser scoring.

    python scripts/export_local_onnx.py <ckpt.pth> <out.onnx>

Input  x_local float32 [N, 3, 18, 128, 128]: EM at 8 nm scaled to [-1, 1],
       then the two segment masks (neuron side, candidate side), z,y,x order,
       exactly as run_twigcapture_v2.py builds it.
Output logits float32 [N, 2]; softmax(logits)[:, 1] is p(merge).
Checks onnxruntime against PyTorch on random and mask-like inputs and prints
the largest probability difference and CPU timing.
"""
import sys, time
import numpy as np, torch, onnxruntime as ort
sys.path.insert(0, __import__("os").path.dirname(__file__))
from model_utils import LocalOnlyClassifier

ckpt, out = sys.argv[1], sys.argv[2]
m = LocalOnlyClassifier(in_channels=3, num_classes=2)
m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
x = torch.zeros(1, 3, 18, 128, 128)
torch.onnx.export(m, x, out, input_names=["x_local"], output_names=["logits"],
                  dynamic_axes={"x_local": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                  dynamo=False)
rng = np.random.default_rng(0)
xs = np.empty((8, 3, 18, 128, 128), np.float32)
xs[:, 0] = rng.uniform(-1, 1, xs[:, 0].shape)
for i in range(8):           # blob-like masks: two half-spaces with a seam
    yy, xx = np.mgrid[:128, :128]
    a = (xx + rng.integers(-20, 20) * (yy / 128)) < 64
    xs[i, 1] = a[None]; xs[i, 2] = ~a[None] & (rng.random((128, 128)) < 0.9)
with torch.no_grad():
    pt = torch.softmax(m(torch.from_numpy(xs)), 1)[:, 1].numpy()
s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
t = time.time(); lo = s.run(None, {"x_local": xs})[0]; dt = time.time() - t
po = np.exp(lo[:, 1]) / np.exp(lo).sum(1)
print("p torch ", np.round(pt, 4)); print("p onnx  ", np.round(po, 4))
print(f"max |dp| {np.abs(pt - po).max():.2e}; onnxruntime CPU {dt/8*1000:.0f} ms/patch ({torch.get_num_threads()} threads)")
