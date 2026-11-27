import sys, json, numpy as np, onnx, onnxruntime as ort, torch
sys.path.insert(0, "/n/data3_vast/hms/neurobio/htem2/users/ab714/twigcapture/scripts")
from model_utils import LocalOnlyClassifier
from onnxconverter_common import float16
src, ckpt, dev = sys.argv[1], sys.argv[2], sys.argv[3]
m16 = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True)
onnx.save(m16, f"{dev}/local_fp16.onnx")
import shutil; shutil.copy(src, f"{dev}/local_fp32.onnx")
rng = np.random.default_rng(1)
xs = np.zeros((2, 3, 18, 128, 128), np.float32)
xs[:, 0] = np.clip(rng.normal(0, 0.4, xs[:, 0].shape), -1, 1)
yy, xx = np.mgrid[:128, :128]
xs[0, 1] = (xx < 64)[None]; xs[0, 2] = (xx >= 66)[None]
xs[1, 1] = (np.hypot(xx - 64, yy - 64) < 30)[None]; xs[1, 2] = (np.hypot(xx - 64, yy - 64) > 33)[None]
xs.tofile(f"{dev}/ref_input_2x3x18x128x128_f32.bin")
m = LocalOnlyClassifier(3, 2); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
with torch.no_grad(): pt = torch.softmax(m(torch.from_numpy(xs)), 1)[:, 1].numpy()
so = ort.SessionOptions(); so.intra_op_num_threads = 8
s16 = ort.InferenceSession(f"{dev}/local_fp16.onnx", so, providers=["CPUExecutionProvider"])
l16 = s16.run(None, {"x_local": xs})[0]; p16 = np.exp(l16[:, 1]) / np.exp(l16).sum(1)
s32 = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
import time; t=time.time(); s32.run(None, {"x_local": xs}); dt=(time.time()-t)/2
json.dump({"p_torch": pt.tolist(), "p_onnx_fp16_cpu": p16.tolist(), "ort_cpu_ms_per_patch_8threads": round(dt*1000)}, open(f"{dev}/ref_output.json", "w"))
print(open(f"{dev}/ref_output.json").read())
