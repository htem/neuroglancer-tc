"""Make the r3d_18 graph run on ONNX Runtime Web's WebGPU backend. Same numbers.
- every Conv with non-zero pads -> Pad(constant 0) -> Conv(pads=0): WebGPU's 3D
  Conv rejects the padded form (load with graph optimisation off, or ORT fuses
  them back);
- GlobalAveragePool over 3 spatial axes -> ReduceMean(axes 2,3,4, keepdims):
  WebGPU pooling supports only 2 spatial axes."""
import sys, onnx, numpy as np
from onnx import helper, numpy_helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph; new = []
for n in g.node:
    if n.op_type == "Conv":
        pads = next((list(a.ints) for a in n.attribute if a.name == "pads"), None)
        if pads and any(pads):
            k = len(pads) // 2
            full = [0, 0] + pads[:k] + [0, 0] + pads[k:]
            pn = f"{n.name}_pads"; po = f"{n.input[0]}__padded_{len(new)}"
            g.initializer.append(numpy_helper.from_array(np.array(full, np.int64), pn))
            new.append(helper.make_node("Pad", [n.input[0], pn], [po], mode="constant", name=f"{n.name}_Pad"))
            n.input[0] = po
            for a in n.attribute:
                if a.name == "pads": a.ints[:] = [0] * len(pads)
    if n.op_type == "GlobalAveragePool":
        # opset < 18: axes is an attribute
        n = helper.make_node("ReduceMean", [n.input[0]], list(n.output), axes=[2, 3, 4], keepdims=1, name=f"{n.name}_ReduceMean")
    new.append(n)
del g.node[:]; g.node.extend(new)
onnx.checker.check_model(m); onnx.save(m, dst)
