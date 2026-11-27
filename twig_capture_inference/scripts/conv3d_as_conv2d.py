"""Rewrite the r3d_18 graph so every 3D convolution runs as 2D convolutions.

Why: ONNX Runtime Web (1.22) runs 3D Conv on WebGPU with a naive kernel
(`Conv3DNaive`), about 3.7 s/patch on an Apple M-series GPU, slower than CPU.
Its 2D Conv uses tuned matmul kernels.

How: activations are kept in (N, D, C, H, W) layout from the input to the
pooling, so a tensor reshaped to (N*D, C, H, W) is a batch of 2D images.
For a Conv3d with kernel (kd, kh, kw), stride (sd, sh, sw), pads (pd, ph, pw):
  1. pad depth only (axis 1) with zeros, if pd > 0;
  2. for each kernel-depth tap k, slice depth planes k, k+sd, ... (Do of them);
  3. mode "concat": stack the kd slices along channels and run ONE Conv2d whose
     weight is W reordered to (Co, kd*Ci, kh, kw);
     mode "sum": run one Conv2d per tap with W[:, :, k] and add the results;
  4. reshape back to (N, Do, Co, Ho, Wo).
H and W padding is left to the Conv2d itself. Relu and the residual Add are
layout-free. GlobalAveragePool becomes ReduceMean over axes (1, 3, 4).

No Pad feeds a Conv directly, so the session can use full graph optimisation.

Usage: python conv3d_as_conv2d.py <in.onnx> <out.onnx> [concat|sum]
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "concat"
assert mode in ("concat", "sum"), mode

m = onnx.load(src)
g = m.graph
init = {i.name: i for i in g.initializer}
inp = g.input[0]
dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
assert dims[1:] == [3, 18, 128, 128], dims  # (n, C, D, H, W)

new_nodes, new_inits, used = [], [], set()
# spatial (D, H, W) and channels of every NDCHW tensor, by name
shape = {}
uid = [0]


def name(stem):
    uid[0] += 1
    return f"{stem}_{uid[0]}"


def const(arr, stem):
    n = name(stem)
    new_inits.append(numpy_helper.from_array(np.asarray(arr), n))
    return n


def node(op, ins, stem, **attrs):
    out = name(stem)
    new_nodes.append(helper.make_node(op, ins, [out], name=out, **attrs))
    return out


# input (N, C, D, H, W) -> (N, D, C, H, W)
x0 = node("Transpose", [inp.name], "in_ndchw", perm=[0, 2, 1, 3, 4])
shape[x0] = (3, 18, 128, 128)  # C, D, H, W
rename = {inp.name: x0}

for n in g.node:
    ins = [rename.get(i, i) for i in n.input]
    if n.op_type == "Conv":
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        W = numpy_helper.to_array(init[ins[1]]).astype(np.float32)
        B = numpy_helper.to_array(init[ins[2]]) if len(ins) > 2 else None
        used.update(ins[1:])
        co, ci, kd, kh, kw = W.shape
        sd, sh, sw = a["strides"]
        pd, ph, pw = a["pads"][:3]
        assert list(a["pads"][3:]) == [pd, ph, pw] and a.get("group", 1) == 1
        assert list(a.get("dilations", [1, 1, 1])) == [1, 1, 1]
        x = ins[0]
        c, D, H, Wd = shape[x]
        assert c == ci
        Do = (D + 2 * pd - kd) // sd + 1
        Ho = (H + 2 * ph - kh) // sh + 1
        Wo = (Wd + 2 * pw - kw) // sw + 1
        if pd:
            x = node("Pad", [x, const(np.array([0, pd, 0, 0, 0, 0, pd, 0, 0, 0], np.int64), "dpads")],
                     "dpad", mode="constant")
        Dp = D + 2 * pd
        conv_attrs = dict(kernel_shape=[kh, kw], pads=[ph, pw, ph, pw], strides=[sh, sw])

        def depth_slice(k):
            end = k + (Do - 1) * sd + 1
            if k == 0 and sd == 1 and end == Dp:
                return x
            return node("Slice", [x, const(np.array([k], np.int64), "st"), const(np.array([end], np.int64), "en"),
                                  const(np.array([1], np.int64), "ax"), const(np.array([sd], np.int64), "sp")],
                        "dslice")

        bias = [const(B.astype(np.float32), "b")] if B is not None else []
        if mode == "concat" or kd == 1:
            sl = [depth_slice(k) for k in range(kd)]
            xc = sl[0] if kd == 1 else node("Concat", sl, "dcat", axis=2)
            to4 = const(np.array([-1, kd * ci, H, Wd], np.int64), "shape4")
            w2 = W.transpose(0, 2, 1, 3, 4).reshape(co, kd * ci, kh, kw)
            y = node("Conv", [node("Reshape", [xc, to4], "to4"), const(w2, "w")] + bias, "conv2d", **conv_attrs)
        else:
            to4 = const(np.array([-1, ci, H, Wd], np.int64), "shape4")
            parts = []
            for k in range(kd):
                xi = node("Reshape", [depth_slice(k), to4], "to4")
                parts.append(node("Conv", [xi, const(np.ascontiguousarray(W[:, :, k]), "w")] + (bias if k == 0 else []),
                                  "conv2d", **conv_attrs))
            y = node("Sum", parts, "dsum")
        y = node("Reshape", [y, const(np.array([-1, Do, co, Ho, Wo], np.int64), "shape5")], "to5")
        shape[y] = (co, Do, Ho, Wo)
        rename[n.output[0]] = y
    elif n.op_type in ("Relu", "Add") and ins[0] in shape:
        out = node(n.op_type, ins, n.op_type.lower())
        shape[out] = shape[ins[0]]
        rename[n.output[0]] = out
    elif n.op_type == "GlobalAveragePool":
        # NDCHW: average over D, H, W -> (N, 1, C, 1, 1); the following Flatten(axis=1) gives (N, C)
        out = node("ReduceMean", ins, "gap", axes=[1, 3, 4], keepdims=1)
        rename[n.output[0]] = out
    else:
        assert not any(i in shape for i in ins), f"unhandled op on a 5D tensor: {n.op_type}"
        new_nodes.append(helper.make_node(n.op_type, ins, list(n.output), name=n.name,
                                          **{x.name: helper.get_attribute_value(x) for x in n.attribute}))

keep = [i for i in g.initializer if i.name not in used]
del g.node[:]
g.node.extend(new_nodes)
del g.initializer[:]
g.initializer.extend(keep + new_inits)
for o in g.output:
    assert o.name not in rename
del g.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, dst)
print(f"{mode}: {sum(1 for x in new_nodes if x.op_type == 'Conv')} Conv2d nodes -> {dst}")
