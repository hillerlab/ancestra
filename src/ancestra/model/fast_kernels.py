"""CUDA inference fast paths: bitwise-identical Triton dwconv + RMSNorm and one CUDA graph.

G1 replaces ``ancestra._vendor.causal_conv1d.cpp_functions._ref_causal_conv1d`` (the k=4 depthwise causal
conv the SSD scan calls through ``causal_conv1d_fwd_function``); G2 replaces ``RMSNorm.forward``; G3
captures the eager forward for the full-window input shape (1, 6, W) in one CUDA graph. Each installed
function falls back to the original for any input outside its handled case. ``enable`` self-checks every
piece once per process (seeded inputs, ``torch.equal``) and keeps only the pieces that pass.
"""

import os
import sys
import time

import torch

from ancestra._vendor.causal_conv1d import cpp_functions as _cf
from ancestra.model import network as _M

_CHECKED = {}  # process-once outcomes: "dwconv"/"rmsnorm" -> installed callable or None

MAXABS = 2.5  # --compile tolerance: max |compiled - eager| over jointly-finite positions (float32)
AGREE = 0.999  # --compile tolerance: joint argmax(dim 2) agreement


class CompileError(RuntimeError):
    """--compile refused: reason names why. Raised only after the G1/G2 originals are restored."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _compile_route(x, shape):
    """The compiled route: exactly shape (1, 6, W) under CUDA bf16 autocast; every other input runs the G1+G2
    eager model. A module-level function so a CPU test can monkeypatch it."""
    return (tuple(x.shape) == tuple(shape) and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16)


def _dwconv_impl():
    """A Triton replacement for the vendored _ref_causal_conv1d (k 4, silu) that mirrors PyTorch 2.10's native depthwise kernel:
    acc = bias; acc = fma(w_k, x_k, acc) over the in-range taps in tap order (fp32), RTNE to bf16, then eager F.silu. It writes
    [B, L, D] and returns the [B, D, L] view, so the scan's split views are row-major (no B/C .contiguous() copies)."""
    import triton
    import triton.language as tl
    import torch.nn.functional as F

    @triton.jit
    def dwconv4(x_ptr, w_ptr, b_ptr, y_ptr, L, D, sxb, sxd, sxl, syb, syl, syd, BL: tl.constexpr, BD: tl.constexpr):
        pl, pd, pb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        l = pl * BL + tl.arange(0, BL)
        d = pd * BD + tl.arange(0, BD)
        md = d < D
        acc = tl.broadcast_to(tl.load(b_ptr + d, mask=md, other=0.0).to(tl.float32)[None, :], (BL, BD))
        for k in tl.static_range(4):
            pos = l - 3 + k
            m = ((pos >= 0) & (pos < L))[:, None] & md[None, :]
            xv = tl.load(x_ptr + pb * sxb + pos[:, None] * sxl + d[None, :] * sxd, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + d * 4 + k, mask=md, other=0.0).to(tl.float32)
            acc = tl.where(m, tl.fma(tl.broadcast_to(wv[None, :], (BL, BD)), xv, acc), acc)
        tl.store(y_ptr + pb * syb + l[:, None] * syl + d[None, :] * syd, acc.to(tl.bfloat16),
                 mask=(l < L)[:, None] & md[None, :])

    def conv(x, weight, bias, silu_activation):
        assert x.dtype == torch.bfloat16 and torch.is_autocast_enabled("cuda") and weight.shape[1] == 4
        bsz, D, L = x.shape
        w = weight.to(torch.bfloat16).contiguous()            # what autocast hands the native kernel
        b = bias.to(torch.bfloat16)
        y = torch.empty(bsz, L, D, device=x.device, dtype=torch.bfloat16)
        grid = (triton.cdiv(L, 64), triton.cdiv(D, 128), bsz)
        dwconv4[grid](x, w, b, y, L, D, x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2),
                      BL=64, BD=128)
        if silu_activation:
            y = F.silu(y)
        return y.transpose(1, 2)
    return conv


def _rmsnorm_impl():
    """RMSNorm.forward with eager's own reduction: sq = float(x)^2 (one Triton pass), ms = sq.mean(dim) (the same torch
    reduction on the same fp32 values), r = rsqrt(ms + eps) (torch), out = bf16((float(x) * r) * w) (one Triton pass, two
    separate fp32 multiplies as eager). Non-contiguous or non-bf16 inputs run eager."""
    import triton
    import triton.language as tl
    from ancestra.model import network as M
    eager = M.RMSNorm.forward

    @triton.jit
    def square(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        xf = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + i, xf * xf, mask=m)

    @triton.jit
    def scale(x_ptr, r_ptr, w_ptr, y_ptr, n, C, Lr, CHANNEL: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        if CHANNEL:                                            # [B, C, L], norm over C: r [B, 1, L], w [C, 1]
            ri = (i // (C * Lr)) * Lr + i % Lr
            wi = (i // Lr) % C
        else:                                                  # [..., C], norm over the last dim: r [..., 1], w [C]
            ri = i // C
            wi = i % C
        xf = tl.load(x_ptr + i, mask=m, other=0.0).to(tl.float32)
        y = xf * tl.load(r_ptr + ri, mask=m, other=0.0)
        y = y * tl.load(w_ptr + wi, mask=m, other=0.0)
        tl.store(y_ptr + i, y.to(tl.bfloat16), mask=m)

    def forward(self, x):
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            return eager(self, x)
        if self.dim == 1 and x.dim() == 3 and x.is_contiguous():
            chan, C, Lr = True, x.shape[1], x.shape[2]
        elif self.dim == -1 and x.is_contiguous():
            chan, C, Lr = False, x.shape[-1], 1
        elif self.dim == -1 and x.dim() == 3 and x.transpose(1, 2).is_contiguous():   # trunk stream: storage [B, d, T]
            chan, C, Lr = True, x.shape[2], x.shape[1]
        else:
            return eager(self, x)
        n = x.numel()
        sq = torch.empty_like(x, dtype=torch.float32)          # x.float().pow(2)'s strides, so .mean reduces the same way
        square[(triton.cdiv(n, 1024),)](x, sq, n, BLOCK=1024)
        r = torch.rsqrt(sq.mean(self.dim, keepdim=True) + self.eps).contiguous()
        y = torch.empty_like(x)
        scale[(triton.cdiv(n, 1024),)](x, r, self.weight, y, n, C, Lr, CHANNEL=chan, BLOCK=1024)
        return y
    return forward, eager


class Graphed(torch.nn.Module):
    """The eager model captured once into a CUDA graph for one input shape (same kernels, replayed); other shapes run eager."""

    def __init__(self, model, shape, dev):
        super().__init__()
        self.model, self.cfg = model, getattr(model, "cfg", None)
        self.x = torch.zeros(shape, device=dev)
        with torch.cuda.device(dev):  # capture streams are the current-device ones; cross-device capture only missed the check
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad(), torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                for _ in range(2):
                    model(self.x)                              # Triton autotune and cuDNN plans before capture
            torch.cuda.current_stream().wait_stream(s)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g), torch.no_grad(), torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                self.out = model(self.x)

    def forward(self, x):
        if (x.shape != self.x.shape or not torch.is_autocast_enabled("cuda")
                or torch.get_autocast_dtype("cuda") != torch.bfloat16):
            return self.model(x)  # captured under bf16 autocast; any other mode runs eager
        self.x.copy_(x)
        self.g.replay()
        return self.out  # static f32-safe only when .float() copies: it copies only non-f32, so an f32 capture
        # would alias this buffer and the RC replay would overwrite it (_maybe_graph refuses f32 captures)


class Compiled(torch.nn.Module):
    """The G1+G2 model under torch.compile for one input shape (full windows); every other input runs the eager model."""

    def __init__(self, model, shape):
        super().__init__()
        self.model, self.cfg = model, getattr(model, "cfg", None)
        self.shape = tuple(shape)
        self._compiled = torch.compile(model, mode="reduce-overhead")

    def forward(self, x):
        if not _compile_route(x, self.shape):
            return self.model(x)
        # Inductor bakes the flag into the kernels it generates, and dynamo resets it after each compiled frame
        # (torch 2.10 convert_frame.py:337 restores torch.use_deterministic_algorithms(prior), which also writes
        # inductor's flag (torch/__init__.py:1506)), so the patch wraps every call. Never call
        # torch.use_deterministic_algorithms: it would flip the process-global default.
        with torch._inductor.config.patch(deterministic=True):
            return self._compiled(x)


def _note(dw, rn, cg):
    print(f"ancestra: fast kernels dwconv={dw} rmsnorm={rn} cudagraph={cg}", file=sys.stderr, flush=True)


def _is_cuda(device):
    try:
        return torch.device(device).type == "cuda" and torch.cuda.is_available()
    except Exception:
        return False


def _seeded_bf16(gen, shape, dev):
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(dev)


def _seeded_onehot(W, dev):
    """The seeded one-hot (1, 6, W) input both the graph check and enable()'s end-to-end check run."""
    W = int(W)
    gen = torch.Generator(device="cpu").manual_seed(2)
    idx = torch.randint(0, 4, (1, W), generator=gen)
    x = torch.zeros((1, 6, W), dtype=torch.float32)
    x[0, idx[0], torch.arange(W)] = 1.0  # ACGT one-hot; mask row stays 0
    return x.to(dev)


def _install_dwconv(model, dev):
    """Install the Triton causal conv if it matches the original on the model's own depthwise k=4 weights."""
    if _CHECKED.get("dwconv", "missing") == "missing":
        try:
            keep = _check_dwconv(model, dev)
        except Exception:
            keep = None
        _CHECKED["dwconv"] = keep
    fast = _CHECKED["dwconv"]
    if fast is None:
        return "off(check)"
    _cf._ref_causal_conv1d = fast
    return "on"


def _check_dwconv(model, dev):
    orig = _cf._ref_causal_conv1d
    raw = _dwconv_impl()  # raises without Triton; asserts on out-of-scope inputs (the wrapper guards those)

    def conv(x, weight, bias, silu_activation):
        if (x.is_cuda and x.dtype == torch.bfloat16 and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.bfloat16
                and x.dim() == 3 and x.shape[-1] >= 8 and weight.dim() == 2 and weight.shape[1] == 4
                and bias is not None):
            return raw(x, weight, bias, silu_activation)
        return orig(x, weight, bias, silu_activation)

    convs = {}
    for m in model.modules():
        if (isinstance(m, torch.nn.Conv1d) and m.kernel_size == (4,)
                and m.groups == m.in_channels == m.out_channels and m.bias is not None):
            # the scan calls the conv with weight.squeeze(1) [D, 4] (model.py); check that exact form
            convs.setdefault(m.out_channels, (m.weight.detach().squeeze(1), m.bias.detach()))
    if not convs:
        return None
    gen = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
        for D, (w, b) in sorted(convs.items()):
            w, b = w.to(dev), b.to(dev)
            for L in (8, 77, 16384):
                for B in (1, 2):
                    base = _seeded_bf16(gen, (B, L, D), dev)  # contiguous [B, S, D], as ssd_combined passes xBC
                    x = base.transpose(1, 2)  # [B, D, S] view (rearrange "b s d -> b d s") incl. ensure_stride form
                    for silu in (True, False):
                        if not torch.equal(orig(x, w, b, silu), conv(x, w, b, silu)):
                            return None
    return conv


def _install_rmsnorm(model):
    """Install the Triton RMSNorm if it matches eager on every RMSNorm layout the model uses."""
    if _CHECKED.get("rmsnorm", "missing") == "missing":
        try:
            keep = _check_rmsnorm(model)
        except Exception:
            keep = None
        _CHECKED["rmsnorm"] = keep
    fast = _CHECKED["rmsnorm"]
    if fast is None:
        return "off(check)"
    _M.RMSNorm.forward = fast
    return "on"


def _check_rmsnorm(model):
    fast, eager = _rmsnorm_impl()  # raises without Triton

    def forward(self, x):
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            return eager(self, x)
        if self.dim == 1 and x.dim() == 3 and x.is_contiguous():
            pass
        elif self.dim == -1 and (x.is_contiguous()
                                 or (x.dim() == 3 and x.transpose(1, 2).is_contiguous())):
            pass
        else:
            return eager(self, x)
        return fast(self, x)

    mods = [m for m in model.modules() if isinstance(m, _M.RMSNorm)]
    if not mods:
        return None
    gen = torch.Generator(device="cpu").manual_seed(1)
    dev = mods[0].weight.device
    _M.RMSNorm.forward = fast
    try:
        # wide shapes: an fp32 reassociation (e.g. x*(r*w)) flips only ~1e-5 of bf16 outputs, so the
        # check needs millions of elements per layout to catch it; the exact kernel stays bit-identical
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            for m in mods:
                C = m.weight.numel()
                if m.dim == 1:
                    x = _seeded_bf16(gen, (2, C, 4096), dev)
                    if not torch.equal(eager(m, x), m(x)):
                        return None
                elif m.dim == -1:
                    x = _seeded_bf16(gen, (2, 4096, C), dev)
                    if not torch.equal(eager(m, x), m(x)):
                        return None
                    y = _seeded_bf16(gen, (2, C, 4096), dev).transpose(1, 2)
                    assert y.transpose(1, 2).is_contiguous()
                    if not torch.equal(eager(m, y), m(y)):
                        return None
                else:
                    return None
    finally:
        _M.RMSNorm.forward = eager
    return forward


def _maybe_graph(model, dev, window, _ref=None, _x=None):
    """Capture the full-window forward in one CUDA graph if replay matches the unpatched model on a seeded one-hot window.

    ``_ref``/``_x`` carry enable()'s pre-patch reference so the comparison is against the unpatched model, not the
    already-patched one. A captured f32 output would alias the static buffer (``.float()`` copies only non-f32,
    and the RC replay would overwrite it), so it refuses with ``off(check)``.
    """
    try:
        W = int(window)
        shape = (1, 6, W)
        x = _x if _x is not None else _seeded_onehot(W, dev)
        if _ref is None:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                ref = model(x)
        else:
            ref = _ref
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            gm = Graphed(model, shape, dev)
            if any(isinstance(v, torch.Tensor) and v.dtype == torch.float32 for v in gm.out.values()):
                return model, "off(check)"
            got = gm(x)
        if set(got) != set(ref) or any(not torch.equal(ref[k], got[k]) for k in ref):
            return model, "off(check)"
        return gm, "on"
    except Exception:
        return model, "off(check)"


def _dicts_equal(a, b):
    """Bitwise equality of two output dicts."""
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _stable_equal(a, b):
    """Stability of the three compiled calls: bitwise equal, with a NaN counting as equal to a NaN (torch.equal
    is NaN-hostile; the finite-mask check against the reference judges NaNs next)."""
    if set(a) != set(b):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if torch.equal(va, vb):
            continue
        if not (va.shape == vb.shape and va.dtype == vb.dtype and va.is_floating_point()
                and bool((((va == vb) | (torch.isnan(va) & torch.isnan(vb))).all()))):
            return False
    return True


def _enable_compiled(model, device, window, amp, mode):
    """torch.compile the G1+G2 model for full windows only (opt-in --compile): the same G1/G2 install and e2e
    bitwise check as enable(), then one compiled callable checked within tolerance against the unpatched reference (three
    bitwise-equal calls, per-key shape/dtype/finite-mask with max |delta| <= MAXABS, joint argmax agreement >=
    AGREE, and the short window bitwise through the eager route). Any refusal raises CompileError with the G1/G2
    originals restored; G3 is never captured. Two stderr lines."""
    if mode != "reduce-overhead":
        raise CompileError(f"unknown compile mode {mode!r}")
    if not _is_cuda(device):
        raise CompileError("not CUDA")
    if os.environ.get("ANCESTRA_FAST_KERNELS", "").strip().lower() in ("0", "false", "off", "no"):
        raise CompileError("ANCESTRA_FAST_KERNELS is off")
    if amp != "bf16":
        raise CompileError(f"amp {amp} is not bf16")
    if _CHECKED.get("e2e") == "off(check)":
        raise CompileError("fast kernels failed their check")
    orig_conv, orig_rn = _cf._ref_causal_conv1d, _M.RMSNorm.forward

    def _fail(reason):
        _cf._ref_causal_conv1d, _M.RMSNorm.forward = orig_conv, orig_rn
        raise CompileError(reason)

    try:
        from ancestra.model.infer import _amp_ctx
        W = int(window)
        dev = torch.device(device)
        x_w, x_s = _seeded_onehot(W, dev), _seeded_onehot(1152, dev)
        ctx = _amp_ctx(device, amp)
        with torch.no_grad(), ctx:
            ref_w = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in model(x_w).items()}
            ref_s = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in model(x_s).items()}
    except Exception as e:
        _fail(f"reference failed: {type(e).__name__}")
    dw = _install_dwconv(model, dev)
    rn = _install_rmsnorm(model)
    if dw != "on" or rn != "on":
        _fail(f"dwconv={dw} rmsnorm={rn}")
    try:
        with torch.no_grad(), ctx:
            got_w, got_s = model(x_w), model(x_s)
    except Exception as e:
        _fail(f"e2e call failed: {type(e).__name__}")
    if not _dicts_equal(got_w, ref_w):
        _fail("e2e full-window mismatch")
    if not _dicts_equal(got_s, ref_s):
        _fail("e2e short-window mismatch")
    if os.environ.get("TORCHDYNAMO_DISABLE", "") == "1" or torch._dynamo.config.disable:
        _fail("TORCHDYNAMO_DISABLE is set: nothing would be compiled")
    t0 = time.perf_counter()
    try:
        wrapped = Compiled(model, (1, 6, W))
    except Exception as e:
        _fail(f"torch.compile failed: {type(e).__name__}")
    if wrapped._compiled is model:
        _fail("TORCHDYNAMO_DISABLE is set: nothing would be compiled")
    try:
        with torch.no_grad(), ctx:
            outs = [{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in wrapped(x_w).items()}
                    for _ in range(3)]
            compile_s = time.perf_counter() - t0
            got_s = wrapped(x_s)
    except Exception as e:
        _fail(f"compiled call failed: {type(e).__name__}")
    if not _stable_equal(outs[0], outs[1]) or not _stable_equal(outs[1], outs[2]):
        _fail("compiled calls differ")
    maxabs, agree = 0.0, 1.0
    try:
        for k in ref_w:
            if k not in outs[0]:
                _fail(f"key {k} missing")
            r, g = ref_w[k], outs[0][k]
            if r.shape != g.shape or r.dtype != g.dtype or not torch.equal(torch.isfinite(r), torch.isfinite(g)):
                _fail(f"key {k} changed shape, dtype or finite mask")
            m = torch.isfinite(r) & torch.isfinite(g)
            if bool(m.any()):
                maxabs = max(maxabs, float((g.float() - r.float()).abs()[m].max()))
            if k == "joint":
                agree = float((r.argmax(dim=2) == g.argmax(dim=2)).float().mean())
    except CompileError:
        raise
    except Exception as e:
        _fail(f"tolerance check failed: {type(e).__name__}")
    if maxabs > MAXABS or agree < AGREE:
        _fail(f"tolerance check failed: maxabs {maxabs:.4f} agree {agree:.6f}")
    if not _dicts_equal(got_s, ref_s):
        _fail("short-window mismatch")
    _note(dw, rn, "off(compile)")
    print(f"ancestra: compile=reduce-overhead compile_s={compile_s:.1f} check_maxabs={maxabs:.4f} "
          f"check_agree={agree:.6f}", file=sys.stderr, flush=True)
    return wrapped


def enable(model, device, window, amp, compile=None):
    """Install G1+G2 and wrap the model in G3 after process-once self-checks; no-op off CUDA or with
    ANCESTRA_FAST_KERNELS=0 (returns ``model`` unchanged and patches nothing). The ANCESTRA_FAST_KERNELS
    environment variable is read when enable() runs, once per process. G3 captures under bf16 autocast,
    so it wraps only when amp == "bf16" (otherwise cudagraph=off(amp)). Before installing anything the
    unpatched model is run under the run's amp on (1, 6, W) and (1, 6, 1152); after G1/G2/G3 both inputs
    must still match bitwise, else everything is restored and all three print off(check). One stderr line.
    With compile="reduce-overhead" (--compile) G3 is skipped and the G1+G2 model is torch.compiled for full
    windows only after a tolerance check; any refusal raises CompileError with the originals restored."""
    if compile is not None:
        return _enable_compiled(model, device, window, amp, compile)
    if not _is_cuda(device):
        _note("off(cpu)", "off(cpu)", "off(cpu)")
        return model
    if os.environ.get("ANCESTRA_FAST_KERNELS", "").strip().lower() in ("0", "false", "off", "no"):
        _note("off(env)", "off(env)", "off(env)")
        return model
    if _CHECKED.get("e2e") == "off(check)":
        _note("off(check)", "off(check)", "off(check)")
        return model
    orig_conv, orig_rn = _cf._ref_causal_conv1d, _M.RMSNorm.forward
    try:
        from ancestra.model.infer import _amp_ctx
        dev = torch.device(device)
        x_w, x_s = _seeded_onehot(window, dev), _seeded_onehot(1152, dev)
        ctx = _amp_ctx(device, amp)
        with torch.no_grad(), ctx:
            ref_w = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in model(x_w).items()}
            ref_s = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in model(x_s).items()}
    except Exception:
        _cf._ref_causal_conv1d, _M.RMSNorm.forward = orig_conv, orig_rn
        _CHECKED["dwconv"] = _CHECKED["rmsnorm"] = None
        _CHECKED["e2e"] = "off(check)"
        _note("off(check)", "off(check)", "off(check)")
        return model
    dw = _install_dwconv(model, dev)
    rn = _install_rmsnorm(model)
    if amp == "bf16":
        wrapped, cg = _maybe_graph(model, dev, window, _ref=ref_w, _x=x_w)
    else:
        wrapped, cg = model, f"off({amp})"
    try:
        with torch.no_grad(), ctx:
            got_w, got_s = wrapped(x_w), wrapped(x_s)
        if set(got_w) != set(ref_w) or any(not torch.equal(got_w[k], ref_w[k]) for k in ref_w):
            raise AssertionError("e2e full-window mismatch")
        if set(got_s) != set(ref_s) or any(not torch.equal(got_s[k], ref_s[k]) for k in ref_s):
            raise AssertionError("e2e short-window mismatch")
    except Exception:
        _cf._ref_causal_conv1d, _M.RMSNorm.forward = orig_conv, orig_rn
        _CHECKED["dwconv"] = _CHECKED["rmsnorm"] = None
        _CHECKED["e2e"] = "off(check)"
        _note("off(check)", "off(check)", "off(check)")
        return model
    _note(dw, rn, cg)
    return wrapped
