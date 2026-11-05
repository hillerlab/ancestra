"""Whole-sequence inference: windows -> network -> RC-TTA -> adapter -> stitched per-base emissions.

- **Windows:** W at stride S, 0 < S <= W; the interior owned margin is (W - S)/2. Each base takes its value from the
  window whose centre is nearest (ties to the earlier window), with no blending: ``owned_central``. A sequence shorter
  than W is read as one window of its own length rounded up to the model's multiple (128 nt), never padded to W with N;
  the padded tail is never copied out.
- **Strands:** one pass gives both planes (plane 0 is the + strand, plane 1 the - strand, genome coordinates).
- **RC-TTA** (``tta="rc"``): a second pass on the reverse complement, mapped back by swapping planes and reversing
  positions (``rc_output``); the joint states are averaged with ``_log_mean2`` (log of the mean softmax), the sites as the
  mean of the sigmoids. Both happen per window, before the adapter (``ancestra.model.adapter.adapt``), so only the
  decoder channels and the native UTR channels are stitched.
"""
import math
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from ancestra.model import network
from ancestra.model.adapter import adapt, native_utr7

NONFINITE_FRAC = 0.001                        # per-window refusal budget: > 0.1 % of a window's values non-finite refuses the sequence
_IDX_LUT = np.full(256, 4, dtype=np.uint8)
for _ch, _code in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _IDX_LUT[ord(_ch)] = _code
    _IDX_LUT[ord(_ch.lower())] = _code
_MASK_LUT = np.zeros(256, dtype=np.uint8)
for _ch in "acgtn":
    _MASK_LUT[ord(_ch)] = 1
_COMP = torch.tensor([3, 2, 1, 0, 4, 5])      # channel permutation of the RC input: A<->T, C<->G, N and the mask bit stay


def tile_starts(L, W, S):
    assert W > 0 and S > 0 and S <= W, "need 0 < stride <= window"
    if L <= 0:
        return []
    if L < W:
        # short contig: single padded tile starting at 0 (caller pads to W)
        return [0]
    starts = list(range(0, L - W + 1, S))
    if starts[-1] != L - W:
        starts.append(L - W)
    return starts


def owned_central(starts, W, L):
    """Owned interval per window: each base goes to the window whose centre is nearest (ties to the earlier one).

    Between windows k and k+1 the boundary is ceil((s_k + s_{k+1} + W) / 2); the first window keeps the sequence start and the last
    the sequence end. With stride 0 < S <= W every owned base lies >= (W - S) / 2 from its window's edges, except at sequence ends.
    """
    if not starts:
        return []
    cuts = [0] + [(a + b + W + 1) // 2 for a, b in zip(starts, starts[1:])] + [L]
    return [(max(0, cuts[k]), min(L, cuts[k + 1])) for k in range(len(starts))]


def encode(raw, device):
    """uint8 bytes ``[B, W]`` -> model input ``[B, 6, W]`` f32: one-hot ACGTN + the soft-mask bit (lower case = masked)."""
    idx = torch.from_numpy(_IDX_LUT[raw]).to(device).long()
    m = torch.from_numpy(_MASK_LUT[raw]).to(device).float()
    return torch.cat((F.one_hot(idx, 5).permute(0, 2, 1).float(), m[:, None]), dim=1)


def rc_input(x):
    """Reverse complement of a model input ``[B, 6, W]`` (an involution)."""
    return x.index_select(1, _COMP.to(x.device)).flip(-1)


def rc_output(y):
    """Map RC-view per-plane outputs ``[B, 2, C, W]`` back to the forward frame: swap the strand planes, reverse positions."""
    return y.flip(1).flip(-1)


def _pad_window(raw_all, s, W):
    raw = raw_all[s:s + W]
    if len(raw) < W:
        raw = np.concatenate([raw, np.full(W - len(raw), ord("N"), dtype=np.uint8)])
    return raw


def build_model(path, label_tag=None):
    """The released network with its EMA weights, read with ``torch.load(..., weights_only=True)``: a pickled
    non-tensor is refused. The released weights carry no label-source tag, so ``label_tag`` must be None."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if label_tag is not None:
        raise ValueError(f"label_tag {label_tag!r} needs a checkpoint trained with data_config.label_tag")
    cfg = dict(ck.get("config") or {})
    if cfg.get("model") != "ancestra":
        raise ValueError(f"unknown model kind {cfg.get('model')!r}: expected 'ancestra'")
    model = network.build_model(dict(cfg.get("model_config") or {}))
    model.load_state_dict(dict(ck["ema"]))
    return model.eval()


class NonfiniteEmissions(RuntimeError):
    """A window's model outputs exceed the non-finite budget (``NONFINITE_FRAC``): the sequence is refused."""


def _window_nonfinite_counts(joint, sites):
    """Per-window non-finite counts of a ``window_emissions`` batch: one int per batch row.

    ``joint``/``sites`` are ``[B, 2, C, W]`` (log-probs / probabilities); the count covers every element of the row.
    """
    bad_j = (~torch.isfinite(joint.float())).sum(dim=(1, 2, 3))
    bad_s = (~torch.isfinite(sites.float())).sum(dim=(1, 2, 3))
    return (bad_j + bad_s).tolist()


def _amp_ctx(device, amp):
    # device type, not the string: --device cuda:0 is still cuda and must still autocast
    if amp == "fp16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", torch.float16)
    if amp == "bf16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", torch.bfloat16)
    return nullcontext()


def _log_mean2(logits, logits_rc):
    """log((softmax(x) + softmax(x_rc)) / 2) over dim 2, bit-identical to that expression wherever it is finite; where both
    probabilities underflow to 0 in f32 (log -> -inf) the exact logaddexp form fills in."""
    lp = torch.log((torch.softmax(logits, dim=2) + torch.softmax(logits_rc, dim=2)) * 0.5)
    bad = ~torch.isfinite(lp)
    if bool(bad.any()):
        exact = torch.logaddexp(torch.log_softmax(logits, dim=2), torch.log_softmax(logits_rc, dim=2)) - math.log(2.0)
        lp = torch.where(bad, exact, lp)
    return lp


@torch.no_grad()
def window_emissions(model, x, tta, amp, device):
    """Model input ``[B, 6, W]`` -> adapter emissions ``(la, lb, c)`` ``[B, 2, {3,3,4}, W]`` f32 (RC-TTA averaged when
    tta='rc') and the native UTR/TSS/TES channels ``[B, 2, 7, W]`` f32 (``adapter.native_utr7`` of the same
    TTA-averaged posteriors)."""
    ctx = _amp_ctx(device, amp)
    with ctx:
        out = model(x)
    joint, sites = out["joint"].float(), out["sites"].float()
    if tta == "rc":
        with ctx:
            out_rc = model(rc_input(x))
        lp = _log_mean2(joint, rc_output(out_rc["joint"].float()))
        sp = (torch.sigmoid(sites) + torch.sigmoid(rc_output(out_rc["sites"].float()))) * 0.5
    elif tta == "none":
        lp, sp = torch.log_softmax(joint, dim=2), torch.sigmoid(sites)
    else:
        raise ValueError(f"unknown tta={tta!r}: expected 'rc' or 'none'")
    counts = _window_nonfinite_counts(lp, sp)
    budget = max(1, int(NONFINITE_FRAC * (lp.shape[1] * lp.shape[2] + sp.shape[1] * sp.shape[2]) * lp.shape[3]))
    for b, nbad in enumerate(counts):
        if nbad:
            print(f"warning: window {b}: {nbad} non-finite model outputs replaced (LOG_FLOOR/0)", file=sys.stderr)
        if nbad > budget:
            raise NonfiniteEmissions(
                f"window {b}: {nbad} non-finite values exceed the 0.1 % budget ({budget}): refusing the sequence")
    la, lb, c = adapt(lp, sp)
    return la, lb, c, native_utr7(lp, sp)


@torch.no_grad()
def infer_arrays(seq, model, window, stride, device, bs, amp, tta):
    """Sequence bytes -> stitched ``(la [2,3,L], lb [2,3,L], c [2,4,L], nat [2,7,L])`` f32, ``nat`` the native
    UTR/TSS/TES channels in the same orientation.

    A sequence shorter than ``window`` is one window of its own length rounded up to the model's multiple
    (``cfg.window_multiple``; 128 nt otherwise), not padded to W with N.
    """
    L = len(seq)
    if 0 < L < window:
        m = getattr(getattr(model, "cfg", None), "window_multiple", 128)
        window = stride = -(-L // m) * m
    la = np.zeros((2, 3, L), dtype=np.float32)
    lb = np.zeros((2, 3, L), dtype=np.float32)
    cc = np.zeros((2, 4, L), dtype=np.float32)
    nat = np.zeros((2, 7, L), dtype=np.float32)
    starts = tile_starts(L, window, stride)
    owned = owned_central(starts, window, L)
    raw_all = np.frombuffer(bytes(seq), dtype=np.uint8)
    for i in range(0, len(starts), bs):
        ks = range(i, min(i + bs, len(starts)))
        x = encode(np.stack([_pad_window(raw_all, starts[k], window) for k in ks]), device)
        arr = [t.cpu().numpy() for t in window_emissions(model, x, tta, amp, device)]
        a, b, c = arr[0], arr[1], arr[2]
        for j, k in enumerate(ks):
            os_, oe = owned[k]
            o = os_ - starts[k]
            la[:, :, os_:oe] = a[j, :, :, o:o + oe - os_]
            lb[:, :, os_:oe] = b[j, :, :, o:o + oe - os_]
            cc[:, :, os_:oe] = c[j, :, :, o:o + oe - os_]
        n = arr[3]
        for j, k in enumerate(ks):
            os_, oe = owned[k]
            o = os_ - starts[k]
            nat[:, :, os_:oe] = n[j, :, :, o:o + oe - os_]
    return la, lb, cc, nat
