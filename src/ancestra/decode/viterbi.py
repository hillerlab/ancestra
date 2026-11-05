"""The CDS Viterbi decode of one item: per strand plane, in overlapping chunks, through the compiled decoder.

Torch-free: the alternative-isoform workers import it.
"""
import sys

import numpy as np

from ancestra import _native


LOG_FLOOR = -69.0   # ~ log(1e-30): a finite stand-in for log(0)


def _finite(la, lb, c):
    """Replace non-finite emissions in place: -inf/nan log-probabilities -> LOG_FLOOR, nan site probabilities -> 0. A single -inf
    makes every Viterbi path through that base impossible and erases the rest of the strand chunk. Returns the
    number of values replaced."""
    n = 0
    for x in (la, lb):
        bad = ~np.isfinite(x)
        if bad.any():
            n += int(bad.sum())
            x[bad] = LOG_FLOOR
    bad = ~np.isfinite(c)
    if bad.any():
        n += int(bad.sum())
        c[bad] = 0.0
    return n


def decode_chunk_bounds(L, chunk, overlap):
    """Chunk [0, L) into overlapping windows with cores partitioning [0, L).

    Chunks start every ``chunk - overlap`` bases, each spanning ``chunk``
    bases clipped to L. Core of chunk k is
    [start_k + overlap/2, start_k + chunk - overlap/2) except the first
    core starts at 0 and the last ends at L. Returns list of
    (cstart, cend, core_start, core_end) in sequence-relative coords.
    """
    assert chunk > 0, f"need chunk > 0, got {chunk}"
    assert 0 <= overlap < chunk, f"need 0 <= overlap < chunk, got {overlap=}, {chunk=}"
    if L <= 0:
        return []
    if L <= chunk:
        return [(0, L, 0, L)]
    step = chunk - overlap
    # an odd overlap splits floor/ceil so cores stay contiguous
    half_left = overlap // 2
    half_right = overlap - half_left
    bounds = []
    s = 0
    while True:
        e = min(s + chunk, L)
        bounds.append((s, e))
        if e == L:
            break
        s += step
    out = []
    for i, (cs, ce) in enumerate(bounds):
        core_s = 0 if i == 0 else cs + half_left
        core_e = L if i == len(bounds) - 1 else cs + chunk - half_right
        out.append((cs, ce, core_s, core_e))
    return out


def _split_a_bias(params):
    """Pop the Python-only decode-time A offsets off a copy of ``params``.

    Returns ``(params_for_rust, a_bias_cds, a_bias_int)``. The caller's dict is never
    mutated and the Rust decoder never sees these keys: they exist only so the Python
    decode path can add a global per-base calibration offset to the collapsed A view.
    The binding has no rule for them, so this is their only check: ValueError unless each
    is within [-5, 5] (nan and inf fail too). They are log-scale offsets, fitted at about
    ±0.02; a large one does not fail in the decoder, it silently degenerates the BED.
    """
    p = dict(params or {})
    def bias(k):
        try:
            return float(p.pop(k, 0.0))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{k}: {e}") from e

    b_cds, b_int = bias("a_bias_cds"), bias("a_bias_int")
    for k, v in (("a_bias_cds", b_cds), ("a_bias_int", b_int)):
        if not abs(v) <= 5.0:  # nan compares False, so it is refused too
            raise ValueError(f"{k}={v!r}: must be finite with |{k}| <= 5 (a log-scale offset)")
    return p, b_cds, b_int


def _apply_a_bias(la, b_cds, b_int):
    """Add per-base CDS/intron offsets to a per-strand ``[3, W]`` f32 array.

    The add happens on a copy, never in place: ``la`` may be a view into a stitched
    whole-sequence array, and overlapping decode chunks would otherwise apply the
    offset once per chunk in the overlap. Both offsets zero returns ``la`` untouched.
    """
    if not b_cds and not b_int:
        return la
    la = la.copy()
    if b_cds:
        la[1, :] += b_cds
    if b_int:
        la[2, :] += b_int
    return la


def _decode_genes(seq, strand, la, lb, c, start, params):
    """Per-strand ``_native.decode`` with the Python-only A offsets applied.

    ``la`` is the per-strand ``[3, W]`` f32 chunk. ``a_bias_cds``/``a_bias_int`` are
    popped from a copy of ``params`` and added to the CDS/intron log-odds of a copy of
    ``la``; the stitched/emitted arrays are left intact.
    """
    p, b_cds, b_int = _split_a_bias(params)
    la = _apply_a_bias(la, b_cds, b_int)
    return _native.decode(seq, strand, la, lb, c, start, p)


def decode_strand(seq, la, lb, cc, si, start, params, chunk=50_000_000, overlap=5_000_000, scores=None):
    """Decode one strand plane (si 0 = +, 1 = -) in overlapping chunks; rows (s, e, strand, score, cds), genes kept by
    CDS midpoint.

    ``scores``, when passed, maps ``id`` of each returned 5-tuple to the unclipped f32 gene score. The tuple keeps the
    clipped integer, so a sort of the rows, and the bed bytes, do not depend on the unclipped value.
    """
    strand = "+-"[si]
    rows = []
    for cs, ce, core_s, core_e in decode_chunk_bounds(len(seq), chunk, overlap):
        a = np.array(la[si, :, cs:ce], dtype=np.float32, order="C")
        b = np.array(lb[si, :, cs:ce], dtype=np.float32, order="C")
        c = np.array(cc[si, :, cs:ce], dtype=np.float32, order="C")
        nbad = _finite(a, b, c)
        if nbad:
            print(f"warning: {nbad} non-finite emission values on strand {strand} at {start + cs} replaced", file=sys.stderr)
        for g in _decode_genes(bytes(seq[cs:ce]), strand, a, b, c, start + cs, params):
            cds = sorted(tuple(int(v) for v in x) for x in g["cds"])
            s, e = cds[0][0], cds[-1][1]
            if start + core_s <= (s + e) // 2 < start + core_e:
                row = (s, e, g["strand"], int(min(1000, max(0, g["score"]))), cds)
                rows.append(row)
                if scores is not None:
                    scores[id(row)] = float(g["score"])
    return rows
