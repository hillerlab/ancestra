"""Per-side semi-Markov UTR decoder with the CDS frozen.

Input and output are the same as ``utr_attach``'s (utr7 emissions, fixed CDS chains, the
same transcript dicts). Reuses attach's helpers (``_windows``, ``_exp``, ``_pieces``,
``merge``), its TSS/TES de-biasing convention (site BCE pos_weight 100, so the de-biased
logit is the raw logit minus ln 100) and the DP's splice terms (``DP_TAU`` .05,
``DP_EPS`` 1e-4, ``MIN_INTRON`` 30, donor-then-acceptor orientation per side).

Search region: per transcript side, from the CDS edge outward up to that side's
window, bounded by the nearest same-strand predicted CDS (attach's ``_windows``).
With ``sides="5"`` the decoded 5' side uses ``window5`` and the held DP 3' side uses
the held parameters' window; each side's span is computed separately, so widening W5
never widens the held side.

Path: in transcript-outward coordinates (t = 0 at the CDS edge, increasing outward) a
path covers [0, T) with alternating UTR-exon and UTR-intron segments and ends at the
transcript end T (TSS on 5', TES on 3'). Bases beyond T are background (score 0).

Score (all in nats; every span base carries the exon term, intron bases add the
intron term, i.e. intron-vs-background by the chain rule):
  - exon bases:  sum log((p_ex + e) / (p_bg + e)), p_bg the intergenic posterior;
  - intron bases: the exon term plus sum log((p_in + e) / (p_ex + e)) (the DP's
    per-base term), so given the span the intron choice maximises the DP objective;
  - splice terms log s_start + log s_end with the DP's orientation (donor first in
    transcript order; on the 5' side the donor sits at the outer intron end because
    transcript order runs inward there), candidates s >= tau;
  - an end term w_end * (de-biased TSS / TES logit at T);
  - log length priors per side and segment type (terminal / internal UTR exon, UTR
    intron); flat (None) means 0 for every length except the degenerate length-0
    segment, which always scores 0. ``lambda_prior`` L multiplies every log
    length prior (1.0 uses them as they are);
  - the empty UTR (end at the CDS edge) scores exactly 0, so a UTR exists only if the
    best path beats it (strictly greater; ties keep the CDS-only output).

End candidates are local maxima of the de-biased end logit after non-maximum
suppression: a position at least as high as every neighbour and strictly higher
than at least one of them (plateau tops give candidates; flat runs give none; a
1 bp window yields its single base), highest first, each kept peak suppresses
+-25 bp with ties toward the CDS edge. The peak base itself belongs to the UTR
(a peak at index i ends the span at T = i + 1). No peak-height threshold; the
score decides.

Decoding is an exact semi-Markov Viterbi over candidate positions, no greedy choices:
one ascending sweep computes, for every splice-start candidate u, the best exon parse
of [0, u) ready to open an intron, and for every splice-end candidate v the best
intron chain closing at v (backpointers kept); every end candidate T is then finished
exactly (best last-boundary, all-exon included) and the best T -- or the empty UTR --
wins. Deterministic tie-breaks: earliest boundary, farthest end.

Complexity per side with window length n (<= W), K splice candidates (s >= tau) and E
end candidates: O(K^2 + E*K) scalar-max steps with numpy-vectorised inner loops, O(n)
memory for the prefix sums plus O(K + E) states. The pruning bounds it: NMS gives
E <= n/26 + 1, tau admits only s >= tau positions into transitions, and W bounds n.

With flat length priors and w_end 0, the intron decomposition on a fixed span equals
the DP's (``utr_attach._introns``): given the span, the exon term is constant and both
maximise the same splice-plus-intron-odds objective with the same keep-only-if-it-
raises rule (the all-exon finish). The outer base stays exon (terminal exon >= 1 bp)
and introns may abut the CDS edge, exactly the DP's conventions, except every
internal UTR exon is required to be >= 1 bp (no abutting introns around a
zero-length exon; the DP may emit zero-length blocks).

``sides="5"`` decodes only the 5' side and takes the 3' side from ``utr_attach``'s
DP path with (theta, theta_i, window) from the held parameters; ``sides="3"`` is the
mirror (the decoded 3' side uses ``window5``). ``"both"`` decodes both sides with
``window5``.
"""
import hashlib
import json
import os
import sys
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from ancestra.decode import utr_attach
from ancestra.decode.tx import make_tx


# Defaults of the function signatures; the released values come from params.json. tau/eps match the DP.
# The held side's (theta, theta_i, window) always comes from the held parameters: no in-code copy.
W_END = 1.0
EPS = 1e-4
TAU = 0.05
WINDOW = 20_000
WINDOW5 = WINDOW
LAMBDA_PRIOR = 1.0
NMS_HALF = 25
BIN_EDGES = (50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 20000)


def _as_window(value, what="window"):
    """Integer window validation: never truncate.

    ``value`` must be an integer (bool rejected); anything else — including
    a non-integer float such as 20000.9 — raises ValueError.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("bad %s %r: want an integer" % (what, value))
    return value


def _lp_lookup(logp, edges, lengths):
    """Log prior for integer lengths. Degenerate length 0 scores 0 (no segment)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    out = np.zeros(lengths.shape, np.float64)
    pos = lengths > 0
    if np.any(pos):
        idx = np.searchsorted(np.asarray(edges, dtype=np.int64), lengths[pos], side="right")
        out[pos] = np.asarray(logp, dtype=np.float64)[idx]
    return out


def _peaks(deb, half=NMS_HALF):
    """Local maxima after non-maximum suppression within +-half.

    A candidate is at least as high as every neighbour and strictly higher than
    at least one of them: plateau tops give candidates, flat runs give none. A
    1 bp window yields its single base.
    """
    n = len(deb)
    if n < 1:
        return []
    if n == 1:
        return [0]
    cand = []
    for i in range(n):
        if i == 0:
            if deb[i] > deb[i + 1]:
                cand.append(i)
        elif i == n - 1:
            if deb[i] > deb[i - 1]:
                cand.append(i)
        elif (deb[i] >= deb[i - 1] and deb[i] >= deb[i + 1]
                and (deb[i] > deb[i - 1] or deb[i] > deb[i + 1])):
            cand.append(i)
    cand.sort(key=lambda i: (-float(deb[i]), i))  # descending value, ties toward the CDS edge (t = 0)
    suppressed = np.zeros(n, bool)
    kept = []
    for i in cand:
        if suppressed[i]:
            continue
        kept.append(i)
        suppressed[max(0, i - half):min(n, i + half + 1)] = True
    return sorted(kept)


def _decode_side(p_ex, p_in, p_bg, s_start, s_end, deb, tau, eps, w_end,
                 lp_term=None, lp_internal=None, lp_intron=None, edges=BIN_EDGES,
                 force_end=None, min_intron=None, lambda_prior=LAMBDA_PRIOR):
    """Exact semi-Markov Viterbi of one side in transcript-outward coordinates.

    ``s_start`` / ``s_end`` are the oriented site tracks (donor first in transcript
    order: donor at the intron's low-t end on 3', at its high-t end on 5'). An intron
    [u, v) needs s_start[u] >= tau, s_end[v - 1] >= tau, v - u >= min_intron.
    ``lambda_prior`` multiplies every log length prior (1.0 uses them as they are).
    Returns (end_T or None, introns [(u, v)] in t coordinates, score). Score 0.0 and
    end None is the empty UTR.
    """
    ua = _decode_side.ua
    if min_intron is None:
        min_intron = ua.MIN_INTRON
    lam = float(lambda_prior)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("bad lambda_prior %r: want a finite value >= 0" % (lambda_prior,))
    n = int(len(p_ex))
    if n < 1:
        return None, [], 0.0
    for _name, _arr in (("p_ex", p_ex), ("p_in", p_in), ("p_bg", p_bg),
                        ("s_start", s_start), ("s_end", s_end), ("end_logit", deb)):
        _a = np.asarray(_arr, np.float64)
        if _a.shape != (n,):
            raise ValueError("length mismatch in %s: %r for n=%d" % (_name, _a.shape, n))
        _bad = np.flatnonzero(np.isnan(_a))
        if len(_bad):
            raise ValueError("NaN in %s at t-position %d" % (_name, int(_bad[0])))
    _inf = np.flatnonzero(np.isinf(np.asarray(deb, np.float64)))
    if len(_inf):
        raise ValueError("inf in end_logit at t-position %d" % int(_inf[0]))
    e = np.log((np.asarray(p_ex, np.float64) + eps) / (np.asarray(p_bg, np.float64) + eps))
    q = np.log((np.asarray(p_in, np.float64) + eps) / (np.asarray(p_ex, np.float64) + eps))
    E = np.empty(n + 1, np.float64)
    E[0] = 0.0
    np.cumsum(e, out=E[1:])
    Q = np.empty(n + 1, np.float64)
    Q[0] = 0.0
    np.cumsum(q, out=Q[1:])
    # Chain rule: an intron base scores exon-vs-background (e) plus intron-vs-exon (q),
    # i.e. intron-vs-background. Every span base carries e; intron bases add q. Given
    # the span, the exon sum is constant and the intron choice maximises the DP's
    # splice-plus-q objective, which is the flat-prior parity below.
    QE = Q + E
    s0 = np.asarray(s_start, np.float64)
    s1 = np.asarray(s_end, np.float64)
    S = [u for u in range(n) if s0[u] >= tau]          # intron starts (0 allowed: abut CDS)
    Vend = [v for v in range(1, n + 1) if s1[v - 1] >= tau]  # half-open intron ends
    in_S = np.zeros(n + 1, bool)
    in_S[S] = True
    in_V = np.zeros(n + 1, bool)
    in_V[Vend] = True

    def lp_term_v(d):
        v = _lp_lookup(lp_term, edges, d) if lp_term is not None else np.zeros_like(np.asarray(d, float))
        return v * lam

    def lp_int_v(d):
        v = _lp_lookup(lp_internal, edges, d) if lp_internal is not None else np.zeros_like(np.asarray(d, float))
        return v * lam

    def lp_in_v(d):
        v = _lp_lookup(lp_intron, edges, d) if lp_intron is not None else np.zeros_like(np.asarray(d, float))
        return v * lam

    closed = {0: 0.0}   # close position v -> best chain score of [0, v) ending with an intron (0: empty)
    back_o = {}         # intron start u -> previous close v'
    back_c = {}         # intron close v -> intron start u
    open1 = {}
    for p in sorted(set(S) | set(Vend) | {0}):
        if in_V[p]:
            # best chain closing an intron at p (the intron bases keep their e term via QE)
            cands = [u for u in open1 if u <= p - min_intron]
            if cands:  # guard only the close computation: the open bookkeeping below always runs
                cands.sort()
                uv = np.array(cands, np.int64)
                vals = (np.array([open1[u] for u in cands], np.float64)
                        + np.log(s0[uv]) - QE[uv] + lp_in_v(p - uv))
                b = int(np.argmax(vals))
                if p not in closed:  # a close position is each computed once (events are unique)
                    closed[p] = float(QE[p] + np.log(s1[p - 1]) + vals[b])
                    back_c[p] = cands[b]
        if in_S[p]:
            # best exon parse of [0, p) ready to open an intron at p (close first at
            # the same position). The exon [v, p) between two introns must be >= 1 bp,
            # so a previous close v > 0 re-opens only strictly inside (v = 0 is the
            # empty chain: opening at p = 0 abuts the CDS edge, as in the DP)
            cands = sorted(v for v in closed if v == 0 or v < p)
            cv = np.array([closed[v] for v in cands], np.float64)
            vals = cv - E[cands] + lp_int_v(np.array([p - v for v in cands], np.int64))
            b = int(np.argmax(vals))  # first maximum: deterministic, earliest boundary
            open1[p] = float(E[p] + vals[b])
            back_o[p] = cands[b]
    ends = [force_end] if force_end is not None else _peaks(np.asarray(deb, np.float64))
    ends = [int(i) for i in ends if 0 <= i <= n - 1]
    best_T, best_v, best_tot = None, None, 0.0
    for i in ends:
        T = i + 1  # the peak base itself belongs to the UTR: the span covers [0, T)
        cands = sorted(v for v in closed if v <= T - 1)  # terminal exon >= 1 bp
        if not cands:
            continue
        cv = np.array([closed[v] for v in cands], np.float64)
        vals = cv - E[cands] + lp_term_v(np.array([T - v for v in cands], np.int64))
        b = int(np.argmax(vals))
        tot = float(E[T] + vals[b] + w_end * float(deb[i]))
        if tot > best_tot or (tot == best_tot and best_T is not None and T > best_T):
            best_T, best_v, best_tot = T, cands[b], tot
    if best_T is None:  # nothing beats the empty UTR
        return None, [], 0.0
    introns = []
    v = best_v
    while v != 0:
        u = back_c[v]
        introns.append((u, v))
        v = back_o[u]
    introns.reverse()
    return best_T, introns, best_tot


def _priors_for(priors, side):
    """(lp_term, lp_internal, lp_intron, edges) logp arrays for a side; flat priors give Nones."""
    if not priors:
        return None, None, None, BIN_EDGES
    edges = tuple(priors.get("edges", BIN_EDGES))
    s = priors["sides"][side]
    return (np.array(s["terminal"]["logp"], np.float64),
            np.array(s["internal"]["logp"], np.float64),
            np.array(s["intron"]["logp"], np.float64), edges)


def _decode_one(tx, origin, u7, cc, span, ua, w_end, eps, tau, priors,
                 sides="both", held=None, lambda_prior=LAMBDA_PRIOR):
    low_stop, cs, ce, high_stop = span
    plus = tx["strand"] == "+"
    L = u7.shape[-1]
    if cs >= origin + L or ce <= origin:
        raise ValueError("%s: CDS span (%d, %d) outside cache [%d, %d)"
                         % (tx["name"], cs, ce, origin, origin + L))
    if sides not in ("5", "3", "both"):
        raise SystemExit("bad sides %r: want 5, 3 or both" % (sides,))

    def run_side(g0, g1, edge, increasing, exon_ch, intron_ch, end_ch, side, donor_first):
        L = u7.shape[-1]
        if edge < origin or edge > origin + L:
            raise ValueError("%s: %s CDS edge %d outside cache [%d, %d)"
                             % (tx["name"], side, edge, origin, origin + L))
        a0, b0 = max(0, g0 - origin), min(L, g1 - origin)
        idx = np.arange(a0, b0, dtype=np.int64)
        if not increasing:
            idx = idx[::-1]
        if len(idx) == 0:
            return None, []
        group = tx["chrom"]

        def gpos(i):
            return origin + a0 + i if increasing else origin + b0 - 1 - i

        def check(arr, channel, kind):
            a = np.asarray(arr)
            bad = np.flatnonzero(np.isnan(a))
            if len(bad):
                i = int(bad[0])
                raise ValueError("NaN in %s %s (side %s) at position %d (group %s)"
                                 % (channel, kind, side, gpos(i), group))
            if channel in ("donor", "acceptor"):
                bad = np.flatnonzero(~np.isfinite(a) | (a < 0) | (a > 1))
                if len(bad):
                    i = int(bad[0])
                    raise ValueError("bad value %r in %s %s (side %s) at position %d (group %s)"
                                     % (float(a[i]), channel, kind, side, gpos(i), group))
            return a

        exon_log = check(u7[exon_ch][idx], "exon", "log cache")
        intron_log = check(u7[intron_ch][idx], "intron", "log cache")
        check(u7[ua.CH_INTERGENIC][idx], "intergenic", "log cache")
        end_log = check(u7[end_ch][idx], "TSS" if side == "5" else "TES", "logit")
        inf = np.flatnonzero(np.isinf(np.asarray(end_log, np.float64)))
        if len(inf):
            raise ValueError("inf in %s logit (side %s) at position %d (group %s)"
                             % ("TSS" if side == "5" else "TES", side, gpos(int(inf[0])), group))
        p_ex = ua._exp(exon_log, side="%s-exon" % side, pos=origin + a0, group=group)
        p_in = ua._exp(intron_log, side="%s-intron" % side, pos=origin + a0, group=group)
        p_bg = ua._exp(u7[ua.CH_INTERGENIC][idx], side="intergenic", pos=origin + a0, group=group)
        if cc is None:
            s_don = np.zeros(len(idx), np.float64)
            s_acc = np.zeros(len(idx), np.float64)
        else:
            s_don = check(np.asarray(cc[ua.CH_DONOR][idx], np.float64), "donor", "splice track")
            s_acc = check(np.asarray(cc[ua.CH_ACCEPTOR][idx], np.float64), "acceptor", "splice track")
        s_start, s_end = (s_don, s_acc) if donor_first else (s_acc, s_don)
        deb = np.asarray(end_log, np.float32).astype(np.float64) - ua.LN100
        lp_t, lp_i, lp_n, edges = _priors_for(priors, side)
        T, introns_t, _ = _decode_side(p_ex, p_in, p_bg, s_start, s_end, deb, tau, eps,
                                       w_end, lp_t, lp_i, lp_n, edges,
                                       lambda_prior=lambda_prior)
        if T is None:
            return None, []
        if increasing:
            return edge + int(T), [(edge + u, edge + v) for u, v in introns_t]
        return edge - int(T), [(edge - v, edge - u) for u, v in introns_t]

    if plus:
        g5, i5 = run_side(low_stop, cs, cs, False, ua.CH_UTR5_EXON, ua.CH_UTR5_INTRON,
                           ua.CH_TSS, "5", False) if sides in ("both", "5") else (None, [])
        g3, i3 = run_side(ce, high_stop, ce, True, ua.CH_UTR3_EXON, ua.CH_UTR3_INTRON,
                           ua.CH_TES, "3", True) if sides in ("both", "3") else (None, [])
    else:
        g5, i5 = run_side(ce, high_stop, ce, True, ua.CH_UTR5_EXON, ua.CH_UTR5_INTRON,
                           ua.CH_TSS, "5", False) if sides in ("both", "5") else (None, [])
        g3, i3 = run_side(low_stop, cs, cs, False, ua.CH_UTR3_EXON, ua.CH_UTR3_INTRON,
                           ua.CH_TES, "3", True) if sides in ("both", "3") else (None, [])

    def pieces(g_end, introns_g, lo, hi, outer_low):
        if g_end is None:
            return []
        span_g = (g_end, hi) if outer_low else (lo, g_end)
        return ua._pieces(span_g[0], span_g[1], sorted(introns_g))

    if plus:
        extra = pieces(g5, i5, low_stop, cs, True) + pieces(g3, i3, ce, high_stop, False)
    else:
        extra = pieces(g5, i5, ce, high_stop, False) + pieces(g3, i3, low_stop, cs, True)
    if sides != "both":
        # the other side comes from utr_attach's DP path with the held file's parameters
        if held is None:
            raise SystemExit("sides %s needs held parameters" % sides)
        att = ua._attach_one(tx, origin, u7, cc, span, held["theta"], held["theta_i"])
        held_iv = att["three_iv"] if sides == "5" else att["five_iv"]
        extra = [p for p in extra] + [tuple(x) for x in held_iv]
        if ((att["name"], att["strand"], att["cds"], att["thick"]) !=
                (tx["name"], tx["strand"], tx["cds"], tx["thick"])):
            raise SystemExit("%s: attach changed a CDS chain" % tx["name"])
    blocks = ua.merge(extra + list(tx["cds"]))
    made = make_tx(tx["chrom"], tx["strand"], tx["name"], blocks,
                   (tx["cds"][0][0], tx["cds"][-1][1]), gene=tx["gene"])
    return made if made is not None else tx


def _windows_with(ua, txs, window):
    """attach's ``_windows`` with an explicit window on both sides, passed as its own arguments, never through the
    module global ``ua.WINDOW``."""
    return ua._windows(txs, window, window)


def _note_pool_broken(executor, exc):
    """Mark a broken decoder pool: later transcripts and later items of the worker skip it and decode
    in the parent. One stderr line per worker, naming the failure's class. A None executor (pool creation itself
    failed) only prints the line."""
    if executor is not None:
        if getattr(executor, "_ancestra_broken", False):
            return
        executor._ancestra_broken = True
    print("ancestra: decoder pool failed (%s); decoding in the main process" % type(exc).__name__,
          file=sys.stderr, flush=True)


def _decode_task(job):
    """One transcript on a spawn worker: the same ``_decode_one`` the parent runs.

    ``job`` is (tx, span, u, cc, origin, w_end, eps, tau, priors, sides, held, lam): the transcript, its window, the
    utr7 plane and the c plane sliced to the window's index range with the origin shifted by the slice start, and the
    scalar arguments. The attach module is the package's own, as in the parent."""
    tx, span, u, cc, origin, w_end, eps, tau, priors, sides, held, lam = job
    ua = utr_attach
    _decode_side.ua = ua
    return _decode_one(tx, origin, u, cc, span, ua, w_end, eps, tau, priors, sides, held, lam)


def decode_txs(txs, groups, ua, priors=None, window5=WINDOW5, w_end=W_END, eps=EPS, tau=TAU,
                 sides="both", held=None, lambda_prior=LAMBDA_PRIOR, executor=None):
    """Decode UTRs for fixed CDS chains: one output list, txs order.

    ``groups`` is {chrom: (start, utr7, c)} over the item's in-memory f16 arrays.

    ``window5`` sets the decoded side's window (the 5' side under ``sides="5"`` / ``"both"``).
    Under ``sides="5"`` / ``"3"`` the held side uses ``held``'s window (a ``load_held_params``
    dict or its FILE path, required); each side's span is computed separately.

    ``executor`` (None) runs a sequential loop. With an executor, each transcript whose CDS span
    and window lie inside the item's arrays becomes one task (its window's slice, origin shifted by the slice
    start); a transcript reaching outside runs ``_decode_one`` in the parent, in its turn, so errors keep the
    sequential path's exact message. Results land by index; the CDS-chain check runs in the parent; the first failure in loop order
    raises. A failed spawn or process start (``OSError``) at a submit, or a ``BrokenProcessPool`` at a submit or a
    result, decodes every transcript with no result through the in-parent path, in index order (byte-identical by
    construction: the same ``_decode_one`` on the same slice), marks the pool so later transcripts and later items
    of the worker skip it, and prints one stderr line naming the class; a task's own exception still propagates
    exactly as in the sequential loop."""
    missing = sorted({t["chrom"] for t in txs} - set(groups))
    if missing:
        raise SystemExit("no emit group for %s" % ", ".join(missing))
    if sides not in ("5", "3", "both"):
        raise SystemExit("bad sides %r: want 5, 3 or both" % (sides,))
    lam = float(lambda_prior)
    if not np.isfinite(lam) or lam < 0:
        raise SystemExit("bad lambda_prior %r: want a finite value >= 0" % (lambda_prior,))
    if isinstance(held, str):
        held, _ = load_held_params(held)
    window5 = _as_window(window5, "window5")
    if window5 <= 0:
        raise SystemExit("bad window5 %r: want a positive int" % (window5,))
    if sides == "both":
        win = _windows_with(ua, txs, window5)
    else:
        if held is None:
            raise SystemExit("sides %s needs held parameters" % sides)
        win_dec = _windows_with(ua, txs, window5)
        win_held = _windows_with(ua, txs, _as_window(held["window"], "window"))
        win = {}
        for t in txs:
            i = id(t)
            cs, ce = t["cds"][0][0], t["cds"][-1][1]
            lo_d, _, _, hi_d = win_dec[i]
            lo_h, _, _, hi_h = win_held[i]
            if t["strand"] == "+":  # 5' is low, 3' is high
                lo, hi = (lo_d, hi_h) if sides == "5" else (lo_h, hi_d)
            else:                   # 5' is high, 3' is low
                lo, hi = (lo_h, hi_d) if sides == "5" else (lo_d, hi_h)
            win[i] = (lo, cs, ce, hi)
    _decode_side.ua = ua
    out = list(txs)
    by = {}
    for i, t in enumerate(txs):
        by.setdefault((t["chrom"], ua._plane(t["strand"])), []).append(i)
    for (chrom, si), idx in sorted(by.items()):
        origin, utr7, c = groups[chrom]
        u = np.asarray(utr7[si])
        cc = None if c is None else np.asarray(c[si, :2])
        if executor is None:
            for i in idx:
                got = _decode_one(txs[i], origin, u, cc, win[id(txs[i])], ua,
                                  w_end, eps, tau, priors, sides, held, lam)
                exp = txs[i]
                if (got["name"], got["strand"], got["cds"], got["thick"]) != (
                        exp["name"], exp["strand"], exp["cds"], exp["thick"]):
                    raise SystemExit("%s: decoder changed a CDS chain" % exp["name"])
                out[i] = got
        else:
            futs = {}
            L = u.shape[-1]
            for i in idx:
                lo, cs, ce, hi = win[id(txs[i])]
                if cs >= origin and ce <= origin + L and lo >= origin and hi <= origin + L:
                    if getattr(executor, "_ancestra_broken", False):
                        continue
                    s, e = lo - origin, hi - origin
                    uw = u[..., s:e]
                    ccw = None if cc is None else cc[..., s:e]
                    try:
                        futs[i] = executor.submit(_decode_task, (txs[i], win[id(txs[i])], uw, ccw, lo,
                                                                 w_end, eps, tau, priors, sides, held, lam))
                    except (BrokenProcessPool, OSError) as e:
                        _note_pool_broken(executor, e)
            for i in idx:
                if i in futs and not getattr(executor, "_ancestra_broken", False):
                    try:
                        got = futs[i].result()
                    except BrokenProcessPool as e:
                        _note_pool_broken(executor, e)
                        got = _decode_one(txs[i], origin, u, cc, win[id(txs[i])], ua,
                                          w_end, eps, tau, priors, sides, held, lam)
                else:
                    got = _decode_one(txs[i], origin, u, cc, win[id(txs[i])], ua,
                                      w_end, eps, tau, priors, sides, held, lam)
                exp = txs[i]
                if (got["name"], got["strand"], got["cds"], got["thick"]) != (
                        exp["name"], exp["strand"], exp["cds"], exp["thick"]):
                    raise SystemExit("%s: decoder changed a CDS chain" % exp["name"])
                out[i] = got
        del u, cc
    return out


def load_priors(path):
    if not path:
        return None
    with open(path) as fh:
        p = json.load(fh)
    for side in ("5", "3"):
        for kind in ("terminal", "internal", "intron"):
            lp = p["sides"][side][kind]["logp"]
            if len(lp) != len(p.get("edges", BIN_EDGES)) + 1:
                raise SystemExit("%s: bad logp length for %s/%s" % (path, side, kind))
    return p


def load_held_params(path):
    """Held-side DP parameters for sides "5"/"3": JSON with theta, theta_i, window.

    Returns (held dict, sha256): the dict carries theta, theta_i, window plus the
    file's absolute path and sha256, which callers record in their outputs."""
    with open(path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        p = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SystemExit("%s: not a JSON file" % path)
    if not isinstance(p, dict):
        raise SystemExit("%s: want a JSON object with theta, theta_i, window" % path)
    ti = p.get("theta_i", p.get("theta-i"))
    try:
        theta, theta_i = float(p["theta"]), float(ti)
    except (KeyError, TypeError, ValueError):
        raise SystemExit("%s: want numeric theta, theta_i and window" % path)
    if "window" not in p:
        raise SystemExit("%s: want numeric theta, theta_i and window" % path)
    window = _as_window(p["window"], "window")  # ValueError on non-integer; never truncated
    if not (0.0 < theta < 1.0 and 0.0 < theta_i < 1.0):
        raise SystemExit("%s: theta=%r theta_i=%r want both in (0, 1)" % (path, theta, theta_i))
    if window <= 0:
        raise SystemExit("%s: window=%r want a positive int" % (path, window))
    return {"theta": theta, "theta_i": theta_i, "window": window,
            "path": os.path.abspath(path), "sha256": sha}, sha
