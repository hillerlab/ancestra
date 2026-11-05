"""Attach UTRs to a fixed CDS chain from the native ``utr7`` emissions.

The ``utr7`` array of a sequence is f16, shape ``[2, 7, L]`` (plane 0 is '+', plane 1 is '−', genome coordinates):

  0  log p(state 0)              intergenic
  1  log(p15+p19+p20)            5′ UTR exon
  2  log p16                     5′ UTR intron
  3  log(p17+p21+p22)            3′ UTR exon
  4  log p18                     3′ UTR intron
  5  TSS logit                   site bit 4
  6  TES logit                   site bit 5

Donor and acceptor come from the decoder emissions ``c`` (channels 0 and 1, probabilities).

For each predicted CDS chain the 5′ end is the upstream position (≤ 20 kb, not crossing
another predicted CDS on the same strand) with the highest TSS score whose run back to the
CDS stays acceptable: every base has 5′ UTR-exon posterior ≥ θ or 5′ UTR-intron posterior
≥ θ_i, and the chosen base itself is exonic (≥ θ). The 3′ end is the same with TES and the
3′ channels. UTR introns are chosen by a splice-gated dynamic program inside each span:
non-overlapping introns [i, j] maximising log s_low(i) + log s_high(j) +
Σ_{k=i..j} log((p_in(k) + ε) / (p_ex(k) + ε)), with τ = .05, ε = 1e-4, minimum
intron 30; the outer base of the span stays exon and an intron is kept only if it
raises the total score.

Each side has its own (θ, θ_i, gap, kappa): the gap G lets a stretch of at most G
consecutive bad bases (exon < θ and intron < θ_i) stay inside the run without ending it
(bases inside a crossed stretch are never chosen); kappa κ picks the base maximising
TSS/TES logit − κ × distance-from-the-CDS-edge in kb, ties farthest. The 5′ end choice
takes three more knobs: e_min, a threshold on the de-biased TSS logit (raw − ln 100) with
a none / median fallback (median attaches a given median 5′ length inside the accepted
span), and λ_len, the weight of a terminal-length prior on the span. The released decoder
calls this module with G = κ = 0 and the neutral knobs (e_min −inf, fallback none, λ_len 0).
"""
import bisect

import numpy as np

from ancestra.decode.tx import make_tx, merge

# utr7 channel index. Collapsed3 is not read except c[donor], c[acceptor].
CH_INTERGENIC, CH_UTR5_EXON, CH_UTR5_INTRON, CH_UTR3_EXON, CH_UTR3_INTRON, CH_TSS, CH_TES = range(7)
CH_DONOR, CH_ACCEPTOR = 0, 1
WINDOW = 20_000
MIN_INTRON = 30
# Splice-gated intron DP settings (fixed in advance, not tuned).
DP_TAU = 0.05
DP_EPS = 1e-4
# TSS de-bias for the 5′ end score: the site head trains with BCE pos_weight 100,
# so the de-biased logit is raw − ln 100 — the same convention as the semi-Markov
# decoder (ancestra.decode.utr_decode).
LN100 = float(np.log(100.0))


def _plane(strand):
    return 0 if strand == "+" else 1


def _exp(logp, side="?", pos=0, group="?"):
    a = np.asarray(logp, np.float32)
    bad = np.flatnonzero(np.isnan(a))
    if len(bad):
        raise ValueError("NaN in %s log cache at position %d (group %s)" % (side, int(pos) + int(bad[0]), group))
    pinf = np.flatnonzero(np.isposinf(a))
    if len(pinf):
        raise ValueError("inf in %s log cache at position %d (group %s)" % (side, int(pos) + int(pinf[0]), group))
    return np.exp(np.clip(a, -80.0, 0.0))


def _lp_lookup(logp, edges, length):
    """Log prior of one span length under binned training counts (right-continuous bins)."""
    i = int(np.searchsorted(np.asarray(edges, dtype=np.int64), int(length), side="right"))
    return float(np.asarray(logp, dtype=np.float64)[i])


def _windows(txs, window5=None, window3=None):
    """Per transcript, genome intervals (lo5, hi5, lo3, hi3) that do not enter another same-strand CDS.

    window5 / window3 bound the 5′ / 3′ reach (None means WINDOW). The 3′ window stays
    at WINDOW unless given. Same-strand chains whose
    CDS span overlaps this chain's own span are the same locus (alternative isoforms)
    and do not bound it; disjoint neighbours bound it."""
    w5 = WINDOW if window5 is None else window5
    w3 = WINDOW if window3 is None else window3
    by = {}
    for t in txs:
        by.setdefault((t["chrom"], t["strand"]), []).append(t)
    win = {}
    for group in by.values():
        group.sort(key=lambda t: (t["cds"][0][0], t["cds"][-1][1], t["name"]))
        ends = []                               # sorted CDS ends of the earlier chains
        for i, t in enumerate(group):
            cs, ce = t["cds"][0][0], t["cds"][-1][1]
            # the furthest disjoint earlier end (overlapping earlier chains are the same locus: skip)
            j = bisect.bisect_right(ends, cs)
            prev_end = ends[j - 1] if j else None
            # the nearest disjoint later start (sorted by start, so the first one past ce)
            next_start = None
            for o in group[i + 1:]:
                ocs = o["cds"][0][0]
                if ocs >= ce:
                    next_start = ocs
                    break
            if t["strand"] == "+":
                lo5 = cs - w5 if prev_end is None else max(cs - w5, prev_end)
                hi3 = ce + w3 if next_start is None else min(ce + w3, next_start)
                win[id(t)] = (lo5, cs, ce, hi3)
            else:
                hi5 = ce + w5 if next_start is None else min(ce + w5, next_start)
                lo3 = cs - w3 if prev_end is None else max(cs - w3, prev_end)
                win[id(t)] = (lo3, cs, ce, hi5)
            bisect.insort(ends, ce)
    return win


def _best_span(logit, exon_log, intron_log, lo, hi, cds_at_hi, theta, theta_i, gap=0, kappa=0.0, side="?", group="?", pos=None,
               e_min=float("-inf"), fallback="none", lambda_len=0.0, lp_logp=None, lp_edges=None, median_len=None):
    """[lo, hi) array slice adjacent to the CDS. Returns the UTR span in the same index space, or None.

    ``cds_at_hi`` means the CDS starts at ``hi`` (the run grows toward smaller indexes).
    ``gap`` G: a stretch of at most G consecutive bad bases is crossed without ending the run;
    a longer one ends it. ``kappa`` κ: the chosen base maximises logit minus κ times the
    distance from the CDS edge in kb; ties keep the farthest base.
    With G = κ = 0 the chosen base is the highest-logit base, ties farthest.
    The 5′ side takes three more knobs: ``e_min`` (a de-biased
    TSS-logit threshold with a ``none`` / ``median`` fallback), ``lambda_len`` (weight of
    the training length prior on the span) and the caller-side window. Neutral values
    are e_min −inf, fallback none, λ_len 0.
    """
    if hi <= lo:
        return None
    if pos is None:
        pos = lo
    exon = _exp(exon_log[lo:hi], side="%s-exon" % side, pos=pos, group=group)
    intron = _exp(intron_log[lo:hi], side="%s-intron" % side, pos=pos, group=group)
    score = np.asarray(logit[lo:hi], np.float32)
    good = (exon >= theta) | (intron >= theta_i)
    n = len(good)
    bad = np.flatnonzero(~good)
    groups = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1) if len(bad) else []
    if cds_at_hi:
        base = 0
        for g in groups[::-1]:              # nearest the CDS first
            if len(g) > gap:
                base = int(g[-1]) + 1
                break
        region = slice(base, n)
    else:
        stop = n
        for g in groups:                     # nearest the CDS first
            if len(g) > gap:
                stop = int(g[0])
                break
        base = 0
        region = slice(0, stop)
    ok = np.flatnonzero(exon[region] >= theta)   # crossed bad bases fail this filter
    if len(ok) == 0:
        return None
    if side == "5":
        bad = np.flatnonzero(~np.isfinite(score[region][ok]))
        if len(bad):
            raise ValueError("NaN or inf in 5 TSS logit at position %d (group %s)" % (int(pos) + int(region.start) + int(ok[int(bad[0])]), group))
        if e_min != float("-inf") or lambda_len != 0.0 or fallback != "none":
            return _best_span5(lo, hi, cds_at_hi, score, region, ok, base, n, kappa,
                               e_min, fallback, lambda_len, lp_logp, lp_edges, median_len)
    if side == "3":
        bad = np.flatnonzero(~np.isfinite(score[region][ok]))
        if len(bad):
            raise ValueError("NaN or inf in 3 TES logit at position %d (group %s)" % (int(pos) + int(region.start) + int(ok[int(bad[0])]), group))
    sc = score[region][ok]
    if kappa:
        dist = (n - 1 - (base + ok)) if cds_at_hi else ok   # bp from the CDS edge
        sc = sc - kappa * (dist / 1000.0)
    top = ok[sc == sc.max()]
    # equal scores: the base farthest from the CDS (smaller index when the CDS is at hi)
    idx = base + int(top[0] if cds_at_hi else top[-1])
    if cds_at_hi:
        return lo + idx, hi
    return lo, lo + idx + 1


def _best_span5(lo, hi, cds_at_hi, score, region, ok, base, n, kappa,
                e_min, fallback, lambda_len, lp_logp, lp_edges, median_len):
    """5′-only end choice: de-biased score (TSS logit − ln 100), existence threshold
    e_min, median fallback, and a training length prior.

    Candidates are the exonic bases (``ok``, region-relative) with de-biased logit ≥
    e_min. The winner maximises de-biased logit + λ_len · log p_len(span) − κ · dist,
    ties farthest from the CDS as before. p_len is the 5′ terminal-exon training
    prior: at end-choice time the candidate span is a single exon piece (UTR introns
    are chosen afterwards by the DP). With no qualifier the median fallback attaches
    the training-median 5′ length inside the accepted span (``none`` returns None).
    """
    if fallback not in ("none", "median"):
        raise SystemExit("bad 5′ fallback %r: want none or median" % (fallback,))
    if lambda_len != 0.0 and (lp_logp is None or lp_edges is None):
        raise SystemExit("5′ length prior needs lp_logp/lp_edges")
    if fallback == "median" and median_len is None:
        raise SystemExit("5′ median fallback needs median_len")
    deb = score[region][ok].astype(np.float64) - LN100
    q = np.flatnonzero(deb >= e_min)
    if len(q) == 0:
        if fallback != "median":
            return None
        m = int(median_len)
        want = (n - m) if cds_at_hi else (m - 1)   # region-relative index for span length m
        want = min(max(want, base), n - 1)
        near = base + ok
        d = np.abs(near - want)
        tied = near[d == d.min()]
        idx = int(tied.min() if cds_at_hi else tied.max())   # ties farthest from the CDS
        if cds_at_hi:
            return lo + idx, hi
        return lo, lo + idx + 1
    sc = deb[q]
    qq = base + ok[q]                              # slice-relative candidate indexes
    if kappa:
        dist = (n - 1 - qq) if cds_at_hi else (qq - base)   # bp from the CDS edge
        sc = sc - kappa * (dist / 1000.0)
    if lambda_len != 0.0:
        span = (n - qq) if cds_at_hi else (qq - base + 1)   # candidate span lengths
        sc = sc + lambda_len * np.array([_lp_lookup(lp_logp, lp_edges, int(v)) for v in span])
    top = q[sc == sc.max()]
    # equal scores: the base farthest from the CDS (smaller index when the CDS is at hi)
    idx = base + int(ok[int(top[0])] if cds_at_hi else ok[int(top[-1])])
    if cds_at_hi:
        return lo + idx, hi
    return lo, lo + idx + 1


def _introns(intron_log, exon_log, donor, acceptor, lo, hi, plus, side="?", group="?", pos=None):
    """Splice-gated intron DP inside [lo, hi) (array indexes), transcript-order donor then acceptor.

    Chooses non-overlapping introns [i, j] (inclusive) maximising log s_low(i) +
    log s_high(j) + Σ_{k=i..j} log((p_in(k) + ε) / (p_ex(k) + ε)), where s_low /
    s_high is donor / acceptor on '+' and acceptor / donor on '−'. Candidate
    endpoints need s ≥ τ (DP_TAU); minimum intron MIN_INTRON; an intron is kept
    only if it raises the total score (empty set scores 0). p_in / p_ex are this
    side's UTR-intron / UTR-exon posteriors. The caller keeps the span's outer
    base as exon by searching strictly inside the span.
    """
    if hi - lo < MIN_INTRON or donor is None or acceptor is None:
        return []
    if pos is None:
        pos = lo
    for _ch, _arr in (("donor", donor), ("acceptor", acceptor)):
        _a = np.asarray(_arr[lo:hi], dtype=np.float64)
        _bad = np.flatnonzero(~np.isfinite(_a) | (_a < 0) | (_a > 1))
        if len(_bad):
            raise ValueError("bad value %r in %s splice track (side %s) at position %d (group %s)"
                             % (float(_a[int(_bad[0])]), _ch, side, int(pos) + int(_bad[0]), group))
    n = int(hi - lo)
    p_in = _exp(intron_log[lo:hi], side="%s-intron" % side, pos=pos, group=group).astype(np.float64)
    p_ex = _exp(exon_log[lo:hi], side="%s-exon" % side, pos=pos, group=group).astype(np.float64)
    w = np.log((p_in + DP_EPS) / (p_ex + DP_EPS))
    prefix = np.empty(n + 1, np.float64)
    prefix[0] = 0.0
    np.cumsum(w, out=prefix[1:])
    if plus:
        s_low = np.asarray(donor[lo:hi], dtype=np.float64)
        s_high = np.asarray(acceptor[lo:hi], dtype=np.float64)
    else:
        s_low = np.asarray(acceptor[lo:hi], dtype=np.float64)
        s_high = np.asarray(donor[lo:hi], dtype=np.float64)
    dp = np.zeros(n + 1, np.float64)
    take = np.full(n + 1, -1, np.int64)   # take[k]: start i of the intron ending at k - 1, else -1
    best_val = float("-inf")
    best_i = -1
    for k in range(n):
        i_new = k - MIN_INTRON + 1        # newly eligible start for an intron ending at k
        if i_new >= 0 and s_low[i_new] >= DP_TAU:
            v = dp[i_new] + float(np.log(s_low[i_new])) - prefix[i_new]
            if v > best_val:
                best_val = v
                best_i = i_new
        dp[k + 1] = dp[k]
        if best_i >= 0 and s_high[k] >= DP_TAU:
            cand = best_val + prefix[k + 1] + float(np.log(s_high[k]))
            if cand > dp[k]:
                dp[k + 1] = cand
                take[k + 1] = best_i
    out = []
    k = n
    while k > 0:
        if take[k] >= 0:
            i = int(take[k])
            out.append((lo + i, lo + k))
            k = i
        else:
            k -= 1
    out.reverse()
    return out


def _pieces(lo, hi, introns):
    cuts = [lo]
    for a, b in introns:
        cuts.extend((a, b))
    cuts.append(hi)
    return [(a, b) for a, b in zip(cuts[0::2], cuts[1::2]) if b > a]


def _attach_one(tx, origin, utr7, c, span, theta, theta_i, five=None, three=None,
                e_min5=float("-inf"), fallback5="none", lambda5=0.0, lp5=None, med5=None):
    """utr7 [7, L] and c [≥2, L] are the transcript's strand plane. span is (low_stop, cds_start,
    cds_end, high_stop) in genome coordinates.

    ``five`` / ``three`` are (θ, θ_i, gap, kappa) side parameters; either defaults to
    (theta, theta_i, 0, 0). UTR introns use the fixed splice-gated DP (τ .05, ε 1e-4).
    The 5′ end choice takes e_min5 / fallback5 / lambda5 with the training prior lp5
    = (logp, edges) and median med5; the 3′ call takes none of them."""
    low_stop, cs, ce, high_stop = span
    L = utr7.shape[-1]
    origin = int(origin)
    t5, ti5, g5, k5 = tuple(five) if five is not None else (theta, theta_i, 0, 0.0)
    t3, ti3, g3, k3 = tuple(three) if three is not None else (theta, theta_i, 0, 0.0)
    lp_logp, lp_edges = lp5 if lp5 is not None else (None, None)

    def sl(g0, g1):
        return max(0, g0 - origin), min(L, g1 - origin)

    def arr(ch):
        return utr7[ch]

    donor = None if c is None else c[CH_DONOR]
    acceptor = None if c is None else c[CH_ACCEPTOR]
    plus = tx["strand"] == "+"
    # 5′ is the low side on '+', the high side on '−'. The CDS sits at the inner edge.
    if plus:
        a, b = sl(low_stop, cs)
        five = _best_span(arr(CH_TSS), arr(CH_UTR5_EXON), arr(CH_UTR5_INTRON), a, b, True, t5, ti5, g5, k5, side="5", group=tx["chrom"], pos=origin + a,
                          e_min=e_min5, fallback=fallback5, lambda_len=lambda5, lp_logp=lp_logp, lp_edges=lp_edges, median_len=med5)
        a, b = sl(ce, high_stop)
        three = _best_span(arr(CH_TES), arr(CH_UTR3_EXON), arr(CH_UTR3_INTRON), a, b, False, t3, ti3, g3, k3, side="3", group=tx["chrom"], pos=origin + a)
    else:
        a, b = sl(ce, high_stop)
        five = _best_span(arr(CH_TSS), arr(CH_UTR5_EXON), arr(CH_UTR5_INTRON), a, b, False, t5, ti5, g5, k5, side="5", group=tx["chrom"], pos=origin + a,
                          e_min=e_min5, fallback=fallback5, lambda_len=lambda5, lp_logp=lp_logp, lp_edges=lp_edges, median_len=med5)
        a, b = sl(low_stop, cs)
        three = _best_span(arr(CH_TES), arr(CH_UTR3_EXON), arr(CH_UTR3_INTRON), a, b, True, t3, ti3, g3, k3, side="3", group=tx["chrom"], pos=origin + a)

    def genome(span_ab):
        if span_ab is None:
            return None
        return origin + span_ab[0], origin + span_ab[1]

    def side_pieces(span_g, intron_ch, exon_ch, outer_low, side):
        if span_g is None:
            return []
        g0, g1 = span_g
        if g1 - g0 < 2:
            return [(g0, g1)]
        # the outer base stays an exon; introns are searched strictly inside the span
        a, b = sl(g0 + 1, g1) if outer_low else sl(g0, g1 - 1)
        runs = _introns(arr(intron_ch), arr(exon_ch), donor, acceptor, a, b, plus, side=side, group=tx["chrom"], pos=origin + a) if b > a else []
        return _pieces(g0, g1, [(origin + x, origin + y) for x, y in runs])

    five_g, three_g = genome(five), genome(three)
    # outer base of a 5′ UTR is low on '+', high on '−'. A 3′ UTR is the opposite.
    extra = side_pieces(five_g, CH_UTR5_INTRON, CH_UTR5_EXON, plus, "5") + side_pieces(three_g, CH_UTR3_INTRON, CH_UTR3_EXON, not plus, "3")
    blocks = merge(extra + list(tx["cds"]))
    made = make_tx(tx["chrom"], tx["strand"], tx["name"], blocks,
                   (tx["cds"][0][0], tx["cds"][-1][1]), gene=tx["gene"])
    return made if made is not None else tx
