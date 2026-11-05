"""The optional ``--alt`` track: alternative isoforms beside each primary.

Per item, after the 5' UTR decoder:

- **Window:** around each Viterbi primary, 5 kb (``alt.json`` pool.flank) clipped at the item and at the nearest
  non-overlapping same-strand primaries (``windows``).
- **Pool** (``pool_one``): FFBS chains of ``decode_mbr`` at T 1, 2 and 5 (seed: the window's 5′ coordinate) and
  ``structural_alternatives`` at tau .5 and .2, every key but the primary's, with 11 values each.
- **Score:** the 15 ``features``, then ``score``: fsum logistic and the isotonic map of ``alt.json``.
- **Contract:** keep p >= T, drop candidates that share no CDS block with the primary, then walk the primaries in
  primary-BED order and each gene's survivors by (−p, canonical identity): the candidate with the primary's UTRs
  (``_copy``), the isoform grammar (``valid_isoform``), distinct CDS keys (the first gene keeps a shared key), at most
  ``cap`` per gene. Kept rows are ranked 1..n in walk order.

``run_item`` runs ``pool_one`` on the worker's spawn pool, at most 2·n_alt tasks in flight, and walks only once every
result of the item is in, so no output depends on the worker count. It writes the item's ``<key>.alt.tsv`` part: the 21
columns of the alt bigBed. This module imports numpy, the compiled binding and ancestra.decode.tx, never torch.
"""
import math
import multiprocessing
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from ancestra import _native
from ancestra.decode.tx import intron_ok, make_tx, merge, valid_isoform, vrow

LOG_FLOOR = -69.0  # viterbi.LOG_FLOOR (~log 1e-30), keeps windows on -inf/nan emissions decodable
PART = ('exon_region', 'exon_phase', 'cds_base_bias', 'intron_region', 'intron_phase')
FEATS = ('ddec', 'dset', 'events', 'minc', 'npriv', 'share1', 'share2', 'share5', 'ffbs', 's_alpha0', 's_alpha1',
         'ffflag', 's05flag', 's02flag', 'feasible')
VALS = ('share1', 'share2', 'share5', 'ffbs', 'ddec', 'dset', 'events', 'minc', 'npriv', 's_alpha0', 's_alpha1')
MIN_INTRON = 30  # the isoform grammar's and the UTR guard's minimum intron, not the decoder's own parameter
_RC = bytes.maketrans(b"ACGTacgt", b"TGCAtgca")


def _split_bias(params):
    """(params_for_rust, b_cds, b_int); mirrors viterbi._split_a_bias."""
    p = dict(params or {})
    return p, float(p.pop("a_bias_cds", 0.0)), float(p.pop("a_bias_int", 0.0))


def _apply_bias(la, b_cds, b_int):
    """CDS/intron offsets on rows 1/2 of a [3, L] array; mirrors viterbi._apply_a_bias."""
    if not b_cds and not b_int:
        return la
    la = la.copy()
    if b_cds:
        la[1, :] += b_cds
    if b_int:
        la[2, :] += b_int
    return la


def _finite(la, lb, c):
    """-inf/nan log-probabilities -> LOG_FLOOR, nan site probabilities -> 0 (viterbi._finite)."""
    for x in (la, lb):
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = LOG_FLOOR
    bad = ~np.isfinite(c)
    if bad.any():
        c[bad] = 0.0


def revcomp(seqb):
    return seqb.translate(_RC)[::-1]


class Win:
    """One locus window: per-strand emissions (A bias applied), sequence in transcript order."""

    def __init__(self, ws, we, strand, la, lb, c, seq):
        self.ws, self.we, self.strand = ws, we, strand
        self.L = we - ws
        self.la, self.lb, self.c = la, lb, c  # genomic order, [3, L] / [3, L] / [4, L] f32
        self.seq = seq  # genomic order bytes
        self.seq_t = revcomp(seq) if strand == "-" else seq
        self.cc_t = np.ascontiguousarray(c[:, ::-1]) if strand == "-" else c

    def donor_ok(self, t):
        """GT at transcript (t, t+1): donor site t = first intron base (decoder convention)."""
        return t >= 0 and t + 2 <= self.L and self.seq_t[t] == 0x47 and self.seq_t[t + 1] == 0x54

    def acc_ok(self, t):
        """AG at transcript (t-1, t): acceptor site t = last intron base."""
        return 1 <= t < self.L and self.seq_t[t - 1] == 0x41 and self.seq_t[t] == 0x47

    def start_ok(self, t):
        return 0 <= t and t + 3 <= self.L and self.seq_t[t:t + 3] == b"ATG"

    def stop_ok(self, t):
        return 0 <= t and t + 3 <= self.L and self.seq_t[t:t + 3] in (b"TAA", b"TAG", b"TGA")

    def site_p(self, kind, t):
        """kind 0 donor, 1 acceptor, 2 start, 3 stop at transcript site t."""
        if t < 0 or t >= self.L:
            return 0.0
        return float(self.cc_t[kind, t])

    def positions(self, kind, lo, hi, tau):
        """transcript positions in [lo, hi) whose site probability is >= tau."""
        lo, hi = max(0, lo), min(self.L, hi)
        if hi <= lo:
            return []
        return [int(t) for t in np.flatnonzero(self.cc_t[kind, lo:hi] >= tau) + lo]

    def tx_to_genomic(self, ptx):
        """transcript blocks -> ascending absolute genomic blocks (chain_terms input)."""
        if self.strand == "-":
            return tuple(sorted((self.we - b, self.we - a) for a, b in ptx))
        return tuple(sorted((self.ws + a, self.ws + b) for a, b in ptx))

    def genomic_to_tx(self, blocks):
        """ascending absolute genomic blocks -> transcript blocks (start codon first)."""
        if self.strand == "-":
            return [(self.we - e, self.we - s) for s, e in reversed(blocks)]
        return [(s - self.ws, e - self.ws) for s, e in blocks]


def structural_alternatives(win, ptx, cfg):
    """Primary chain transcript blocks -> [(blocks_tx, kind)] alternative structures.

    Families: internal-exon skips; cassette exons inside primary
    introns (acceptor/donor p >= tau, GT/AG motifs, in-frame CDS, residual introns >= intron_min);
    alternative 5'/3' splice sites within +-alt_site nt (p >= tau, motif valid, frame kept);
    alternative first/last coding exon (a new terminal block at another donor/acceptor with an
    ATG/stop head, or another in-frame start/stop at the same boundary). Duplicates collapse.
    """
    tau = cfg["tau"]
    intron_min, exon_min, cas_min = cfg["intron_min"], cfg["exon_min"], cfg["cas_min"]
    exon_max, alt = cfg["exon_max"], cfg["alt_site"]
    L = win.L
    n = len(ptx)
    out = {}

    def add(blocks, kind):
        if blocks and all(0 <= a < b <= L for a, b in blocks):
            out.setdefault(tuple(blocks), kind)

    # 1. skip each internal exon
    for i in range(1, n - 1):
        add(ptx[:i] + ptx[i + 1:], "skip")

    # 2. cassette exons inside each primary intron
    for i in range(n - 1):
        u, v = ptx[i][1], ptx[i + 1][0]
        if v - u < 2 * intron_min + cas_min:
            continue
        accs = [(t, win.site_p(1, t)) for t in win.positions(1, u, v, tau) if win.acc_ok(t)]
        dons = [(t, win.site_p(0, t)) for t in win.positions(0, u, v, tau) if win.donor_ok(t)]
        pairs = []
        for a, pa in accs:
            if a - u + 1 < intron_min:
                continue
            for d, pd in dons:
                ln = d - a - 1
                if d <= a + 1 or not (cas_min <= ln <= exon_max) or ln % 3 or v - d < intron_min:
                    continue
                pairs.append((pa * pd, a, d))
        pairs.sort(reverse=True)
        for _sc, a, d in pairs[:cfg["cas_per_intron"]]:
            add(ptx[:i + 1] + [(a + 1, d)] + ptx[i + 1:], "cassette")

    # 3. alternative 5'/3' splice sites within +-alt nt (frame kept)
    for i, (a, b) in enumerate(ptx):
        if i < n - 1:  # 5' site = the donor at the block end
            v = ptx[i + 1][0]
            for d in range(max(0, b - alt), min(L, b + alt + 1)):
                if d == b or (d - b) % 3 or d - a < 3 or v - d < intron_min:
                    continue
                if win.donor_ok(d) and win.site_p(0, d) >= tau:
                    add(ptx[:i] + [(a, d)] + ptx[i + 1:], "alt5")
        if i > 0:  # 3' site = the acceptor at the block start
            u = ptx[i - 1][1]
            for acc in range(max(1, a - 1 - alt), min(L, a - 1 + alt + 1)):
                if acc == a - 1 or (acc - (a - 1)) % 3 or acc - u + 1 < intron_min or b - acc - 1 < 3:
                    continue
                if win.acc_ok(acc) and win.site_p(1, acc) >= tau:
                    add(ptx[:i] + [(acc + 1, b)] + ptx[i + 1:], "alt3")

    # 4. alternative first/last coding exon
    starts = [(t, win.site_p(2, t)) for t in win.positions(2, 0, L, tau) if win.start_ok(t)]
    stops = [(t, win.site_p(3, t)) for t in win.positions(3, 0, L, tau) if win.stop_ok(t)]
    if n >= 2:
        a1, d1 = ptx[0]
        frame1 = (d1 - a1) % 3
        # 4a. new first block at another donor, spliced to block 2 (intron >= intron_min)
        dpairs = []
        for d in win.positions(0, 0, ptx[1][0] - intron_min + 1, tau):
            if d == d1 or not win.donor_ok(d):
                continue
            pd = win.site_p(0, d)
            for s, ps in starts:
                ln = d - s
                if s < d and exon_min <= ln <= exon_max and ln % 3 == frame1:
                    dpairs.append((ps * pd, s, d))
        dpairs.sort(reverse=True)
        for _sc, s, d in dpairs[:cfg["term_cap"]]:
            add([(s, d)] + ptx[1:], "afe")
        # 4b. another in-frame start at the same first donor
        spairs = [(ps, s) for s, ps in starts
                  if s != a1 and exon_min <= d1 - s <= exon_max and (d1 - s) % 3 == frame1]
        spairs.sort(reverse=True)
        for ps, s in spairs[:cfg["term_cap"]]:
            add([(s, d1)] + ptx[1:], "alt_start")
        an, bn = ptx[n - 1]
        frame_n = (bn - an) % 3
        # 4c. new last block from another acceptor (intron >= intron_min) to a stop end
        apairs = []
        for acc in win.positions(1, ptx[n - 2][1] + intron_min - 1, L, tau):
            if acc == an - 1 or not win.acc_ok(acc):
                continue
            pa = win.site_p(1, acc)
            for t, ps in stops:
                e = t + 3
                ln = e - (acc + 1)
                if e > acc + 1 and exon_min <= ln <= exon_max and ln % 3 == frame_n:
                    apairs.append((ps * pa, acc + 1, e))
        apairs.sort(reverse=True)
        for _sc, s, e in apairs[:cfg["term_cap"]]:
            add(ptx[:n - 1] + [(s, e)], "ale")
        # 4d. another in-frame stop end at the same last acceptor
        tpairs = [(ps, t + 3) for t, ps in stops
                  if t + 3 != bn and exon_min <= t + 3 - an <= exon_max and (t + 3 - an) % 3 == frame_n]
        tpairs.sort(reverse=True)
        for ps, e in tpairs[:cfg["term_cap"]]:
            add(ptx[:n - 1] + [(an, e)], "alt_stop")
    else:
        a1, d1 = ptx[0]
        for ps, s in sorted(((ps, s) for s, ps in starts if s != a1 and exon_min <= d1 - s), reverse=True)[:cfg["term_cap"]]:
            add([(s, d1)], "alt_start")
        for ps, e in sorted(((ps, t + 3) for t, ps in stops if t + 3 != d1 and exon_min <= t + 3 - a1), reverse=True)[:cfg["term_cap"]]:
            add([(a1, e)], "alt_stop")
    return [(list(k), kind) for k, kind in out.items()]


# --- the window and the pool -----------------------------------------------------------------------------------------


def windows(vit, start, end, flank):
    """[ws, we) around each Viterbi primary of one item, in ``vit``'s order. ``vit`` holds
    (name, chrom, strand, ascending CDS blocks). Among the same-strand primaries ordered by CDS start, ws stops at the
    largest CDS end <= b0.start before it and we at the smallest CDS start >= b_last.end after it; overlapping
    primaries do not clip. None (``skip_window``) when no flank is left on a side."""
    by = defaultdict(list)
    for q in vit:
        by[(q[1], q[2])].append(q)
    out = {}
    for group in by.values():
        group.sort(key=lambda q: q[3][0][0])
        for i, q in enumerate(group):
            b = q[3]
            left = max((x[3][-1][1] for x in group[:i] if x[3][-1][1] <= b[0][0]), default=0)
            right = min((x[3][0][0] for x in group[i + 1:] if x[3][0][0] >= b[-1][1]), default=10 ** 18)
            ws, we = max(start, b[0][0] - flank, left), min(end, b[-1][1] + flank, right)
            out[q[0]] = (ws, we) if ws < b[0][0] and we > b[-1][1] else None
    return [out[q[0]] for q in vit]


def _window(task):
    """The task's window: its f16 arrays as f32 C-contiguous, ``_finite``, then the A biases off the Rust params."""
    la, lb, c = (np.ascontiguousarray(np.asarray(x), dtype=np.float32) for x in task["arrays"])
    _finite(la, lb, c)
    p, b_cds, b_int = _split_bias(task["params"])
    la = _apply_bias(la, b_cds, b_int)
    return Win(task["ws"], task["we"], task["strand"], la, lb, c, task["seq"]), p


def terms(win, blocks, params):
    """(S_dec, S_set) of one chain in transcript blocks: chain_terms' total, and the total with the
    A-region terms replaced by the summed CDS-vs-intron log-odds."""
    t = _native.chain_terms(win.seq, win.strand, win.la, win.lb, win.c, win.ws, win.tx_to_genomic(blocks), params)
    total = float(t['total'])
    parts = sum(float(t[k]) for k in ('gene_start', 'site_start', 'site_donor', 'site_acceptor', 'site_stop', *PART,
                                      'intron_geom', 'exon_short'))
    assert (math.isfinite(total) and abs(total - parts) < 2e-5 * max(1., abs(total))) or (
        total == -math.inf and math.isfinite(parts)), (total, parts)
    tx = win.genomic_to_tx(win.tx_to_genomic(blocks))
    a, b = tx[0][0], tx[-1][1]
    la = win.la[:, ::-1] if win.strand == '-' else win.la
    z = np.logaddexp(la[1, a:b], la[2, a:b]) - la[0, a:b]
    sset = total - sum(float(t[k]) for k in PART) + float(z.sum())
    return total, sset


def pool_one(task):
    """One primary's candidate pool: (ordinal, [(key, source, values)], VmHWM kB).

    ``task``: ordinal, strand, ws, we, seq (upper-case bytes of [ws, we)), V (the Viterbi CDS, ascending genomic
    blocks), params (the declared Viterbi params), cfg (``alt.json``'s pool) and ``arrays`` (the strand plane's f16 la,
    lb, c over the window). ``key`` is ascending genomic CDS blocks, ``source`` (``ffbs``, ``struct0.5``, ``struct0.2``,
    sorted), ``values`` the 11 of ``VALS``. The
    list is None when S_dec(V) is not finite on the window (``skip_primary_infeasible``)."""
    win, p = _window(task)
    cfg, strand, ws, we = task["cfg"], task["strand"], task["ws"], task["we"]
    pblocks = task["V"]
    ptx = win.genomic_to_tx(pblocks)
    pdec, pset = terms(win, ptx, p)
    if not math.isfinite(pdec):
        return task["ordinal"], None, _vmhwm_kb()
    rp = dict(_native.default_params(), **p)  # the binding's resolved params: no Python copy of a Rust default
    scfg = dict(cfg["struct"], intron_min=int(rp["intron_min"]), exon_max=int(rp["exon_max"]))
    mbr = cfg["mbr"]
    temps = mbr["temperatures"]
    primary = tuple(pblocks)
    shares = {T: {} for T in temps}
    for T in temps:
        # decode_mbr's sample_seed XORs the minus constant again, so the effective seed is the 5' coordinate
        pos5 = ws if strand == '+' else we - 1
        seed = pos5 if strand == '+' else pos5 ^ 0xa5a55a5ad00df00d
        prim, low = _native.decode_mbr(win.seq, strand, win.la, win.lb, win.c, ws, p, n_samples=mbr["n_samples"],
                                           temperature=T, tau=mbr["tau"], seed=seed)
        for d in prim:
            chains = [(d['cds'], d['posterior'])] + list(d.get('alternatives', ()))
            for cds, post in chains:
                key = tuple(tuple(x) for x in cds)
                if key != primary:
                    shares[T][key] = max(shares[T].get(key, 0.), float(post))
        for d in low:
            key = tuple(tuple(x) for x in d['cds'])
            if key != primary:
                shares[T][key] = max(shares[T].get(key, 0.), float(d['posterior']))
    src = defaultdict(set)
    for T in temps:
        for key in shares[T]:
            src[key].add('ffbs')
    for tau in cfg["struct"]["taus"]:
        for btx, _kind in structural_alternatives(win, ptx, dict(scfg, tau=tau)):
            key = win.tx_to_genomic(btx)
            if key != primary:
                src[key].add('struct' + str(tau))
    out = []
    for key, sources in src.items():
        tx = win.genomic_to_tx(key)
        ddec, dset = terms(win, tx, p)
        intr = lambda x: {(x[i][1], x[i + 1][0]) for i in range(len(x) - 1)}  # noqa: E731
        ep = len(intr(tx) ^ intr(ptx)) + (tx[0][0] != ptx[0][0]) + (tx[-1][1] != ptx[-1][1])
        pd = {(ptx[i][1], 0) for i in range(len(ptx) - 1)} | {(ptx[i + 1][0] - 1, 1) for i in range(len(ptx) - 1)}
        priv = []
        for i in range(len(tx) - 1):
            for pos, k in ((tx[i][1], 0), (tx[i + 1][0] - 1, 1)):
                if (pos, k) not in pd:
                    priv.append(float(win.cc_t[k, pos]))
        sh = [shares[T].get(key, 0.) for T in temps]
        out.append((key, ','.join(sorted(sources)), (*sh, max(sh), ddec - pdec, dset - pset, ep,
                                                     min(priv) if priv else 1., len(priv), dset - ep, ddec - ep)))
    return task["ordinal"], out, _vmhwm_kb()


def features(x):
    """The 15 features of one pool row: ``VALS`` and ``source``, as text or as numbers.
    A non-finite value among the first 11 becomes 0."""
    src = x['source'].split(',')
    return [float(x[k]) if math.isfinite(float(x[k])) else 0. for k in FEATS[:11]] + [
        float('ffbs' in src), float('struct0.5' in src), float('struct0.2' in src), float(math.isfinite(float(x['ddec'])))]


def score(f, model):
    """z = fsum(β0, βj·(fj − μj)/σj) over Python floats in feature order, then p = the isotonic map,
    clamped to its end values."""
    mu, sd, beta, ix, iy = model["mu"], model["sd"], model["beta"], model["iso_x"], model["iso_y"]
    z = math.fsum([beta[0]] + [beta[j + 1] * ((f[j] - mu[j]) / sd[j]) for j in range(len(f))])
    return float(np.interp(z, ix, iy, left=iy[0], right=iy[-1]))


# --- the post-score contract ------------------------------------------------------------------------------------------


def _first(k, st):
    return k[0] if st == "+" else k[-1]


def _last(k, st):
    return k[-1] if st == "+" else k[0]


def _canon(k, st):
    """The canonical identity: the CDS blocks in transcript order, as the window's transcript coordinates sort them."""
    return k if st == "+" else tuple((-e, -s) for s, e in reversed(k))


def _copy(x, V, E, ch, chrom, st, name, guard):
    """The candidate x′ (START inherited as start_extend.apply_one extends the primary) and its transcript, with
    E's UTR pieces on each side that matches V's (5′ also E's) terminal coding block and passes the guard."""
    m5v, m3 = _first(x, st) == _first(V, st), _last(x, st) == _last(V, st)
    m5 = m5v or _first(x, st) == _first(E["cds"], st)
    xp = list(x)
    if m5v and ch is not None:
        i = 0 if st == "+" else -1
        a, b = xp[i]
        assert (a if st == "+" else b) == ch["old_start"], (name, x, ch)
        xp[i] = (ch["new_start"], b) if st == "+" else (a, ch["new_start"])
    xp = tuple(xp)
    assert not m5 or _first(xp, st) == _first(E["cds"], st), (name, x)
    c5, c3 = m5 and guard["5"], m3 and guard["3"]
    t = make_tx(chrom, st, name, merge(list(xp) + list(E["five_iv"] if c5 else ()) + list(E["three_iv"] if c3 else ())),
                (xp[0][0], xp[-1][1]))
    assert t["cds"] == xp, (name, x)
    return t, "5,3" if c5 and c3 else "5" if c5 else "3" if c3 else "none"


def emit(vit, txs, changes, cands, genome, T, cap):
    """The post-score walk of one item, after every candidate is scored. ``vit``: [(name, chrom, strand, V)] in primary-BED order;
    ``txs`` and ``changes``: the 5' UTR decoder's transcript and the START record by name; ``cands``: {name: [(x, source, p)]},
    x ascending genomic CDS blocks; ``genome``: a str fetch (``Str``). Returns the kept [(name, rank, transcript, p,
    source, utr)] in walk order and per-strand counts (candidates, above_threshold, the drops by reason, emitted,
    genes_with_alt, ``utr <mix>``)."""
    counts = {"+": Counter(), "-": Counter()}
    prim = defaultdict(set)  # (chrom, strand) -> the item's emitted primary CDS keys
    for name, chrom, st, _V in vit:
        prim[(chrom, st)].add(txs[name]["cds"])
    kept, rows = defaultdict(dict), []  # (chrom, strand) -> {kept CDS key: gene}
    for name, chrom, st, V in vit:
        E, c, n, surv = txs[name], counts[st], 0, []
        for x, source, p in cands.get(name, ()):
            c["candidates"] += 1
            if p < T:
                continue
            c["above_threshold"] += 1
            if set(x) & set(V):  # (a) shares a CDS block with V; a failure (p2) takes no rank, cap slot or key
                surv.append((x, source, p))
            else:
                c["p2"] += 1
        if surv:
            guard = {side: all(intron_ok(genome, chrom, st, a1, b0, MIN_INTRON) for a1, b0 in E["utr_introns"][side])
                     for side in "53"}
        for x, source, p in sorted(surv, key=lambda r: (-r[2], _canon(r[0], st))):
            t, utr = _copy(x, V, E, changes.get(name), chrom, st, name, guard)
            key = t["cds"]
            if not valid_isoform(vrow(t), genome, MIN_INTRON)[0]:  # (b)
                why = "grammar"
            elif key == E["cds"]:  # (c)
                why = "primary"
            elif key in prim[(chrom, st)]:
                why = "other_primary"
            elif key in kept[(chrom, st)]:
                why = "dup" if kept[(chrom, st)][key] == name else "dup_xgene"
            elif n >= cap:  # (d)
                why = "cap"
            else:
                n += 1
                kept[(chrom, st)][key] = name
                rows.append((name, n, t, p, source, utr))
                c["emitted"] += 1
                c["utr " + utr] += 1
                continue
            c[why] += 1
        c["genes_with_alt"] += n > 0
    return rows, counts


# --- one item --------------------------------------------------------------------------------------------------------


def _watch_parent():
    """Pool initializer: end this worker once its parent is gone (SIGKILL/SIGTERM to the parent orphans spawn
    children, so poll getppid). Daemon thread, 0.5 s poll."""
    parent = os.getppid()

    def _watch():
        while True:
            time.sleep(0.5)
            try:
                if os.getppid() != parent:
                    os._exit(1)
            except Exception:
                pass
    threading.Thread(target=_watch, daemon=True).start()


def pool(n):
    """A worker's alt pool: ``n`` spawn processes, which inherit none of the parent's memory, CUDA state or threads.
    They start at the first submit. Workers exit when their parent dies (the initializer's watchdog)."""
    return ProcessPoolExecutor(n, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_watch_parent)


def model_fault(doc):
    """None when ``doc`` is a usable ``alt.json``, else what is wrong with it: the CLI exits 1 before any item
    instead of failing in the first alt step, and an unsorted isotonic map would interpolate silently."""
    try:
        m, p, fit = doc["model"], doc["pool"], doc["fit"]
        mbr, struct = p["mbr"], p["struct"]
        vals = [x for k in ("mu", "sd", "beta", "iso_x", "iso_y") for x in m[k]]
        checks = ((doc["schema"] == "ancestra-alt-1", "schema is not ancestra-alt-1"),
                  (doc["features"] == list(FEATS), f"features are not {list(FEATS)}"),
                  (isinstance(fit, dict) and [len(m[k]) for k in ("mu", "sd", "beta")] == [15, 15, 16]
                   and 0 < len(m["iso_x"]) == len(m["iso_y"]),
                   "model is not mu[15], sd[15], beta[16], iso_x[k], iso_y[k]"),
                  (all(type(x) in (int, float) and math.isfinite(x) for x in vals)
                   and all(x > 0 for x in m["sd"]),
                   "model values are not finite or sd is not positive"),
                  (all(a < b for a, b in zip(m["iso_x"], m["iso_x"][1:])), "model.iso_x is not increasing"),
                  (all(0 <= a <= b <= 1 for a, b in zip(m["iso_y"], m["iso_y"][1:]))
                   and bool(m["iso_y"]) and 0 <= m["iso_y"][0] <= 1,
                   "model.iso_y is not a nondecreasing probability map"),
                  (type(doc["cap"]) is int and doc["cap"] >= 1, "cap is not a positive integer"),
                  (type(p["flank"]) is int and p["flank"] >= 0
                   and type(mbr["n_samples"]) is int and mbr["n_samples"] >= 1
                   and isinstance(mbr["temperatures"], list) and bool(mbr["temperatures"])
                   and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in mbr["temperatures"])
                   and type(mbr["tau"]) in (int, float) and 0 < mbr["tau"] <= 1
                   and isinstance(struct["taus"], list) and bool(struct["taus"])
                   and all(type(x) in (int, float) and math.isfinite(x) and 0 < x <= 1 for x in struct["taus"])
                   and all(type(struct[k]) is int and struct[k] >= 1 for k in
                           ("exon_min", "cas_min", "alt_site", "cas_per_intron", "term_cap")),
                   "pool is not flank, mbr and struct as registered"),
                  # the registered pool, exact: ``pool_one``'s feature tuple and its 11-value order, the
                  # structural candidates and the per-gene cap are all positional, so a changed value is silently
                  # read as a different pool (a shifted tuple, a dropped key), never as itself.
                  (doc["cap"] == 3, "cap is not 3"),
                  (p["flank"] == 5000, "pool.flank is not 5000"),
                  (mbr["n_samples"] == 1000, "mbr.n_samples is not 1000"),
                  (mbr["temperatures"] == [1.0, 2.0, 5.0], "mbr.temperatures is not [1.0, 2.0, 5.0]"),
                  (mbr["tau"] == 0.5, "mbr.tau is not 0.5"),
                  (struct["taus"] == [0.5, 0.2], "struct.taus is not [0.5, 0.2]"),
                  (struct["exon_min"] == 30 and struct["cas_min"] == 15 and struct["alt_site"] == 30
                   and struct["cas_per_intron"] == 12 and struct["term_cap"] == 12,
                   "struct counts are not 30/15/30/12/12"))
    except (KeyError, TypeError, IndexError, OverflowError) as e:
        return f"missing or malformed {e}"
    return next((why for ok, why in checks if not ok), None)


def run_item(ex, n_alt, twobit, item, vit, txs, changes, src, params, alt, T, part, cols):
    """The alt step of one item: windows, the pool on ``ex`` (the worker's spawn pool of ``n_alt`` processes), then
    score, ``emit`` and the ``<key>.alt.tsv`` part ``part`` (the alt bigBed's 21 columns, walk order).

    Tasks are built longest window first (ties by ordinal) and at most 2·n_alt are in flight; results are keyed by
    ordinal and the walk starts once all are in. ``item`` {chrom, start, end}; ``vit`` the item's Viterbi rows [(name,
    chrom, strand, CDS)] in ordinal order; ``txs`` the 5' UTR decoder's transcripts, one per Viterbi primary; ``changes`` the
    item's START records; ``src`` the item's f32 (la, lb, cc) [2, ·, L];
    ``params`` the declared Viterbi params; ``alt`` the alt.json document; ``cols`` (version, ckpt sha256, params
    sha256, alt.json sha256). Returns {n, counts, vmhwm_kb, max_window_bp}."""
    if list(alt["features"]) != list(FEATS):
        raise ValueError(f"alt.json features {alt['features']} are not {list(FEATS)}")
    by = {t["name"]: t for t in txs}
    if len(by) != len(txs) or set(by) != {v[0] for v in vit} or len(vit) != len(txs):
        raise ValueError(f"{item['chrom']}: the 5' UTR decoder's transcripts are not one per Viterbi primary")
    genome = _native.Genome(twobit)
    chrom, start = item["chrom"], item["start"]
    wins = windows(vit, start, item["end"], alt["pool"]["flank"])
    counts = {"+": Counter(), "-": Counter()}
    for v, w in zip(vit, wins):
        counts[v[2]]["windows" if w else "skip_window"] += 1

    def task(i):
        _name, _chrom, st, V = vit[i]
        ws, we = wins[i]
        t = {"ordinal": i, "strand": st, "ws": ws, "we": we, "V": V, "params": params, "cfg": alt["pool"],
             "seq": genome.fetch(chrom, ws, we).upper()}
        t["arrays"] = tuple(x[0 if st == "+" else 1, :, ws - start:we - start].astype(np.float16) for x in src)
        return t

    res, kb, pending = {}, [0], set()

    def take(done):
        for f in done:
            i, c, k = f.result()
            res[i] = c
            kb[0] = max(kb[0], k)

    for i in sorted((i for i, w in enumerate(wins) if w), key=lambda i: (wins[i][0] - wins[i][1], i)):
        while len(pending) >= 2 * n_alt:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            take(done)
        pending.add(ex.submit(pool_one, task(i)))
    take(wait(pending)[0])
    cands = defaultdict(list)
    for i in sorted(res):
        name, _chrom, st, _V = vit[i]
        if res[i] is None:
            counts[st]["skip_primary_infeasible"] += 1
            continue
        for key, source, vals in res[i]:
            cands[name].append((key, source, score(features(dict(zip(VALS, vals), source=source)), alt["model"])))
    rows, walk = emit(vit, by, {c["gene"]: c for c in changes}, cands, Str(genome), T, alt["cap"])
    for st in "+-":
        counts[st].update(walk[st])
    with open(part, "w") as fh:
        for name, rank, t, p, source, utr in rows:
            fh.write(_line(name, rank, t, p, source, utr, cols))
    return {"n": len(rows), "counts": {st: dict(c) for st, c in counts.items()}, "vmhwm_kb": kb[0],
            "max_window_bp": max((w[1] - w[0] for w in wins if w), default=0)}


def _line(name, rank, t, p, source, utr, cols):
    """One ``.alt.tsv`` row: BED12 (name ``<geneId>.<rank>``, score ⌊1000p + ½⌋), geneId, rank, sources, p, utr,
    release, ckptSha256, paramsSha256, altSha256."""
    blocks, (ts, te), src = t["blocks"], t["thick"], source.split(",")
    s0 = blocks[0][0]
    fields = [t["chrom"], s0, blocks[-1][1], f"{name}.{rank}", math.floor(1000.0 * p + 0.5), t["strand"], ts, te, 0,
              len(blocks), ",".join(str(e - s) for s, e in blocks) + ",", ",".join(str(s - s0) for s, e in blocks) + ",",
              name, rank, ",".join(k for k in ("ffbs", "struct") if any(x.startswith(k) for x in src)),
              format(p, ".17g"), utr, f"ancestra-{cols[0]}", *cols[1:]]
    return "\t".join(map(str, fields)) + "\n"


class Str:
    """``valid_isoform``'s fetch over ``_native.Genome``, whose ``fetch`` returns bytes."""

    def __init__(self, genome):
        self.genome = genome

    def fetch(self, chrom, start, end):
        return self.genome.fetch(chrom, start, end).decode("ascii")


def _vmhwm_kb():
    """This process's VmHWM (as ``ancestra.io.items.peak_rss_kb``): ru_maxrss survives fork+exec."""
    with open("/proc/self/status") as fh:
        return next(int(line.split()[1]) for line in fh if line.startswith("VmHWM:"))
