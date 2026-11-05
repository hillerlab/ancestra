"""Post-decode START extension.

For each predicted chain, walk upstream in frame from the current start codon
along the first CDS block's contiguous genomic sequence (thickStart side on +,
thickEnd side on −) until an in-frame stop, W bp, a sequence/emission end, or another
predicted gene on either strand. The new start is the most upstream in-frame ATG
with START-head p >= tau; the first block is extended if the new start leaves it.
"""
import os

import numpy as np

from ancestra import _native

STOPS = {b"TAA", b"TAG", b"TGA"}
COMP = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")


def read_bed(path):
    rows = []
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 12:
            continue
        s0 = int(f[1])
        sizes = [int(x) for x in f[10].rstrip(",").split(",") if x]
        starts = [int(x) for x in f[11].rstrip(",").split(",") if x]
        rows.append({
            "f": f,
            "blocks": [(s0 + a, s0 + a + b) for a, b in zip(starts, sizes)],
            "ts": int(f[6]),
            "te": int(f[7]),
        })
    return rows


def write_row(r):
    f = list(r["f"])
    bl = r["blocks"]
    s0, e0 = bl[0][0], bl[-1][1]
    f[1], f[2], f[6], f[7] = str(s0), str(e0), str(r["ts"]), str(r["te"])
    f[9] = str(len(bl))
    f[10] = ",".join(str(e - s) for s, e in bl) + ","
    f[11] = ",".join(str(s - s0) for s, e in bl) + ","
    return "\t".join(f)


def fences(rows):
    """Per row, plus-walk lower bound and minus-walk upper bound from every other gene on the chrom (either strand)."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["f"][0], []).append(i)
    lo = [0] * len(rows)
    hi = [1 << 62] * len(rows)
    for idx in by.values():
        spans = [(int(rows[i]["f"][1]), int(rows[i]["f"][2]), i) for i in idx]
        for i in idx:
            ts, te = rows[i]["ts"], rows[i]["te"]
            plo, phi = 0, 1 << 62
            for s, e, j in spans:
                if j == i:
                    continue
                if e <= ts:
                    plo = max(plo, e)
                elif s < ts < e:
                    plo = max(plo, ts)
                if s >= te:
                    phi = min(phi, s)
                elif s < te < e:
                    phi = min(phi, te)
            lo[i], hi[i] = plo, phi
    return lo, hi


def start_block_i(blocks, strand, ts, te):
    if strand == "+":
        for i, (a, b) in enumerate(blocks):
            if a <= ts < b:
                return i
        return 0
    for i in range(len(blocks) - 1, -1, -1):
        a, b = blocks[i]
        if a < te <= b:
            return i
    return len(blocks) - 1


def candidates(genome, cache, row, fen_lo, fen_hi, window):
    """[(distance, p, new_cds_boundary)] in-frame ATGs upstream of the current start, nearest first."""
    chrom, strand = row["f"][0], row["f"][5]
    if chrom not in cache or chrom not in genome.chrom_sizes() or row["ts"] >= row["te"]:
        return []
    g, gs, ge = cache[chrom]
    size = int(genome.chrom_sizes()[chrom])
    si = 0 if strand == "+" else 1
    out = []
    if strand == "+":
        s0 = row["ts"]
        lo = max(gs, fen_lo, s0 - window)
        if s0 > ge or lo >= s0:
            return []
        seq = bytes(genome.fetch(chrom, lo, s0)).upper()
        cs = np.asarray(g["c"][si, 2, lo - gs:s0 - gs], dtype=np.float32)
        k = 1
        while s0 - 3 * k >= lo:
            a = s0 - 3 * k
            cod = seq[a - lo:a - lo + 3]
            if len(cod) < 3 or cod in STOPS:
                break
            if cod == b"ATG":
                out.append((3 * k, float(cs[a - lo]), a))
            k += 1
    else:
        e0 = row["te"]
        hi = min(ge, size, fen_hi, e0 + window)
        if e0 < gs or hi - e0 < 3:
            return []
        seq = bytes(genome.fetch(chrom, e0, hi)).upper()
        cs = np.asarray(g["c"][si, 2, e0 - gs:hi - gs], dtype=np.float32)
        k = 1
        while e0 + 3 * k <= hi:
            b = e0 + 3 * k
            trip = seq[b - 3 - e0:b - e0]
            if len(trip) < 3:
                break
            cod = trip.translate(COMP)[::-1]
            if cod in STOPS:
                break
            if cod == b"ATG":
                out.append((3 * k, float(cs[b - 1 - e0]), b))
            k += 1
    return out


def choose(cands, tau, window):
    ok = [c for c in cands if c[0] <= window and c[1] >= tau]
    return max(ok, key=lambda c: c[0]) if ok else None


def apply_one(row, choice):
    r = dict(row, blocks=list(row["blocks"]))
    r["blocks"] = [tuple(b) for b in r["blocks"]]
    strand = r["f"][5]
    i = start_block_i(r["blocks"], strand, r["ts"], r["te"])
    bl = list(r["blocks"])
    if strand == "+":
        new = choice[2]
        a, b = bl[i]
        if new < a:
            bl[i] = (new, b)
        r["ts"] = new
    else:
        new = choice[2]
        a, b = bl[i]
        if new > b:
            bl[i] = (a, new)
        r["te"] = new
    r["blocks"] = bl
    return r


def scan(bed, twobit, window, groups):
    """``groups``: the item's in-memory emissions {chrom: (start, utr7, c)} (the 5' UTR decoder's ``groups``), f16."""
    genome = _native.Genome(twobit)
    bad = sorted(k for k, (_s, _u, c) in groups.items() if c.dtype != np.float16)
    if bad:
        raise ValueError("in-memory emit for %s is not float16, the emit cache's dtype" % ", ".join(bad))
    cache = {k: ({"c": c}, s, s + c.shape[-1]) for k, (s, _u, c) in groups.items()}
    rows = read_bed(bed)
    fen_lo, fen_hi = fences(rows)
    cands = [candidates(genome, cache, r, fen_lo[i], fen_hi[i], window) for i, r in enumerate(rows)]
    return rows, cands


def changes_of(rows, cands, tau, window):
    out = []
    new_rows = []
    for r, cs in zip(rows, cands):
        c = choose(cs, tau, window)
        if c is None:
            new_rows.append(r)
            continue
        nr = apply_one(r, c)
        new_rows.append(nr)
        strand = r["f"][5]
        out.append({
            "gene": r["f"][3],
            "old_start": r["ts"] if strand == "+" else r["te"],
            "new_start": nr["ts"] if strand == "+" else nr["te"],
            "p": c[1],
            "distance": c[0],
        })
    return new_rows, out


def write_bed(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(write_row(r) + "\n")
