"""Transcript geometry of the post-decoders: BED12 rows in, transcript dicts out. Stdlib only.

``valid_isoform`` checks one isoform's coding structure; ``intron_ok`` is its splice test and ``vrow`` the --alt
track's adapter from ``make_tx``'s dict. Coordinates are 0-based half-open, genome order on both strands.
"""

STOPS = ("TAG", "TAA", "TGA")
SPLICE = {("GT", "AG"), ("GC", "AG"), ("AT", "AC")}
_RC = str.maketrans("ACGT", "TGCA")


def merge(blocks):
    """Sorted union of (start, end) intervals (touching intervals are merged)."""
    out = []
    for s, e in sorted(blocks):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def clip_cds(exons, ts, te):
    if te <= ts:
        return ()
    return tuple((max(s, ts), min(e, te)) for s, e in exons if max(s, ts) < min(e, te))


def gene_symbol(name):
    """`hg38.ENST00000541576.5#THBS3#6` -> `THBS3` (stand-in for query gene id)."""
    parts = name.split("#")
    return parts[1] if len(parts) >= 3 else name


def read_bed12(path):
    """BED12 rows as dicts; skips track/#/browser lines."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("track", "#", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 12:
                continue
            blocks = []
            sizes = f[10].rstrip(",").split(",")
            starts = f[11].rstrip(",").split(",")
            chrom_start = int(f[1])
            for size, off in zip(sizes, starts):
                if not size:
                    continue
                s = chrom_start + int(off)
                blocks.append((s, s + int(size)))
            rows.append({
                "name": f[3], "chrom": f[0], "strand": f[5],
                "tstart": int(f[6]), "tend": int(f[7]), "blocks": blocks,
            })
    return rows


def gene_of(name):
    return gene_symbol(name)


def _side_intervals(blocks, thick, strand):
    """UTR exon pieces on each side. '+' 5′ is below thickStart; '−' 5′ is above thickEnd."""
    ts, te = thick
    if strand == "+":
        five = tuple((s, min(e, ts)) for s, e in blocks if s < ts and min(e, ts) > s)
        three = tuple((max(s, te), e) for s, e in blocks if e > te and e > max(s, te))
        five_at, three_at = blocks[0][0], blocks[-1][1]
    else:
        five = tuple((max(s, te), e) for s, e in blocks if e > te and e > max(s, te))
        three = tuple((s, min(e, ts)) for s, e in blocks if s < ts and min(e, ts) > s)
        five_at, three_at = blocks[-1][1], blocks[0][0]
    return five, three, five_at, three_at


def _utr_side(intron, thick, strand):
    """'5', '3', or None when the intron is not entirely outside the CDS."""
    a, b = intron
    ts, te = thick
    if strand == "+":
        if b <= ts:
            return "5"
        if a >= te:
            return "3"
    else:
        if a >= te:
            return "5"
        if b <= ts:
            return "3"
    return None


def make_tx(chrom, strand, name, blocks, thick, gene=None):
    """One transcript. CDS blocks are exons clipped to thick, empty pieces dropped."""
    blocks = tuple(sorted((int(s), int(e)) for s, e in blocks if e > s))
    if not blocks or strand not in ("+", "-"):
        return None
    ts, te = int(thick[0]), int(thick[1])
    if te <= ts:
        return None
    cds = tuple((max(s, ts), min(e, te)) for s, e in blocks if min(e, te) > max(s, ts))
    if not cds:
        return None
    introns = tuple((blocks[i][1], blocks[i + 1][0]) for i in range(len(blocks) - 1))
    five, three, five_at, three_at = _side_intervals(blocks, (ts, te), strand)
    utr_i = {"5": [], "3": []}
    for intron in introns:
        side = _utr_side(intron, (ts, te), strand)
        if side:
            utr_i[side].append(intron)
    return {"chrom": chrom, "strand": strand, "name": name, "gene": gene if gene is not None else gene_of(name),
            "blocks": blocks, "thick": (ts, te), "cds": cds, "introns": introns,
            "five_iv": five, "three_iv": three, "five": five_at, "three": three_at,
            "utr_introns": {"5": tuple(utr_i["5"]), "3": tuple(utr_i["3"])},
            "single_exon": len(blocks) == 1}


def _paths(paths):
    return [paths] if isinstance(paths, str) else list(paths)


def load_bed(paths, chroms=None):
    """BED12 paths -> transcripts with a CDS. `chroms` None keeps every sequence."""
    chroms = set(chroms) if chroms is not None else None
    out = []
    for path in _paths(paths):
        for row in read_bed12(path):
            if chroms is not None and row["chrom"] not in chroms:
                continue
            tx = make_tx(row["chrom"], row["strand"], row["name"], row["blocks"], (row["tstart"], row["tend"]))
            if tx is not None:
                out.append(tx)
    return out


def _rc(seq):
    return seq.translate(_RC)[::-1]


def _fetch(twobit, chrom, start, end):
    if end < start:
        raise ValueError(f"{chrom} [{start},{end}): negative fetch")
    if start == end:
        return ""
    seq = twobit.fetch(chrom, start, end)
    if len(seq) != end - start:
        raise ValueError(f"{chrom} [{start},{end}): 2bit returned {len(seq)} bases")
    return seq.upper()


def intron_ok(twobit, chrom, strand, a1, b0, min_intron):
    """The intron [a1, b0) has >= ``min_intron`` bases and GT-AG, GC-AG or AT-AC in transcript orientation."""
    if b0 - a1 < min_intron:
        return False
    if strand == "+":
        donor, acceptor = _fetch(twobit, chrom, a1, a1 + 2), _fetch(twobit, chrom, b0 - 2, b0)
    else:
        donor = _rc(_fetch(twobit, chrom, b0 - 2, b0))
        acceptor = _rc(_fetch(twobit, chrom, a1, a1 + 2))
    return (donor, acceptor) in SPLICE


def valid_isoform(row, twobit, min_intron=50):
    """ATG start, canonical stop, CDS length % 3 == 0, no internal stop, canonical splice, no intron < ``min_intron``.

    Splice pairs are GT-AG, GC-AG, AT-AC in transcript orientation, on every intron between exon blocks
    (UTR introns included). A non-ACGT CDS base fails closed. Returns ``(ok, cds_len)``. ``row`` is a dict with
    ``chrom``, ``strand``, ``blocks``, ``thick_start`` and ``thick_end`` (``vrow``'s).

    ``min_intron`` is the CDS decoder's 50 by default; the alternative-isoform track passes 30 explicitly.
    """
    d, cds = row, clip_cds(tuple(row["blocks"]), row["thick_start"], row["thick_end"])
    cds_len = sum(e - s for s, e in cds)
    exons = list(d["blocks"])
    for (a0, a1), (b0, b1) in zip(exons, exons[1:]):
        if b0 - a1 < min_intron:
            return False, cds_len
    if cds_len < 3 or cds_len % 3:
        return False, cds_len
    # Genome order, then reverse-complement the whole CDS on the − strand (RC of each block concatenated
    # in transcript order is the same string; RC of the transcript-order concatenation is not).
    seq = "".join(_fetch(twobit, d["chrom"], s, e) for s, e in cds)
    if d["strand"] == "-":
        seq = _rc(seq)
    if len(seq) != cds_len or not set(seq) <= set("ACGT"):
        return False, cds_len
    if seq[:3] != "ATG" or seq[-3:] not in STOPS:
        return False, cds_len
    for i in range(0, cds_len - 3, 3):
        if seq[i:i + 3] in STOPS:
            return False, cds_len
    for (a0, a1), (b0, b1) in zip(exons, exons[1:]):
        if not intron_ok(twobit, d["chrom"], d["strand"], a1, b0, min_intron):
            return False, cds_len
    return True, cds_len


def vrow(tx):
    """make_tx's dict (``thick`` = (ts, te)) as ``valid_isoform``'s row."""
    return {"chrom": tx["chrom"], "strand": tx["strand"], "blocks": tx["blocks"], "thick_start": tx["thick"][0],
            "thick_end": tx["thick"][1]}
