"""Torch-free item parsing for the CLI, which checks flags, sealed files, items, names and RAM before torch is ever
imported: this module imports nothing at all. ``range_error`` is the strict spec syntax: a malformed spec is refused
with exit 2, never a traceback, and never as an out-of-range one (exit 3), which is ``parse_specs``'s lenient rule."""


def parse_specs(specs, sizes):
    """``chrom`` or ``chrom:start-end`` -> ([(name, chrom, start, end)], [(spec, reason)]). A name absent from the 2bit
    and an empty or out-of-range range are returned as refused, never skipped or clamped silently."""
    out, bad = [], []
    for spec in specs:
        if spec in sizes:
            chrom, rs, re_, name = spec, 0, sizes[spec], spec
        elif ":" in spec:
            chrom, rng = spec.rsplit(":", 1)
            start, dash, end = rng.partition("-")
            if not chrom or not dash or not _uint(start) or not _uint(end):
                bad.append((spec, f"malformed range '{rng}': expected start-end with two integers"))
                continue
            rs, re_ = int(start), int(end)
            name = f"{chrom}_{rs}_{re_}"
        else:
            chrom, rs, re_, name = spec, 0, None, spec
        size = sizes.get(chrom)
        re_ = size if re_ is None else re_
        if size is None:
            bad.append((spec, "absent from the 2bit"))
        elif not 0 <= rs < re_ <= size:
            bad.append((spec, f"empty or outside [0, {size})"))
        else:
            out.append((name, chrom, rs, re_))
    return out, bad


def lpt_shards(items, n):
    """Largest first to the least-loaded worker; each shard in largest-first order. ``items``: (name, chrom, start,
    end)."""
    order = sorted(items, key=lambda t: (-(t[3] - t[2]), t[0]))
    load, shards = [0] * n, [[] for _ in range(n)]
    for it in order:
        k = min(range(n), key=lambda i: (load[i], i))
        shards[k].append(it)
        load[k] += it[3] - it[2]
    return shards


def range_error(spec, sizes=None):
    """The release CLI's strict spec syntax (``chrom`` or a 0-based half-open ``chrom:start-end``): the reason this spec
    is refused with exit 2, or None. ``c1:0-nope``, ``c1:5-2`` and ``c1:-1-3`` are malformed specs (refused before any
    work), so no spec reaches ``parse_specs`` as syntax it would refuse as out of range; a chrom's own safety is the
    name preflight's, not this one's."""
    if sizes is not None and spec in sizes:
        return None
    if ":" not in spec:
        return None
    chrom, rng = spec.rsplit(":", 1)
    start, dash, end = rng.partition("-")
    if not chrom:
        return "no sequence name before ':'"
    if not dash or not _uint(start) or not _uint(end):
        return f"malformed range '{rng}': expected start-end with two integers"
    if not int(start) < int(end):
        return f"range {start}-{end} is empty or reversed: expected 0 <= start < end"
    return None


def _uint(tok):
    """A non-empty run of ASCII digits: ``int`` alone also takes unicode digits, ``_``, ``+``, ``-`` and spaces."""
    return bool(tok) and all("0" <= c <= "9" for c in tok)
