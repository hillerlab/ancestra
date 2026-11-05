"""The L2 START chooser: a frozen gradient-boosted classifier (``data/l2_model.json``) picks each chain's start
codon among the extended start, the Viterbi start and the in-frame ATGs around them, from emission and sequence
features."""
import bisect
from functools import lru_cache
import json
import math
import os

import numpy as np

from ancestra import DATA_DIR
from ancestra.decode import start_extend as se


FEATURES = ["f_d", "f_absd", "f_d_sx", "f_p", "f_logit_p", "f_p_s0", "f_p_sx", "f_p_max_other",
            "f_rank_p", "f_n_atg_between", "f_seg_len", "f_cds_mean", "f_cds_min", "f_nc_mean",
            "f_phase_mean", "f_acc_max_seg", "f_don_max_seg", "f_up_nc_mean", "f_up_utr5_mean",
            "f_kozak_m3", "f_kozak_p4", "f_bound_dist", "f_cds_len", "f_n_blocks", "f_first_block_len",
            "f_extended"]


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def read_bed(path, *, coding=False):
    rows, items = [], []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                items.append((None, line))
                continue
            f = line.rstrip("\r\n").split("\t")
            _require(len(f) >= 12, "%s:%d: expected BED12" % (path, line_no))
            try:
                start, end, ts, te, n = int(f[1]), int(f[2]), int(f[6]), int(f[7]), int(f[9])
                sizes = [int(x) for x in f[10].rstrip(",").split(",") if x]
                starts = [int(x) for x in f[11].rstrip(",").split(",") if x]
            except ValueError as e:
                raise ValueError("%s:%d: invalid BED12 integer: %s" % (path, line_no, e)) from e
            _require(f[5] in ("+", "-"), "%s:%d: invalid strand %r" % (path, line_no, f[5]))
            _require(start < end and n > 0 and len(sizes) == len(starts) == n,
                     "%s:%d: invalid BED12 block count or bounds" % (path, line_no))
            blocks = [(start + a, start + a + b) for a, b in zip(starts, sizes)]
            _require(all(a >= start and a < b <= end for a, b in blocks),
                     "%s:%d: invalid BED12 block" % (path, line_no))
            _require(all(blocks[i - 1][1] <= blocks[i][0] for i in range(1, n)),
                     "%s:%d: blocks are not sorted and disjoint" % (path, line_no))
            if coding:
                _require(start <= ts < te <= end, "%s:%d: invalid CDS bounds" % (path, line_no))
            row = {"f": f, "blocks": blocks, "ts": ts, "te": te, "line_no": line_no}
            rows.append(row)
            items.append((row, line.endswith(("\n", "\r"))))
    names = [r["f"][3] for r in rows]
    _require(len(names) == len(set(names)), "%s: duplicate BED chain names" % path)
    return rows, items


def _s0(row):
    return row["ts"] if row["f"][5] == "+" else row["te"]


def _start_i(row):
    return se.start_block_i(row["blocks"], row["f"][5], row["ts"], row["te"])


def _cds_chain(row, chrom=None):
    if row["ts"] >= row["te"]:
        return None
    blocks = sorted({(max(a, row["ts"]), min(b, row["te"])) for a, b in row["blocks"]
                     if max(a, row["ts"]) < min(b, row["te"])})
    if not blocks:
        return None
    return (chrom if chrom is not None else row["f"][0], row["f"][5], tuple(blocks))


def _validate_cache(cache):
    for chrom, (g, start, end) in cache.items():
        _require(start < end, "emit %s: invalid start/end" % chrom)
        for key, n in (("c", 4), ("log_a", 3), ("log_b", 3), ("utr7", 7)):
            _require(key in g, "emit %s: missing %s" % (chrom, key))
            a = g[key]
            _require(tuple(a.shape) == (2, n, end - start),
                     "emit %s: %s shape %r, expected (2,%d,%d)" % (chrom, key, a.shape, n, end - start))
            _require(np.dtype(a.dtype) == np.dtype(np.float16), "emit %s: %s is not float16" % (chrom, key))


def _array(g, key, plane, channel, start, end, origin):
    return np.asarray(g[key][plane, channel, start - origin:end - origin], dtype=np.float32)


def _downstream_candidates(genome, row):
    chrom, strand = row["f"][0], row["f"][5]
    blocks, s0 = row["blocks"], _s0(row)
    a, b = blocks[_start_i(row)]
    out = []
    if strand == "+":
        seq = bytes(genome.fetch(chrom, s0, b)).upper()
        for d in range(3, b - s0 - 2, 3):
            cand = s0 + d
            if cand + 3 <= b and seq[d:d + 3] == b"ATG":
                out.append(cand)
    else:
        seq = bytes(genome.fetch(chrom, a, s0)).upper()
        for d in range(3, s0 - a - 2, 3):
            cand = s0 - d
            lo = cand - 3 - a
            if cand - 3 >= a and seq[lo:lo + 3].translate(se.COMP)[::-1] == b"ATG":
                out.append(cand)
    return out


def _walk_boundary(genome, row, cache_entry, fen_lo, fen_hi, window):
    """Furthest in-frame boundary the registered upstream walk can reach."""
    chrom, strand = row["f"][0], row["f"][5]
    _g, gs, ge = cache_entry
    if strand == "+":
        s0 = row["ts"]
        lo = max(gs, fen_lo, s0 - window)
        if s0 > ge or lo >= s0:
            return s0
        seq = bytes(genome.fetch(chrom, lo, s0)).upper()
        boundary = s0 - 3 * ((s0 - lo) // 3)
        for k in range(1, (s0 - lo) // 3 + 1):
            a = s0 - 3 * k
            if seq[a - lo:a - lo + 3] in se.STOPS:
                return a + 3
        return boundary
    e0 = row["te"]
    hi = min(ge, int(genome.chrom_sizes()[chrom]), fen_hi, e0 + window)
    if e0 < gs or hi - e0 < 3:
        return e0
    seq = bytes(genome.fetch(chrom, e0, hi)).upper()
    boundary = e0 + 3 * ((hi - e0) // 3)
    for k in range(1, (hi - e0) // 3 + 1):
        b = e0 + 3 * k
        if seq[b - 3 - e0:b - e0].translate(se.COMP)[::-1] in se.STOPS:
            return b - 3
    return boundary


def _base(genome, chrom, pos, size, strand):
    if pos < 0 or pos >= size:
        return b""
    b = bytes(genome.fetch(chrom, pos, pos + 1)).upper()
    return b.translate(se.COMP) if strand == "-" else b


def _signed_distance(strand, anchor, cand):
    return anchor - cand if strand == "+" else cand - anchor


def _prefix(values):
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)))


def _anchored(values, anchor, op):
    accum = np.minimum.accumulate if op == "min" else np.maximum.accumulate
    left = accum(values[:anchor][::-1])[::-1] if anchor else np.empty(0, dtype=values.dtype)
    right = accum(values[anchor:]) if anchor < len(values) else np.empty(0, dtype=values.dtype)
    return left, right


def make_table_records(row_name, vit_rows, sx_rows, genome, cache, wmax=3000):
    _require(isinstance(wmax, int) and not isinstance(wmax, bool) and wmax > 0,
             "wmax must be a positive integer")
    _require(len(vit_rows) == len(sx_rows), "VIT and SX row counts differ (%d != %d)" % (len(vit_rows), len(sx_rows)))
    _require([r["f"][3] for r in vit_rows] == [r["f"][3] for r in sx_rows], "VIT and SX row order differs")
    vby = {r["f"][3]: r for r in vit_rows}
    sby = {r["f"][3]: r for r in sx_rows}
    _require(vby.keys() == sby.keys(), "VIT and SX chain names differ")
    sizes = genome.chrom_sizes()
    _validate_cache(cache)
    for chrom in sorted({r["f"][0] for r in vit_rows}):
        _require(chrom in cache, "VIT chrom %s is absent from emit" % chrom)
        _require(chrom in sizes, "VIT chrom %s is absent from 2bit" % chrom)
    f_lo, f_hi = se.fences(vit_rows)
    result = []
    for i, row in enumerate(vit_rows):
        chrom, strand = row["f"][0], row["f"][5]
        _require(chrom in sizes and row["f"][0] in cache, "VIT chrom %s is unavailable" % chrom)
        size = int(sizes[chrom])
        _require(0 <= int(row["f"][1]) < int(row["f"][2]) <= size,
                 "%s: VIT bounds are outside the 2bit" % row["f"][3])
        _require(cache[chrom][1] <= (row["ts"] if strand == "+" else row["te"] - 1) < cache[chrom][2],
                 "%s: VIT START is outside its emit" % row["f"][3])
        sx = sby[row["f"][3]]
        _validate_pair(row, sx)
        s0, sx0 = _s0(row), _s0(sx)
        up = se.candidates(genome, cache, row, f_lo[i], f_hi[i], wmax)
        boundary = _walk_boundary(genome, row, cache[chrom], f_lo[i], f_hi[i], wmax)
        raw = [(s0, None)] + [(p[2], p) for p in up] + [(c, None) for c in _downstream_candidates(genome, row)]
        _require(len({c for c, _ in raw}) == len(raw), "%s: duplicate candidate" % row["f"][3])
        candidates = [c for c, _ in raw]
        _require(sx0 in candidates, "%s: SX start %d is not a candidate" % (row["f"][3], sx0))
        chosen = se.choose(up, 0.99, 300)
        _require((chosen[2] if chosen is not None else s0) == sx0,
                 "%s: SX start differs from start_extend.choose(0.99, 300)" % row["f"][3])
        g, gs, ge = cache[chrom]
        plane = 0 if strand == "+" else 1
        p_coords = [c if strand == "+" else c - 1 for c in candidates]
        p_coords += [s0 if strand == "+" else s0 - 1, sx0 if strand == "+" else sx0 - 1]
        _require(all(gs <= p < ge for p in p_coords), "%s: candidate START is outside its emit" % row["f"][3])
        p_lo, p_hi = min(p_coords), max(p_coords) + 1
        p_values = _array(g, "c", plane, 2, p_lo, p_hi, gs)
        p_by = {c: float(p_values[(c if strand == "+" else c - 1) - p_lo]) for c in candidates}
        _require(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in p_by.values()),
                 "%s: invalid START probability" % row["f"][3])
        p_s0, p_sx = p_by[s0], p_by[sx0]
        sorted_p = sorted(p_by.values())
        first_i = max(range(len(candidates)), key=lambda j: p_by[candidates[j]])
        second_max = max((p_by[c] for j, c in enumerate(candidates) if j != first_i), default=0.0)
        max_p = p_by[candidates[first_i]]
        lo, hi = min([s0] + candidates), max([s0] + candidates)
        arr_lo, arr_hi = max(gs, lo), min(ge, hi)
        # Candidate segments are anchored at s0; one contiguous read serves every candidate.
        a_cds = _array(g, "log_a", plane, 1, arr_lo, arr_hi, gs)
        a_nc = _array(g, "log_a", plane, 0, arr_lo, arr_hi, gs)
        b_phase = _array(g, "log_b", plane, 0, arr_lo, arr_hi, gs)
        b1 = _array(g, "log_b", plane, 1, arr_lo, arr_hi, gs)
        b2 = _array(g, "log_b", plane, 2, arr_lo, arr_hi, gs)
        donor = _array(g, "c", plane, 0, arr_lo, arr_hi, gs)
        acceptor = _array(g, "c", plane, 1, arr_lo, arr_hi, gs)
        _require(all(np.isfinite(a).all() for a in (a_cds, a_nc, b_phase, b1, b2, donor, acceptor)),
                 "%s: non-finite emission feature" % row["f"][3])
        local_anchor = s0 - arr_lo
        cds_pre, nc_pre = _prefix(a_cds), _prefix(a_nc)
        phase_pos = np.arange(arr_lo, arr_hi, dtype=np.int64)
        phases = ((phase_pos - s0) % 3) if strand == "+" else ((s0 - 1 - phase_pos) % 3)
        phase_values = np.choose(phases, (b_phase, b1, b2)) if len(phases) else np.empty(0, dtype=np.float32)
        phase_pre = _prefix(phase_values)
        cds_min_left, cds_min_right = _anchored(a_cds, local_anchor, "min")
        acc_left, acc_right = _anchored(acceptor, local_anchor, "max")
        don_left, don_right = _anchored(donor, local_anchor, "max")
        clipped = _cds_chain(row)
        cds_len, n_blocks = sum(b - a for a, b in clipped[2]), len(clipped[2])
        first_block = row["blocks"][_start_i(row)]
        first_len = max(0, min(first_block[1], row["te"]) - max(first_block[0], row["ts"]))
        sorted_coords = sorted(candidates)
        for cand in candidates:
            d = _signed_distance(strand, s0, cand)
            lo_seg, hi_seg = min(cand, s0), max(cand, s0)
            if lo_seg == hi_seg:
                seg_len = cds_mean = cds_min = nc_mean = phase_mean = acc_max = don_max = 0.0
            else:
                left, right = lo_seg - arr_lo, hi_seg - arr_lo
                seg_len = hi_seg - lo_seg
                cds_mean = (cds_pre[right] - cds_pre[left]) / seg_len
                nc_mean = (nc_pre[right] - nc_pre[left]) / seg_len
                phase_mean = (phase_pre[right] - phase_pre[left]) / seg_len
                if cand < s0:
                    cds_min, acc_max, don_max = cds_min_left[left], acc_left[left], don_left[left]
                else:
                    cds_min, acc_max, don_max = cds_min_right[right - local_anchor - 1], \
                        acc_right[right - local_anchor - 1], don_right[right - local_anchor - 1]
            ctx_lo = max(gs, cand - 60) if strand == "+" else max(gs, cand)
            ctx_hi = min(ge, cand) if strand == "+" else min(ge, cand + 60)
            if ctx_lo < ctx_hi:
                up_nc = _array(g, "log_a", plane, 0, ctx_lo, ctx_hi, gs)
                up_utr = _array(g, "utr7", plane, 1, ctx_lo, ctx_hi, gs)
                _require(np.isfinite(up_nc).all() and np.isfinite(up_utr).all(),
                         "%s: non-finite upstream emission feature" % row["f"][3])
                up_nc_mean, up_utr_mean = float(up_nc.mean()), float(up_utr.mean())
            else:
                up_nc_mean = up_utr_mean = 0.0
            if strand == "+":
                m3, p4 = _base(genome, chrom, cand - 3, size, strand), _base(genome, chrom, cand + 3, size, strand)
            else:
                m3, p4 = _base(genome, chrom, cand + 2, size, strand), _base(genome, chrom, cand - 4, size, strand)
            between = (0 if cand == s0 else bisect.bisect_left(sorted_coords, max(cand, s0)) -
                       bisect.bisect_right(sorted_coords, min(cand, s0)))
            rank = 1 + len(sorted_p) - bisect.bisect_right(sorted_p, p_by[cand])
            phase_p = min(max(p_by[cand], 1e-4), 1.0 - 1e-4)
            record = dict(row=row_name, chain=row["f"][3], chrom=chrom, strand=strand, s0=s0, sx=sx0,
                          cand=cand, d_nt=d, is_s0=int(cand == s0), is_sx=int(cand == sx0), label="",
                          f_d=d, f_absd=abs(d), f_d_sx=_signed_distance(strand, sx0, cand), f_p=p_by[cand],
                          f_logit_p=math.log(phase_p / (1.0 - phase_p)), f_p_s0=p_s0, f_p_sx=p_sx,
                          f_p_max_other=(second_max if cand == candidates[first_i] else max_p),
                          f_rank_p=rank, f_n_atg_between=between, f_seg_len=seg_len, f_cds_mean=cds_mean,
                          f_cds_min=cds_min, f_nc_mean=nc_mean, f_phase_mean=phase_mean, f_acc_max_seg=acc_max,
                          f_don_max_seg=don_max, f_up_nc_mean=up_nc_mean, f_up_utr5_mean=up_utr_mean,
                          f_kozak_m3=int(m3 in (b"A", b"G")), f_kozak_p4=int(p4 == b"G"),
                          f_bound_dist=(cand - boundary if strand == "+" else boundary - cand), f_cds_len=cds_len,
                          f_n_blocks=n_blocks, f_first_block_len=first_len, f_extended=int(sx0 != s0))
            result.append(record)
    return result


def _validate_pair(vit, sx):
    name = vit["f"][3]
    allowed = {1, 2, 6, 7, 10, 11}
    _require(len(vit["f"]) == len(sx["f"]) and
             all(vit["f"][i] == sx["f"][i] for i in range(len(vit["f"])) if i not in allowed),
             "%s: VIT and SX fields differ outside thick bounds/start block" % name)
    vi, si = _start_i(vit), _start_i(sx)
    _require(vi == si, "%s: VIT and SX start block index differs" % name)
    for row, idx in ((vit, vi), (sx, si)):
        a, b = row["blocks"][idx]
        _require(a <= row["ts"] < b if row["f"][5] == "+" else a < row["te"] <= b,
                 "%s: thick bounds are outside the start block" % name)
    _require(len(vit["blocks"]) == len(sx["blocks"]), "%s: VIT and SX block counts differ" % name)
    for j, (vb, sb) in enumerate(zip(vit["blocks"], sx["blocks"])):
        if j != vi:
            _require(vb == sb, "%s: non-start block %d changed" % (name, j))
    if vit["f"][5] == "+":
        vb, sb = vit["blocks"][vi], sx["blocks"][si]
        expected = (sx["ts"], vb[1]) if sx["ts"] < vb[0] else vb
        _require(sb == expected and sx["te"] == vit["te"], "%s: SX start block is not a START-only move" % name)
    else:
        vb, sb = vit["blocks"][vi], sx["blocks"][si]
        expected = (vb[0], sx["te"]) if sx["te"] > vb[1] else vb
        _require(sb == expected and sx["ts"] == vit["ts"], "%s: SX start block is not a START-only move" % name)
    _require((int(vit["f"][1]), int(vit["f"][2])) == (min(a for a, _ in vit["blocks"]), max(b for _, b in vit["blocks"]))
             and (int(sx["f"][1]), int(sx["f"][2])) == (min(a for a, _ in sx["blocks"]), max(b for _, b in sx["blocks"])),
             "%s: chrom bounds do not match blocks" % name)


def apply_choices(vit_path, records, choices, out_path):
    rows, items = read_bed(vit_path, coding=True)
    candidates = {}
    for r in records:
        c = candidates.setdefault(r["chain"], {"cand": set(), "s0": set(), "sx": set(), "rows": set()})
        c["cand"].add(int(r["cand"]))
        c["s0"].add(int(r["s0"]))
        c["sx"].add(int(r["sx"]))
        c["rows"].add(r["row"])
    by_name = {r["f"][3]: r for r in rows}
    _require(by_name.keys() == choices.keys(), "%s: choices must cover exactly the VIT chains" % vit_path)
    _require(by_name.keys() == candidates.keys(), "%s: candidate table chains differ from VIT" % vit_path)
    for name, row in by_name.items():
        c = candidates[name]
        _require(len(c["rows"]) == len(c["s0"]) == len(c["sx"]) == 1 and c["s0"] == {_s0(row)},
                 "%s: candidate table does not match VIT start" % name)
        choice = choices[name]
        cand = int(choice["cand"])
        _require(math.isfinite(float(choice["score"])), "%s: non-finite choice score" % name)
        _require(cand in c["cand"], "%s: choice %d is outside its candidate set" % (name, cand))
    output = {}
    for name, row in by_name.items():
        choice = choices[name]
        cand = int(choice["cand"])
        output[name] = se.write_row(se.apply_one(row, (abs(cand - _s0(row)), float(choice["score"]), cand)))
    changes = []
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        for item in items:
            row, ending = item
            if row is None:
                fh.write(ending)
            else:
                fh.write(output[row["f"][3]])
                if ending:
                    fh.write("\n")
                name = row["f"][3]
                cand = int(choices[name]["cand"])
                if cand != _s0(row):
                    changes.append({"gene": name, "old_start": _s0(row), "new_start": cand,
                                    "p": float(choices[name]["score"]), "distance": abs(cand - _s0(row))})
    return changes


@lru_cache(maxsize=1)
def read_model(path=None):
    path = path or os.path.join(DATA_DIR, "l2_model.json")
    with open(path, encoding="utf-8") as fh:
        model = json.load(fh)
    _require(model.get("features") == FEATURES and len(model.get("trees", [])) == 300,
             "embedded L2 model feature or tree count mismatch")
    return model


def raw_predict(model, x):
    x = np.asarray(x, dtype=np.float64, order="C")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    _require(x.ndim == 2 and x.shape[1] == len(model["features"]), "L2 feature matrix shape mismatch")
    out = np.full((len(x), 1), float(model["baseline"]), dtype=np.float64)
    for tree in model["trees"]:
        feature = np.asarray(tree["feature_idx"], dtype=np.int32)
        threshold = np.asarray(tree["num_threshold"], dtype=np.float64)
        missing_left = np.asarray(tree["missing_go_to_left"], dtype=np.bool_)
        left, right = np.asarray(tree["left"], dtype=np.int32), np.asarray(tree["right"], dtype=np.int32)
        leaf, value = np.asarray(tree["is_leaf"], dtype=np.bool_), np.asarray(tree["value"], dtype=np.float64)
        nodes = np.zeros(len(x), dtype=np.int32)
        while True:
            active = np.flatnonzero(~leaf[nodes])
            if not len(active):
                break
            current = nodes[active]
            vals = x[active, feature[current]]
            go_left = np.where(np.isnan(vals), missing_left[current], vals <= threshold[current])
            nodes[active] = np.where(go_left, left[current], right[current])
        # sklearn's TreePredictor node values already include learning-rate shrinkage.
        out[:, 0] += value[nodes]
    return out


def probabilities(raw):
    x = np.asarray(raw, dtype=np.float64)
    def expit(v):
        try:
            return 1.0 / (1.0 + math.exp(-float(v)))
        except OverflowError:
            return 0.0
    return np.fromiter((expit(v) for v in x.flat), dtype=np.float64, count=x.size).reshape(x.shape)


def select_choices(scored):
    grouped = {}
    for records in scored.values() if isinstance(scored, dict) else (scored,):
        for r in records:
            grouped.setdefault(r["chain"], []).append(r)
    return {chain: max(records, key=lambda r: (r["score"], r["is_sx"], r["is_s0"]))
            for chain, records in grouped.items()}


def score_records(model, records):
    if not records:
        return []
    x = np.asarray([[float(str(r[k])) for k in FEATURES] for r in records], dtype=np.float64)
    _require(np.isfinite(x).all(), "non-finite feature values")
    return [dict(r, score=float(p)) for r, p in zip(records, probabilities(raw_predict(model, x))[:, 0])]


def choose_item(vit_path, sx_path, out_path, genome, cache, row_name, wmax=3000, model=None):
    vit, _ = read_bed(vit_path, coding=True)
    sx, _ = read_bed(sx_path, coding=True)
    records = make_table_records(row_name, vit, sx, genome, cache, wmax)
    choices = select_choices(score_records(model or read_model(), records))
    changes = apply_choices(vit_path, records, choices, out_path)
    return records, choices, changes


def emission_cache(la, lb, cc, utr7, cast=None):
    """f16 copies of the item's f32 emissions in the emit-cache layout; cast may reuse the worker's chunked caster."""
    cast = cast or (lambda a: a.astype(np.float16, order="C"))
    return {"c": cast(cc[:, :4]), "log_a": cast(la), "log_b": cast(lb), "utr7": cast(utr7)}
