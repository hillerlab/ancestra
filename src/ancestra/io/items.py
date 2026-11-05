"""One worker's ``run_items(plan)``: per item, inference, the CDS Viterbi decode, START extension, the L2 START
chooser, the UTR decoders, the gene scores, optional tracks and alternative isoforms, and the item's parts under
``PREFIX.parts/``. The CLI (``ancestra.cli``) spawns the workers and merges the parts.

This module imports neither torch nor the model at import time: the CLI's name preflight stays torch-free.

**Weights.** ``infer.build_model(plan["weights"], params["recipe"]["label_tag"])``:
``torch.load(..., weights_only=True)``.

**``plan["params"]`` (the sealed params.json).** ``recipe`` {amp, window, stride, tta, bs, decode_chunk, decode_overlap,
emit_native ``"utr7"``, label_tag}; ``viterbi`` (the declared decoder params); ``start`` {tau, window}; ``l2``
{model_sha256, wmax}; ``decoder_b`` {sides ``"5"``, tau, w_end, lambda_prior, window5, eps, held, priors}; ``tracks``
{scale}. Held and priors are written once under ``PREFIX.parts/`` and read with ``utr_decode.load_held_params`` /
``load_priors``.
"""
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np

from ancestra import DATA_DIR, _native

# Compared on resume. Timings and peak RSS are recorded and not compared: they are the run's
# results. A skip also needs every recorded output on disk with its sha256 and record count; RAM bounds
# live in PREFIX.prov.json and are not in this dict. Adding tracks is not a new identity: an item
# whose identity matches but whose run files are absent is redone.
_IDENT = ("stamp", "item", "recipe", "device", "gpu", "runtime", "ssd_backend", "triton_version",
          "twobit", "ckpt_sha256", "params_sha256", "weights_sha256", "track_features", "track_scale")
# Part filenames. Bytes outside this set are upper-case %XX, the same rule as the stamp's 2bit basename.
_KEY_SAFE = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
# NAME_MAX is 255. The longest component is the track temp `{key}.{feature}.{strand}.runs.tmp.{pid}.{id}`
# (tracks.rs TempRun: pid is a u32, id a u64, feature one of the eight track names). vit/done temps are shorter.
_KEY_MAX = 255 - max(
    [len(f".vit.bed.{2**32 - 1}.part"), len(f".done.{2**32 - 1}.part")]
    + [len(f".{feat}.{strand}.runs.tmp.{2**32 - 1}.{2**64 - 1}")
       for feat in ("cds", "exon", "fiveutr", "threeutr", "donor", "acceptor", "start", "stop")
       for strand in ("plus", "minus")])
_UTR = "B5+DP3"


def _die(msg):
    raise ValueError(msg)


def _need(obj, key, where):
    if not isinstance(obj, dict) or key not in obj:
        _die(f"{where}: missing {key}")
    return obj[key]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.part"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _item_name(it):
    """``parse_specs``'s name: the chrom, or ``chrom_start_end`` for a range."""
    if it["spec"] == it["chrom"]:
        return it["chrom"]
    return f"{it['chrom']}_{it['start']}_{it['end']}"


def _part_key(name):
    """One path component for an item.

    Bytes outside ``[A-Za-z0-9._-]`` are upper-case ``%XX``. That string is the key while it fits in
    ``_KEY_MAX``; ``urllib.parse.unquote`` inverts those short keys. A longer encoding is fixed-length:
    a prefix of the encoding, ``_``, and the sha256 of the name. ``.done`` stores ``key`` and the true
    name (``item``). ``.`` and ``..`` are refused. Rows and run headers keep the true chrom.
    """
    enc = "".join(chr(b) if b in _KEY_SAFE else f"%{b:02X}" for b in name.encode("utf-8"))
    if enc in (".", ".."):
        _die(f"item name {name!r} is not a part filename")
    if len(enc) <= _KEY_MAX:
        return enc
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    head = _KEY_MAX - len(digest) - 1
    if head < 1:
        _die(f"part key budget {_KEY_MAX} cannot hold a sha256 suffix")
    return f"{enc[:head]}_{digest}"


def _bed12_row(chrom, s, e, name, score, strand, cds):
    cds = sorted(cds)
    n = len(cds)
    sizes = ",".join(str(b - a) for a, b in cds) + ","
    starts = ",".join(str(a - s) for a, b in cds) + ","
    return f"{chrom}\t{s}\t{e}\t{name}\t{score}\t{strand}\t{s}\t{e}\t0\t{n}\t{sizes}\t{starts}\n"


def _write_rows(path, all_rows):
    """Write the BED rows atomically: <path>.<pid>.part, then rename, so a crash never leaves a partial BED and two
    concurrent runs with one output name never share a staging file; a failed write removes its staging file."""
    tmp = "%s.%d.part" % (path, os.getpid())
    try:
        with open(tmp, "w") as fh:
            for _, _, line in all_rows:
                fh.write(line)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def peak_rss_kb():
    # VmHWM belongs to the address space created at exec. ru_maxrss survives fork+exec on Linux, so a
    # child of a large parent (pytest after full-size model tests, a wrapper script) reports the
    # PARENT's high-water mark instead of its own.
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _check_plan(plan):
    for key in ("twobit", "twobit_sha256", "prefix", "items", "tracks", "track_features", "weights",
                "weights_sha256", "stamp", "params", "worker", "n_workers", "version", "checkpoint",
                "params_sha256"):
        if key not in plan:
            _die(f"plan missing {key}")
    if not os.path.isfile(plan["twobit"]):
        _die(f"2bit not found: {plan['twobit']}")
    if _sha256(plan["twobit"]) != plan["twobit_sha256"]:
        _die(f"{plan['twobit']}: sha256 != plan twobit_sha256")
    params = plan["params"]
    recipe = _need(params, "recipe", "params")
    for key in ("amp", "window", "stride", "tta", "bs", "decode_chunk", "decode_overlap", "emit_native",
                "label_tag"):
        _need(recipe, key, "params.recipe")
    if recipe["emit_native"] != "utr7":
        _die(f"params.recipe.emit_native {recipe['emit_native']!r} != 'utr7'")
    w, s = recipe["window"], recipe["stride"]
    if isinstance(w, bool) or not isinstance(w, int) or isinstance(s, bool) or not isinstance(s, int) \
            or not 0 < s <= w:
        _die(f"params.recipe stride {s!r} is not an int with 0 < stride <= window {w!r}")
    _need(params, "viterbi", "params")
    start = _need(params, "start", "params")
    _need(start, "tau", "params.start")
    _need(start, "window", "params.start")
    l2 = _need(params, "l2", "params")
    model_sha, wmax = _need(l2, "model_sha256", "params.l2"), _need(l2, "wmax", "params.l2")
    if not (isinstance(model_sha, str) and len(model_sha) == 64 and
            all(c in "0123456789abcdef" for c in model_sha)):
        _die("params.l2.model_sha256 is not 64 lower-case hex")
    if isinstance(wmax, bool) or not isinstance(wmax, int) or wmax < 1:
        _die(f"params.l2.wmax {wmax!r} is not a positive integer")
    dec = _need(params, "decoder_b", "params")
    for key in ("sides", "tau", "w_end", "lambda_prior", "window5", "eps", "held", "priors"):
        _need(dec, key, "params.decoder_b")
    if dec["sides"] != "5":
        _die(f"params.decoder_b.sides {dec['sides']!r} != '5'")
    if not isinstance(dec["held"], dict) or not isinstance(dec["priors"], dict):
        _die("params.decoder_b held and priors must be JSON objects")
    scale = _need(_need(params, "tracks", "params"), "scale", "params.tracks")
    if isinstance(scale, bool) or not isinstance(scale, int) or not 1 <= scale <= 65535:
        _die(f"params.tracks.scale {scale!r} is not an int in 1..65535")
    if plan["tracks"] and not plan["track_features"]:
        _die("tracks requested but track_features is empty")
    ckpt = plan["checkpoint"]
    sha = _need(ckpt, "sha256", "plan.checkpoint") if isinstance(ckpt, dict) else _die("plan.checkpoint is not an object")
    for label, value in (("checkpoint.sha256", sha), ("params_sha256", plan["params_sha256"])):
        if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)):
            _die(f"{label} is not a 64-hex sha256")
    seen, seen_keys = {}, {}
    items = []
    for it in plan["items"]:
        for key in ("chrom", "start", "end", "spec"):
            _need(it, key, "item")
        start_i, end_i = int(it["start"]), int(it["end"])
        if not 0 <= start_i < end_i:
            _die(f"item {it['spec']}: range [{start_i}, {end_i}) is empty")
        if it["chrom"] in seen:
            _die(f"{it['chrom']} is requested twice ({seen[it['chrom']]}, {it['spec']}): one item per sequence")
        seen[it["chrom"]] = it["spec"]
        item = dict(it, start=start_i, end=end_i)
        name = _item_name(item)
        key = _part_key(name)  # refuses '.' / '..' before any part file is created
        if key in seen_keys:
            _die(f"part key {key!r} is shared by {seen_keys[key]!r} and {name!r}")
        seen_keys[key] = name
        items.append(item)
    n = int(plan["n_workers"])
    worker = int(plan["worker"])
    if n < 1 or not 0 <= worker < n:
        _die(f"worker {worker} of n_workers {n} is out of range")
    if plan.get("compile") is not None and plan["compile"] != "reduce-overhead":
        _die(f"plan.compile {plan['compile']!r} is not 'reduce-overhead'")
    if "alt" in plan:
        from ancestra.decode.alt import model_fault
        alt = plan["alt"]
        t, sha, n_alt, doc = (_need(alt, k, "plan.alt") for k in ("threshold", "sha256", "n_alt", "model"))
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t <= 1:
            _die(f"plan.alt.threshold {t!r} is not in (0, 1]")
        if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
            _die("plan.alt.sha256 is not a 64-hex sha256")
        if isinstance(n_alt, bool) or not isinstance(n_alt, int) or n_alt < 1:
            _die(f"plan.alt.n_alt {n_alt!r} is not a positive int")
        why = model_fault(doc)
        if why:
            _die(f"plan.alt.model: {why}")
    return items


def cast_f16(buf):
    """Cast ``utr7`` first, drop ``nat``, then make the other f16 copies.

    Each cast fills a preallocated C-order float16 array of the same shape in chunks along the last axis on a
    thread pool of ``buf["cast_threads"]`` threads (run_items sizes it from --threads), casting straight into
    the destination with no chunk-sized temporary, with the same elementwise numpy cast (numpy releases the GIL
    there)."""
    n = max(1, min(8, int(buf.get("cast_threads") or 1)))

    def _one(src, out, s, e):
        np.copyto(out[..., s:e], src[..., s:e], casting="unsafe")

    def _all(src, ex):
        out = np.empty(src.shape, dtype=np.float16, order="C")
        L = src.shape[-1] if src.ndim else 0
        step = max(1, (L + n - 1) // n)
        futs = []
        for s in range(0, L, step):
            try:
                futs.append(ex.submit(_one, src, out, s, min(s + step, L)))
            except RuntimeError:
                _one(src, out, s, L)  # this slice and every later one run here; the cast is elementwise
                break
        for f in futs:
            f.result()
        return out

    nat = buf.pop("nat")
    from ancestra.decode import l2_choose as lc
    with ThreadPoolExecutor(max_workers=n) as ex:
        utr7 = _all(nat, ex)
        del nat
        cache = lc.emission_cache(buf["la"], buf["lb"], buf["cc"], utr7,
                                  lambda src: src if src.dtype == np.float16 else _all(src, ex))
        la1 = _all(buf["la"][:, 1], ex)
    buf["l2_emit_cache"] = cache
    return cache["utr7"], la1, cache["c"]


def gene_score(la1, item_start, blocks, thick, strand, table):
    """The BED score. m is the f64 mean of exp16[bits(la1[s, i − item_start])] over exon ∩ [thickStart, thickEnd)
    (stop codon included); score = ⌊1000·m + ½⌋ clamped to 0..1000. Returns (score, m)."""
    if strand not in ("+", "-"):
        _die(f"strand {strand!r} is not + or -")
    if getattr(la1, "dtype", None) != np.float16 or getattr(la1, "ndim", None) != 2 or la1.shape[0] != 2:
        _die(f"la1 must be float16 [2, L], got {getattr(la1, 'dtype', None)} {getattr(la1, 'shape', None)}")
    plane = 0 if strand == "+" else 1
    bits = np.ascontiguousarray(la1).view(np.uint16)
    length = bits.shape[1]
    ts, te = int(thick[0]), int(thick[1])
    parts = []
    for a, b in blocks:
        lo, hi = max(int(a), ts), min(int(b), te)
        if lo >= hi:
            continue
        i0, i1 = lo - item_start, hi - item_start
        if i0 < 0 or i1 > length:
            _die(f"CDS [{lo}, {hi}) outside the item starting at {item_start} (length {length})")
        parts.append(table[bits[plane, i0:i1]])
    if not parts:
        _die(f"CDS [{ts}, {te}) meets no exon block")
    m = math.fsum(np.concatenate(parts).tolist()) / sum(p.shape[0] for p in parts)
    score = math.floor(1000.0 * m + 0.5)
    return (0 if score < 0 else 1000 if score > 1000 else score), m


def start_prob(c4, item_start, thick, strand):
    """The f16 START-head at the A of the start codon. ``+`` is thickStart, ``−`` is thickEnd − 1."""
    if strand not in ("+", "-") or getattr(c4, "dtype", None) != np.float16 or c4.ndim != 3 or c4.shape[1] != 4:
        _die(f"start prob needs a strand and float16 c4 with 4 channels, got {strand!r} {getattr(c4, 'shape', None)}")
    pos = int(thick[0]) if strand == "+" else int(thick[1]) - 1
    idx = pos - item_start
    if not 0 <= idx < c4.shape[2]:
        _die(f"start codon {pos} outside the item starting at {item_start}")
    return float(c4[0 if strand == "+" else 1, 2, idx])


def _fmt(x):
    return format(float(x), ".17g")


def _tx_line(tx, score, m, gene_score_f, start_p, extended, shift, version, ckpt_sha, params_sha):
    blocks = tx["blocks"]
    if not blocks:
        _die(f"{tx['name']}: no blocks")
    s0, e0 = blocks[0][0], blocks[-1][1]
    ts, te = tx["thick"]
    fields = [tx["chrom"], s0, e0, tx["name"], int(score), tx["strand"], int(ts), int(te), 0, len(blocks),
              ",".join(str(b - a) for a, b in blocks) + ",",
              ",".join(str(a - s0) for a, b in blocks) + ",",
              tx["name"], _fmt(gene_score_f), _fmt(m), _fmt(start_p), 1 if extended else 0, int(shift),
              len(tx["five_iv"]), len(tx["three_iv"]), _UTR, f"ancestra-{version}", ckpt_sha, params_sha]
    if len(fields) != 24:
        _die(f"{tx['name']}: part row has {len(fields)} columns, not 24")
    return "\t".join(map(str, fields))


def _load_decoder_b(dec, parts):
    """Write the inline held and priors JSON once and parse them with the 5' UTR decoder's loaders."""
    from ancestra.decode import utr_decode as ud
    held_path = os.path.join(parts, "held.json")
    priors_path = os.path.join(parts, "priors.json")
    # Once per run (this is called once), not once per item. A later run whose params changed must not keep the old file.
    _json_atomic(held_path, dec["held"])
    _json_atomic(priors_path, dec["priors"])
    held, _sha = ud.load_held_params(held_path)
    return held, ud.load_priors(priors_path)


def _start_and_b(vit, twobit, groups, start, dec, held, priors, sx_path, executor=None,
                 l2_cache=None, l2_path=None, l2=None):
    """START, L2, then the 5' UTR decoder, in this order.

    ``executor`` goes to ``decode_txs`` (None: a sequential loop)."""
    from ancestra.decode import l2_choose as lc
    from ancestra.decode import start_extend as se
    from ancestra.decode import utr_attach as ua
    from ancestra.decode import utr_decode as ud
    from ancestra.decode.tx import load_bed
    rows, cands = se.scan(vit, twobit, int(start["window"]), groups)
    sx_rows, _sx_changes = se.changes_of(rows, cands, start["tau"], int(start["window"]))
    se.write_bed(sx_path, sx_rows)
    if l2_cache is None:
        _die("L2 emissions are missing")
    if l2_path is None or not isinstance(l2, dict):
        _die("L2 output or params are missing")
    chrom = next(iter(groups))
    item_start = groups[chrom][0]
    genome = _native.Genome(twobit)
    _records, _choices, changes = lc.choose_item(vit, sx_path, l2_path, genome,
                                                  {chrom: (l2_cache, item_start, item_start + l2_cache["c"].shape[-1])},
                                                  os.path.basename(vit).removesuffix(".vit.bed"), int(l2["wmax"]))
    txs = ud.decode_txs(load_bed(l2_path), groups, ua, priors, int(dec["window5"]), float(dec["w_end"]),
                        float(dec["eps"]), float(dec["tau"]), "5", held, float(dec["lambda_prior"]), executor)
    return txs, changes


def _vit_lines(chrom, seq, la, lb, cc, start, params, chunk, overlap):
    """Named Viterbi BED12 rows sorted by (chrom bytes, CDS start), plus the unclipped f32 score by name.

    ``decode_strand`` returns 5-tuples with the clipped integer score. The unclipped score rides in ``raw``
    so a sort of the tuples, and therefore the bed bytes, do not depend on it.
    """
    from ancestra.decode.viterbi import decode_strand
    raw = {}
    rows = decode_strand(seq, la, lb, cc, 0, start, params, chunk, overlap, raw)
    rows += decode_strand(seq, la, lb, cc, 1, start, params, chunk, overlap, raw)
    rows = sorted(rows)
    rows.sort(key=lambda r: (chrom.encode(), r[0]))  # stable: (chrom bytes, CDS start)
    lines, scores = [], {}
    for n, row in enumerate(rows, 1):
        s, e, strand, sc, cds = row
        name = f"ancestra_{chrom}_{n}"
        if id(row) not in raw:
            _die(f"{name}: decode produced no unclipped score")
        scores[name] = raw[id(row)]
        lines.append((chrom, s, _bed12_row(chrom, s, e, name, sc, strand, cds)))
    return lines, scores


def _runs_ok(base, features):
    for feature in features:
        for strand in ("plus", "minus"):
            if not os.path.isfile(f"{base}.{feature}.{strand}.runs"):
                return False
    return True


def _record_n(path, meta):
    """The count a ``.done`` output records: run records, or non-empty lines of a text part."""
    if "feature" in meta:
        with open(path, "rb") as fh:
            head = fh.read(38)
            if len(head) < 38 or head[:8] != b"APRRUNS\0":
                return -1
            header = 38 + int.from_bytes(head[30:34], "little") + int.from_bytes(head[34:36], "little")
            fh.seek(0, os.SEEK_END)
            body = fh.tell() - header
        if header < 38 or body < 0 or body % 10:
            return -1
        return body // 10
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def _output_fault(done, alt=False):
    """``None`` when every output named in ``.done`` is present with its sha256 and record count.

    Otherwise ``(path, field)``: ``sha256`` or ``n`` (a missing file is ``sha256``), or ``outputs``
    when the record has no output map. Finalize names this field (exit 1). An ``"alt": true`` output (the
    ``--alt`` part) is checked only under ``alt``: without ``--alt`` it neither redoes nor fails an item."""
    outputs = done.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return "", "outputs"
    for path, meta in outputs.items():
        if not isinstance(meta, dict):
            return path, "outputs"
        if meta.get("alt") is True and not alt:
            continue
        try:
            if _sha256(path) != meta.get("sha256"):
                return path, "sha256"
            n = meta.get("n")
            if isinstance(n, bool) or not isinstance(n, int) or _record_n(path, meta) != n:
                return path, "n"
        except OSError:
            return path, "sha256"
    return None


def _outputs_ok(done, alt=False):
    """Every output named in ``.done`` is present with the recorded sha256 and record count."""
    return _output_fault(done, alt) is None


def _alt_ok(done, alt, part):
    """Under ``--alt``: ``.done`` records this run's threshold and alt.json sha and the item's alt ``part``. Its
    bytes are ``_output_fault(done, True)``'s to check."""
    rec, meta = done.get("alt"), done.get("outputs", {}).get(part)
    counts = rec.get("counts") if isinstance(rec, dict) else None
    return (isinstance(rec, dict) and type(rec.get("threshold")) in (int, float)
            and rec["threshold"] == alt["threshold"] and rec.get("sha256") == alt["sha256"]
            and isinstance(counts, dict) and all(isinstance(counts.get(st), dict) and
                all(type(n) is int and n >= 0 for n in counts[st].values()) for st in ("+", "-"))
            and type(rec.get("vmhwm_kb")) is int and rec["vmhwm_kb"] >= 0
            and type(rec.get("max_window_bp")) is int and rec["max_window_bp"] >= 0
            and isinstance(meta, dict) and meta.get("alt") is True)


def _identity_ok(done, ident):
    if not isinstance(done, dict):
        return False
    if done.get("compile") != ident.get("compile"):
        return False
    want = json.loads(json.dumps({k: ident[k] for k in _IDENT}))
    return {k: done.get(k) for k in _IDENT} == want


def _read_done(path):
    try:
        with open(path) as fh:
            done = json.load(fh)
    except (OSError, ValueError):
        return None
    return done if isinstance(done, dict) else None


def done_records(parts):
    """Item ``.done`` records under ``parts``. A directory listing, never a glob: a key may start with ``.``.

    The CLI's finalize enumerates parts from these records. Each record maps ``key`` ↔ ``item["name"]``.
    """
    if not os.path.isdir(parts):
        _die(f"no parts directory: {parts}")
    out = []
    for entry in sorted(os.listdir(parts)):
        if not entry.endswith(".done"):
            continue
        path = os.path.join(parts, entry)
        if not os.path.isfile(path):
            continue
        done = _read_done(path)
        item = done.get("item") if isinstance(done, dict) else None
        key = done.get("key") if isinstance(done, dict) else None
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(key, str) or not key or not isinstance(name, str) or entry != f"{key}.done":
            _die(f"{path}: .done does not map its key to the item name")
        out.append(done)
    return out


def _finish(buf):
    try:
        _finish_item(buf)
    finally:
        buf.clear()  # the dict is empty before the worker drops the argument


def _finish_item(buf):
    t1 = time.perf_counter()
    item = buf["item"]
    chrom, start = item["chrom"], item["start"]
    recipe = buf["recipe"]
    utr7, la1, c4 = cast_f16(buf)
    lines, gene_scores = _vit_lines(chrom, buf["seq"], buf["la"], buf["lb"], buf["cc"], start, buf["viterbi"],
                                    int(recipe["decode_chunk"]), int(recipe["decode_overlap"]))
    _write_rows(buf["vit"], lines)
    _finish_decoded(buf, utr7, la1, c4, gene_scores, t1, _decode_executor(buf))


def _decode_executor(buf):
    """The 5' UTR decoder's executor for this item: the worker's alt spawn pool under ``--alt``, else the ``run_items``
    spawn pool (``buf["dec_ex"]``, None when too small: sequential). None once ``decode_txs`` marked
    the pool broken: later items of the worker decode in the parent."""
    ex = buf["alt_ex"] if buf["plan_bits"].get("alt") else buf.get("dec_ex")
    if ex is not None and getattr(ex, "_ancestra_broken", False):
        return None
    return ex


def _finish_decoded(buf, utr7, la1, c4, gene_scores, t1, executor):
    """START, the 5' UTR decoder, the BED score, the metadata and the parts, once the Viterbi bed is on disk.

    ``utr7``, ``la1`` and ``c4`` are the f16 copies, ``gene_scores`` the unclipped Viterbi scores by name. ``t1`` is
    the finish clock ``_finish_item`` started before Viterbi. ``executor`` is the 5' UTR decoder's pool (None: sequential).
    """
    item, name = buf["item"], buf["name"]
    chrom, start = item["chrom"], item["start"]
    plan = buf["plan_bits"]
    groups = {chrom: (start, utr7, c4)}
    txs, changes = _start_and_b(buf["vit"], plan["twobit"], groups, buf["start"], buf["dec"], buf["held"],
                                buf["priors"], buf["sx"], executor, buf["l2_emit_cache"], buf["l2_bed"],
                                plan["l2"])
    by = {c["gene"]: c for c in changes}
    if len(by) != len(changes):
        _die(f"{name}: two START changes share a gene name")
    table = _native.exp16_table()
    out_lines = []
    for tx in txs:
        if tx["name"] not in gene_scores:
            _die(f"{tx['name']}: the 5' UTR decoder returned a transcript the Viterbi did not name")
        score, m = gene_score(la1, start, tx["blocks"], tx["thick"], tx["strand"], table)
        sp = start_prob(c4, start, tx["thick"], tx["strand"])
        gs = gene_scores[tx["name"]]
        if not (math.isfinite(m) and math.isfinite(sp) and math.isfinite(gs)):
            _die(f"{tx['name']}: non-finite geneScore, cdsProb or startProb")
        ch = by.get(tx["name"])
        upstream = ch is not None and (ch["new_start"] < ch["old_start"] if tx["strand"] == "+"
                                       else ch["new_start"] > ch["old_start"])
        out_lines.append(_tx_line(tx, score, m, gs, sp, upstream, 0 if ch is None else ch["distance"],
                                  plan["version"], plan["ckpt_sha256"], plan["params_sha256"]))
    with open(buf["tx"], "w") as fh:
        fh.write("".join(line + "\n" for line in out_lines))
    outputs = {}
    for kind, path in (("vit", buf["vit"]), ("sx", buf["sx"]), ("l2", buf["l2_bed"]), ("tx", buf["tx"])):
        with open(path, "rb") as fh:
            n = sum(1 for line in fh if line.strip())
        outputs[path] = {"sha256": _sha256(path), "n": n}
    if plan["tracks"]:
        runs = _native.track_runs(buf["run_prefix"], chrom, start, la1.view(np.uint16), utr7.view(np.uint16),
                                    c4.view(np.uint16),
                                    list(plan["track_features"]), plan["track_scale"])
        for path, count in runs.items():
            stem, strand = path[: -len(".runs")].rsplit(".", 1)
            _stem, feature = stem.rsplit(".", 1)
            outputs[path] = {"sha256": _sha256(path), "n": int(count), "feature": feature, "strand": strand}
    alt = plan.get("alt")
    if alt:  # the item's alternatives from its window arrays (the f32 buffers)
        from ancestra.decode import alt as alt_mod
        from ancestra.decode.tx import load_bed
        part = buf["tx"][: -len(".tx.tsv")] + ".alt.tsv"
        vit = [(t["name"], t["chrom"], t["strand"], t["cds"]) for t in load_bed(buf["vit"])]
        got = alt_mod.run_item(buf["alt_ex"], alt["n_alt"], plan["twobit"], item, vit, txs, changes,
                               (buf["la"], buf["lb"], buf["cc"]), buf["viterbi"], alt["model"],
                               alt["threshold"], part,
                               (plan["version"], plan["ckpt_sha256"], plan["params_sha256"], alt["sha256"]))
        outputs[part] = {"sha256": _sha256(part), "n": got["n"], "alt": True}
    done = dict(buf["ident"], outputs=outputs, timings={"infer_s": buf["infer_s"],
                "finish_s": time.perf_counter() - t1, "wall_start": time.time()}, peak_rss_kb=peak_rss_kb(),
                worker=buf["worker"])  # not in _IDENT: which worker ran it does not change the part
    if alt:  # observations, never compared on resume: only the threshold and the alt.json sha are (_alt_ok)
        done["alt"] = {"threshold": alt["threshold"], "sha256": alt["sha256"], "counts": got["counts"],
                       "vmhwm_kb": got["vmhwm_kb"], "max_window_bp": got["max_window_bp"]}
    _json_atomic(buf["done"], done)  # provenance last, so a crash leaves no matching .done
    print(f"{name}: n={len(txs)}", flush=True)


def _maybe_enable_fast_kernels(model, device, window, amp, compile=None):
    """Enable the fast kernels, keeping the model unchanged when the import itself fails. With
    compile="reduce-overhead" (--compile) a refusal exits 2 with the line, before any item is written."""
    try:
        from ancestra.model import fast_kernels
    except Exception as e:
        if compile is not None:
            print(f"ancestra: --compile failed: fast kernels import ({type(e).__name__})", file=sys.stderr,
                  flush=True)
            raise SystemExit(2)
        print(f"ancestra: fast kernels off(import): {type(e).__name__}", file=sys.stderr, flush=True)
        return model
    try:
        return fast_kernels.enable(model, device, int(window), amp, compile=compile)
    except fast_kernels.CompileError as e:
        print(f"ancestra: --compile failed: {e.reason}", file=sys.stderr, flush=True)
        raise SystemExit(2)


def _device_of(plan):
    import torch
    cuda = torch.cuda.is_available()
    device = "cuda:0" if cuda else "cpu"
    gpu = torch.cuda.get_device_name(0) if cuda else ""
    runtime = {"torch": torch.__version__, "cuda": torch.version.cuda or "", "numpy": np.__version__}
    return device, gpu, runtime


def run_items(plan):
    """Finish this worker's items. One item per chrom; largest first (``lpt_shards``).

    The model is loaded only when an item of this shard still needs work (adding ``--gtf`` /
    ``--bb`` on a finished run loads no model)."""
    from ancestra.io.specs import lpt_shards
    items = _check_plan(plan)
    l2_path = os.path.join(DATA_DIR, "l2_model.json")
    if _sha256(l2_path) != plan["params"]["l2"]["model_sha256"]:
        _die("L2 model sha256 differs from params.l2.model_sha256")
    if not items:
        return 0
    parts = plan["prefix"] + ".parts"
    if os.path.isdir(parts):
        owners = {d["key"]: d["item"]["name"] for d in done_records(parts)}
        for it in items:
            name = _item_name(it)
            key = _part_key(name)
            if key in owners and owners[key] != name:
                _die(f"part key {key!r} is shared by {owners[key]!r} and {name!r}")
    os.makedirs(parts, exist_ok=True)
    device, gpu, runtime = _device_of(plan)
    if device.startswith("cuda"):
        import triton
        ssd_backend, triton_version = "triton", triton.__version__
    else:
        ssd_backend, triton_version = "ref", None
    recipe = plan["params"]["recipe"]
    twobit = {"path": os.path.abspath(plan["twobit"]), "size": os.path.getsize(plan["twobit"]),
              "sha256": plan["twobit_sha256"]}
    tuples = [(_item_name(it), it["chrom"], it["start"], it["end"], it) for it in items]
    # The shard is n_workers/worker. The CLI spawns one process per device, or per
    # --n-workers when no device list is given; this process runs only its shard.
    mine = lpt_shards(tuples, int(plan["n_workers"]))[int(plan["worker"])]
    alt = plan.get("alt")
    plan_bits = {"twobit": plan["twobit"], "version": plan["version"], "ckpt_sha256": plan["checkpoint"]["sha256"],
                 "params_sha256": plan["params_sha256"], "tracks": bool(plan["tracks"]),
                 "track_features": list(plan["track_features"]), "track_scale": plan["params"]["tracks"]["scale"],
                 "alt": alt, "l2": plan["params"]["l2"]}
    todo = []
    for name, chrom, rs, re_, it in mine:
        key = _part_key(name)
        ident = {"stamp": plan["stamp"], "key": key, "item": {"name": name, "chrom": chrom, "start": rs,
                 "end": re_, "spec": it["spec"]}, "recipe": recipe, "device": device, "gpu": gpu,
                 "runtime": runtime, "ssd_backend": ssd_backend, "triton_version": triton_version,
                 "twobit": twobit, "ckpt_sha256": plan_bits["ckpt_sha256"],
                 "params_sha256": plan_bits["params_sha256"], "weights_sha256": plan["weights_sha256"],
                 "track_features": plan_bits["track_features"], "track_scale": plan_bits["track_scale"]}
        if "compile" in plan:  # opt-in only: a default .done carries no compile key
            ident["compile"] = plan["compile"]
        base = os.path.join(parts, key)
        done_path = base + ".done"
        done = _read_done(done_path)
        if _identity_ok(done, ident) and _outputs_ok(done, alt is not None) and (
                not plan["tracks"] or _runs_ok(base, plan["track_features"])) and (
                alt is None or _alt_ok(done, alt, base + ".alt.tsv")):
            print(f"{name}: done, skip", flush=True)
            continue
        todo.append((name, chrom, rs, re_, it, ident, base, done_path))
    if not todo:
        return 0
    from ancestra.model import infer
    held, priors = _load_decoder_b(plan["params"]["decoder_b"], parts)
    try:
        _aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        _aff = os.cpu_count() or 1
    _threads = int(plan.get("threads") or 0) or _aff
    if plan.get("compile") is not None:  # one inductor/triton cache dir per worker, stable across runs
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir
        base = os.environ["TORCHINDUCTOR_CACHE_DIR"] if "TORCHINDUCTOR_CACHE_DIR" in os.environ \
            else default_cache_dir()
        cdir = os.path.join(base, f"ancestra-w{int(plan['worker'])}")
        os.makedirs(cdir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cdir
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cdir, "triton")
        import torch._inductor.config as _icfg
        _icfg.compile_threads = max(1, min(_icfg.compile_threads, _threads // int(plan["n_workers"])))
    model = infer.build_model(plan["weights"], recipe["label_tag"]).to(device)
    model = _maybe_enable_fast_kernels(model, device, recipe["window"], recipe["amp"],
                                       compile=plan.get("compile"))
    genome = _native.Genome(plan["twobit"])
    refused = []
    done_n = 0
    dec_ex_ctx = nullcontext()
    _cast_n = max(1, min(8, _threads // int(plan["n_workers"])))
    if alt:
        from ancestra.decode.alt import pool as alt_pool
        alt_ex_ctx = alt_pool(alt["n_alt"])
    else:
        from ancestra.decode.alt import pool as spawn_pool  # the same spawn pool, sized like the --alt pool
        _n_dec = min(16, _threads // int(plan["n_workers"]) - 1)
        alt_ex_ctx = nullcontext()
        if _n_dec >= 2:
            try:
                dec_ex_ctx = spawn_pool(_n_dec)
            except (OSError, RuntimeError) as e:
                from ancestra.decode.utr_decode import _note_pool_broken
                _note_pool_broken(None, e)  # no pool: the sequential path
    # This worker's alt processes. Its context exits after the finishing thread's, so the pool outlives the last item.
    # The decoder pool (no --alt) is sized and ordered the same way; below 2 it stays None (sequential).
    with alt_ex_ctx as alt_ex, dec_ex_ctx as dec_ex, ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for name, chrom, rs, re_, it, ident, base, done_path in todo:
            t0 = time.perf_counter()
            seq = bytes(genome.fetch(chrom, rs, re_))
            if len(seq) != re_ - rs:
                _die(f"{name}: fetch returned {len(seq)} bytes, not {re_ - rs}")
            if os.path.exists(done_path):
                os.remove(done_path)  # a crash while rewriting must not leave the old match in place
            try:
                la, lb, cc, nat = infer.infer_arrays(seq, model, int(recipe["window"]), int(recipe["stride"]), device,
                                                     int(recipe["bs"]), recipe["amp"], recipe["tta"])
            except infer.NonfiniteEmissions as e:
                print(f"{name}: {e}: refusing the sequence (predict-nonfinite)", file=sys.stderr, flush=True)
                refused.append(name)
                continue
            # The main loop names nothing of item k after submit: F_k's one dict holds the buffers.
            buf = {"seq": seq, "la": la, "lb": lb, "cc": cc, "nat": nat, "item": it, "name": name,
                   "infer_s": time.perf_counter() - t0, "viterbi": plan["params"]["viterbi"], "recipe": recipe,
                   "start": plan["params"]["start"], "dec": plan["params"]["decoder_b"], "held": held,
                   "priors": priors, "plan_bits": plan_bits, "ident": ident, "worker": int(plan["worker"]),
                   "vit": base + ".vit.bed", "sx": base + ".sx.bed", "l2_bed": base + ".l2.bed",
                   "tx": base + ".tx.tsv", "done": done_path,
                   "run_prefix": base, "alt_ex": alt_ex, "dec_ex": dec_ex, "cast_threads": _cast_n}
            del seq, la, lb, cc, nat
            if pending is not None:
                pending.result()  # F_k finished and cleared its dict; item k is free
            pending = pool.submit(_finish, buf)
            del buf
            done_n += 1
        if pending is not None:
            pending.result()
    if refused:
        print(f"{len(refused)} sequence(s) refused (predict-nonfinite), e.g. {refused[:3]}", file=sys.stderr)
        raise SystemExit(4)
    return done_n
