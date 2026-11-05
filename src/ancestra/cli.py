"""The ``ancestra`` CLI. Flags, sealed files, items, the sequence-name preflight and the RAM print run before any model
is loaded. ``main`` then runs one worker in-process, or re-executes one worker per device (or per the hidden
``--n-workers``) and finalizes. A worker (``--no-merge``) never finalizes.

Nothing here imports torch at module scope, so the name preflight creates nothing and loads nothing.

**RELEASE.json's schema.** Every key is required and a missing or malformed one is an integrity failure (exit 1) naming
it: ``version`` (exactly ``1.0.0``), ``checkpoint`` (a JSON object with ``name``, ``sha256`` (64 hex), ``step``,
``tokens``), ``weights_sha256`` (64 hex), ``alt_sha256`` (64 hex), ``l2_model_sha256`` (64 hex), ``defaults_digest``
(64 hex, the compiled decoder defaults' digest, which the binding must match), ``runtime`` (a JSON object with
``python``, ``torch``, ``cuda``, ``numpy``) and ``parity_gpu`` (the GPU type of the parity runs). The stamp's fields
must be printable ASCII without `"` (``str.isprintable()`` accepts Unicode, so the bytes decide). The worker reads
``version``, ``checkpoint``, ``params_sha256`` and ``defaults_digest`` straight out of the plan this module builds,
and never parses the stamp.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter

from ancestra import DATA_DIR, _native
from ancestra.io.items import _alt_ok, _item_name, _json_atomic, _output_fault, done_records, peak_rss_kb, run_items
from ancestra.io.specs import lpt_shards, parse_specs, range_error

EXIT_BAD_ARGS = 2                          # a refused flag, spec or sequence name, before any work
EXIT_MISSING = 3                           # a requested sequence absent from the 2bit or out of range
EXIT_INTEGRITY = 1                         # an integrity failure (a sealed file, the weights' sha256)
B_I, B_F16, B_E = 137, 153, 165            # bytes per base: I (fetch + inference), the f16 item, E's transient peak
D_BYTES, H_BYTES = 9_000_000_000, 3_000_000_000            # D <= 9 GB decoded chunk, H <= 3 GB per process, decimal GB
# --alt: one alt process's peak A at a window of W_ALT, plus its 2 queued tasks of 42 B per base in the parent
A_BYTES, W_ALT = 710_000_000, 2_000_000
ALT_BYTES = A_BYTES + 2 * 42 * W_ALT                        # 0.88 GB per alt process
_READ_BLOCK = 1 << 20


def die(msg, code=EXIT_BAD_ARGS):
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_READ_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


# --- flags and list parsing -------------------------------------------------------------------------------------------

# List parsing, one function for both list flags: `ss`, `utr` and `all` expand; the result is deduplicated and emitted
# in canonical order. A spec is (flag, input tokens, canonical output order, the noun of an error, default occurrences).
EXPAND = {"ss": ("donor", "acceptor"), "utr": ("fiveutr", "threeutr"), "codon": ("start", "stop"),
          "all": ("cds", "exon", "fiveutr", "threeutr", "donor", "acceptor", "start", "stop")}
FORMATS = ("bed", "gtf", "gff", "bb")                      # the BED12 is always written; the others add files
FEATURES = ("cds", "exon", "fiveutr", "threeutr", "donor", "acceptor", "start", "stop")
FORMAT_SPEC = ("--format", FORMATS, FORMATS, "format", ("bed",))
FEATURE_SPEC = ("--track-feature", ("cds", "exon", "utr", "fiveutr", "threeutr", "ss", "donor", "acceptor",
                                    "codon", "start", "stop", "all"),
                FEATURES, "feature", ("all",))


def parse_list(values, spec):
    """List parsing: split every occurrence on `,`, strip spaces, no case folding; an empty or an unknown token is an
    error (exit 2); expand, deduplicate and emit in canonical order. Redundancy is never an error."""
    flag, tokens, canonical, kind, default = spec
    got = set()
    for raw in list(values) if values else default:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                die(f"{flag}: empty item in '{raw}'")
            if item not in tokens:
                die(f"{flag}: unknown {kind} '{item}' (choose from {', '.join(tokens)})")
            got.update(EXPAND.get(item, (item,)))
    return [c for c in canonical if c in got]


def read_specs(a):
    """--seqs and --seq-list (one spec per line, `#` comments), each with the flag it came from; the file adds to
    --seqs."""
    out = [(spec.strip(), "--seqs") for spec in (a.seqs.split(",") if a.seqs else [])]
    if a.seq_list:
        try:
            with open(a.seq_list) as fh:
                lines = fh.readlines()
        except OSError as e:
            die(f"--seq-list: {a.seq_list}: {e.strerror}")
        out += [(ln.split()[0], "--seq-list") for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    return out


def check_one_item_per_chrom(parsed_specs):
    """At most one item per sequence, whichever two specs ask for it (a whole sequence and a range of it count as
    two): the parts of one sequence are keyed by its name."""
    seen = {}
    for item, spec, flag in parsed_specs:
        _, chrom, _, _ = item
        if chrom in seen:
            die(f"{flag}: {chrom} is requested twice ({seen[chrom]}, {spec}): one item per sequence")
        seen[chrom] = spec


def parse_devices(value, check_visible=True):
    """Validate physical GPU ids in the parent; re-exec workers inherit a one-GPU visibility mask."""
    if value is None:
        return []
    message = "--devices must be unique non-negative integer GPU IDs within the visible GPU count"
    tokens = value.split(",")
    if any(not token or not all("0" <= c <= "9" for c in token) for token in tokens):
        die(message)
    try:
        devices = [int(token) for token in tokens]
    except ValueError:
        die(message)
    if len(set(devices)) != len(devices):
        die(message)
    if check_visible:
        import torch
        if any(device >= torch.cuda.device_count() for device in devices):
            die(message)
    return [str(device) for device in devices]


# --- the sealed files --------------------------------------------------------------------------------------------------

def sealed(name):
    """(path, document, sha256) of a sealed file in the package's data dir. A missing or malformed one is an integrity
    failure."""
    path = os.path.join(DATA_DIR, name)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        die(f"ancestra: no sealed {name} ({path})", EXIT_INTEGRITY)
    try:
        doc = json.loads(raw)
    except ValueError as e:
        die(f"ancestra: sealed {name} ({path}) is not JSON: {e}", EXIT_INTEGRITY)
    if not isinstance(doc, dict):
        die(f"ancestra: sealed {name} ({path}) is not a JSON object", EXIT_INTEGRITY)
    return path, doc, hashlib.sha256(raw).hexdigest()


def _stampable(value, hexlen=0):
    """One field of the stamp: printable ASCII bytes without `"` — 0x21-0x7E, so no space either, since the stamp is one
    line of space-separated fields. ``str.isprintable()`` accepts printable Unicode, so the bytes decide; with
    ``hexlen``, that many lower-case hex digits."""
    if not isinstance(value, str) or not value or '"' in value:
        return False
    try:
        raw = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    if not all(0x21 <= b <= 0x7E for b in raw):
        return False
    if hexlen:
        return len(value) == hexlen and all(c in "0123456789abcdef" for c in value)
    return True


def sealed_facts():
    """The install stamp, ``ancestra <version> git=<sha> rs=<sha>,clean ckpt=<name>
    ckpt_sha256=<64> weights=<64> params=<64>``, from RELEASE.json (the checkpoint's name and sha, the sealed weights
    sha), the binding's build sha and dirty flag (so a dirty build says so) and the sha of the sealed params.json, plus
    the parsed params themselves.

    RELEASE.json's schema is the module docstring's: every key required, a missing or malformed one an integrity failure
    (exit 1) naming it, and the stamp's fields printable ASCII without `"` with the version exactly ``1.0.0``. The
    binding's compiled decoder defaults must have RELEASE.json's digest."""
    _, sealed_doc, _ = sealed("RELEASE.json")
    _, params, params_sha = sealed("params.json")
    for key in ("version", "checkpoint", "weights_sha256", "defaults_digest", "runtime", "parity_gpu",
                "alt_sha256", "l2_model_sha256"):
        if key not in sealed_doc:
            die(f"ancestra: sealed RELEASE.json: missing key '{key}'", EXIT_INTEGRITY)
    ckpt, runtime = sealed_doc["checkpoint"], sealed_doc["runtime"]
    if not isinstance(ckpt, dict):
        die("ancestra: sealed RELEASE.json: 'checkpoint' is not a JSON object", EXIT_INTEGRITY)
    if not isinstance(runtime, dict):
        die("ancestra: sealed RELEASE.json: 'runtime' is not a JSON object", EXIT_INTEGRITY)
    for key, kind in (("name", str), ("sha256", str), ("step", int), ("tokens", int)):
        if key not in ckpt:
            die(f"ancestra: sealed RELEASE.json: missing key 'checkpoint.{key}'", EXIT_INTEGRITY)
        if isinstance(ckpt[key], bool) or not isinstance(ckpt[key], kind):
            die(f"ancestra: sealed RELEASE.json: checkpoint.{key} {ckpt[key]!r} is not a {kind.__name__}", EXIT_INTEGRITY)
    for key in ("python", "torch", "cuda", "numpy"):
        if key not in runtime:
            die(f"ancestra: sealed RELEASE.json: missing key 'runtime.{key}'", EXIT_INTEGRITY)
        if not isinstance(runtime[key], str) or not runtime[key]:
            die(f"ancestra: sealed RELEASE.json: runtime.{key} is not a non-empty string", EXIT_INTEGRITY)
    version = sealed_doc["version"]
    if version != "1.0.0":
        die(f"ancestra: sealed RELEASE.json: version {version!r} is not '1.0.0'", EXIT_INTEGRITY)
    built = _native.build_info()
    git = built["git_sha"]
    defaults_digest, parity_gpu = sealed_doc["defaults_digest"], sealed_doc["parity_gpu"]
    fields = (("version", version, 0), ("checkpoint.name", ckpt["name"], 0),
              ("checkpoint.sha256", ckpt["sha256"], 64), ("weights_sha256", sealed_doc["weights_sha256"], 64),
              ("defaults_digest", defaults_digest, 64))
    bad = [f"{k} {v!r}" for k, v, hexlen in fields if not _stampable(v, hexlen)]
    if bad:
        die("ancestra: sealed RELEASE.json: " + "; ".join(bad), EXIT_INTEGRITY)
    if not _stampable(git):
        die(f"ancestra: the binding's build sha {git!r} is not printable ASCII", EXIT_INTEGRITY)
    if built["defaults_digest"] != defaults_digest:
        die(f"ancestra: the binding's decoder defaults digest {built['defaults_digest']} != RELEASE.json "
            f"defaults_digest {defaults_digest}: the binding was built from other defaults", EXIT_INTEGRITY)
    if not isinstance(parity_gpu, str) or not parity_gpu:
        die("ancestra: sealed RELEASE.json: parity_gpu is not a non-empty string", EXIT_INTEGRITY)
    if not _stampable(sealed_doc["alt_sha256"], 64):
        die("ancestra: sealed RELEASE.json: alt_sha256 is not 64 lower-case hex", EXIT_INTEGRITY)
    l2 = params.get("l2")
    if (not _stampable(sealed_doc["l2_model_sha256"], 64) or not isinstance(l2, dict) or
            l2.get("model_sha256") != sealed_doc["l2_model_sha256"] or l2.get("wmax") != 3000):
        die("ancestra: sealed L2 model contract is invalid", EXIT_INTEGRITY)
    l2_path = os.path.join(DATA_DIR, "l2_model.json")
    try:
        l2_sha = sha256_file(l2_path)
    except OSError:
        die(f"ancestra: sealed L2 model missing: {l2_path}", EXIT_INTEGRITY)
    if l2_sha != sealed_doc["l2_model_sha256"]:
        die(f"ancestra: sealed L2 model sha256 {l2_sha} != RELEASE.json l2_model_sha256", EXIT_INTEGRITY)
    s0 = (f"ancestra {version} git={git} rs={git},{'clean' if not built['dirty'] else 'dirty'} "
          f"ckpt={ckpt['name']} ckpt_sha256={ckpt['sha256']} weights={sealed_doc['weights_sha256']} params={params_sha}")
    if '"' in s0 or not (s0.isascii() and s0.isprintable()):
        die(f"ancestra: the stamp is not one line of printable ASCII without '\"': {s0!r}", EXIT_INTEGRITY)
    return {"version": version, "checkpoint": ckpt, "defaults_digest": defaults_digest,
            "weights_sha256": sealed_doc["weights_sha256"], "stamp0": s0, "params": params, "params_sha256": params_sha,
            "runtime": runtime, "parity_gpu": parity_gpu, "alt_sha256": sealed_doc["alt_sha256"],
            "l2_model_sha256": sealed_doc["l2_model_sha256"]}


def resolve_weights(arg, facts):
    """--weights: the argument, else $ANCESTRA_WEIGHTS, else the package's data dir. Every run re-hashes it and refuses
    a mismatch with the sealed sha, so a swapped weights file never reaches an output silently."""
    path = arg or os.environ.get("ANCESTRA_WEIGHTS") or os.path.join(DATA_DIR, "ancestra-%s.weights.pt" % facts["version"])
    try:
        sha = sha256_file(path)
    except OSError as e:
        die(f"ancestra: weights {path}: {e.strerror}", EXIT_INTEGRITY)
    if sha != facts["weights_sha256"]:
        die(f"ancestra: weights {path} sha256 {sha} != sealed {facts['weights_sha256']}", EXIT_INTEGRITY)
    return path, sha


# --- sequence names (before any model is loaded) -----------------------------------------------------------------------

def preflight_names(twobit, genome, chroms, formats):
    """Sequence names on every item chrom (every 2bit sequence by default), the GTF rule only when gtf is requested,
    and a 2bit whose index holds a name twice. Every refusal is listed in the ascii() escaping check_names returns,
    and the run exits 2 with nothing created: --allow-missing does not apply, select around a name instead."""
    refused = _native.check_names(genome, chroms, "gtf" in formats)
    for _, reason in refused:
        if " in the index: " in reason:                   # out.rs: the whole index, named after the 2bit
            print(f"{twobit}: {reason}", file=sys.stderr)
        elif "(GTF attribute values" in reason:           # out.rs: only when gtf is requested
            print(f"--format gtf: {reason}", file=sys.stderr)
        else:
            print(reason, file=sys.stderr)
    if refused:
        raise SystemExit(EXIT_BAD_ARGS)


# --- the RAM bound -----------------------------------------------------------------------------------------------------

def worker_ram(l1, l2):
    """Bound: ``P = max(165·L1 + 137·L2, 153·L1 + 137·L2 + D) + H`` — item 1's f16 copies beside item 2's inference,
    or item 1's Viterbi beside it — with D = 9 GB and H = 3 GB, in decimal GB. Unrounded: the total is the sum of the
    unrounded worker bounds, so rounding happens only where a number is printed or compared in a message."""
    return (max(B_E * l1 + B_I * l2, B_F16 * l1 + B_I * l2 + D_BYTES) + H_BYTES) / 1e9


def worker_rams(items, n):
    """Each worker's P_w from its own two largest items after ``lpt_shards`` (a shard runs largest first, so its
    first two; a shard with one item has L2 = 0). Every value stays unrounded; ``lpt_shards`` is torch-free, so the whole
    RAM print runs before any model is loaded."""
    out = []
    for shard in lpt_shards(items, n):
        lens = sorted((e - s for _, _, s, e in shard), reverse=True)[:2]
        # an idle worker (more workers than items) holds no item and decodes no chunk: only its process overhead H
        out.append(worker_ram(lens[0], lens[1] if len(lens) > 1 else 0) if lens else H_BYTES / 1e9)
    return out


def memory_limit():
    """The machine's memory limit: the cgroup's memory.max, else MemTotal (bytes); None when neither reads."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as fh:
                v = fh.read().strip()
        except OSError:
            continue
        if v and v != "max":                              # cgroup v1's PAGE_COUNTER_MAX sentinel means no limit
            n = int(v)
            if 0 < n < 1 << 60:
                return n
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def print_ram(items, n, devs, n_alt=0):
    """What a user needs: every worker's P_w and the total Σ P_w, plus H for the parent when --devices is given. The
    sum is over the unrounded worker bounds and every number is rounded only for printing and for the warning's text
    A total above the machine's limit warns and never refuses: P over-counts the second inference. Under
    --alt a last line adds each worker's ``n_alt`` alt processes to the total."""
    rams = worker_rams(items, n)
    lines = [f"worker {k}: {ram:.1f} GB" for k, ram in enumerate(rams)]
    parent = f" + {H_BYTES / 1e9:.1f} GB for the parent" if devs else ""
    total = sum(rams) + (H_BYTES / 1e9 if devs else 0.0)
    lines.append(f"RAM: {total:.1f} GB in total ({len(rams)} worker{'s' if len(rams) > 1 else ''}{parent})")
    if n_alt:
        per = n_alt * ALT_BYTES / 1e9
        total += len(rams) * per
        lines.append(f"alt: {n_alt} x {ALT_BYTES / 1e9:.2f} GB = {per:.1f} GB per worker, {total:.1f} GB in total with "
                     f"--alt (assumes windows <= {W_ALT / 1e6:.1f} Mb; measured alt VmHWM <= 0.6 GB)")
    print("\n".join(lines))
    limit = memory_limit()
    if limit is not None and total * 1e9 > limit:
        print(f"warning: {total:.1f} GB is above this machine's memory limit {limit / 1e9:.1f} GB: the bound "
              f"over-counts the second inference, so it warns instead of refusing", file=sys.stderr)
    return rams, lines  # the lines are what PREFIX.prov.json records; the warning is not a bound


# --- the CLI ---------------------------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(prog="ancestra", description="Ancestra: one transcript per gene, with UTRs.")
    ap.add_argument("-t", "--twobit", default=None, help="the genome; chrom sizes come from its header (required)")
    ap.add_argument("-o", "--out", default=None, help="outputs PREFIX.*, work dir PREFIX.parts/ (required)")
    ap.add_argument("--seqs", default=None, help="comma list of chrom or chrom:start-end (default: every 2bit sequence)")
    ap.add_argument("--seq-list", default=None, help="file: one spec per line, '#' comments; adds to --seqs")
    ap.add_argument("--allow-missing", action="store_true",
                    help="skip specs absent from the 2bit or out of range instead of exiting with code 3")
    ap.add_argument("-f", "--format", action="append", default=None, metavar="LIST",
                    help="comma list of bed gtf gff bb (default: bed; the BED12 is always written)")
    ap.add_argument("--gtf", action="store_true", help="the same as adding gtf to --format")
    ap.add_argument("--gff", action="store_true", help="the same as adding gff to --format")
    ap.add_argument("--bb", action="store_true", help="the same as adding bb to --format")
    ap.add_argument("--tracks", action="store_true", help="write per-nucleotide tracks")
    ap.add_argument("--track-type", default=None, choices=["bigwig", "bedgraph"],
                    help="exactly one value (default: bigwig); needs --tracks")
    ap.add_argument("--track-feature", action="append", default=None, metavar="LIST",
                    help="comma list of cds exon utr fiveutr threeutr ss donor acceptor codon start stop all "
                         "(default: all); needs --tracks; may be repeated")
    ap.add_argument("--alt", action="store_true",
                    help="also write the alternative-isoform track, PREFIX.alt.* beside the primary outputs")
    ap.add_argument("--alt-threshold", type=float, default=None, metavar="T",
                    help="with --alt: emit an alternative only if its score p >= T, 0 < T <= 1 (default 0.5). "
                         "Changing T on an existing PREFIX re-runs every item: compare thresholds under separate prefixes")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model for full windows (CUDA with the fast kernels only). Each GPU worker "
                         "first compiles: about 2.5 min with 32 compile processes (longer with fewer), about 1 min when its compile cache is warm; inference is then "
                         "about 1.11x faster. Outputs are reproducible between --compile runs (verified on A100 80GB "
                         "PCIe with torch 2.10.0+cu128) and differ from a run without it in about 0.5%% of gene "
                         "models, about half gained and half lost, from bf16 rounding of the model outputs; "
                         "PREFIX.prov.json records it. It "
                         "pays off for a whole genome on one GPU, or on two with a warm compile cache. Each GPU worker "
                         "briefly runs up to min(32, threads / workers) compile processes (about 15 GB of RAM at 32, "
                         "for about 2 minutes), not counted in the RAM estimate. Concurrent --compile runs on one host "
                         "share worker k's cache directory; give each its own TORCHINDUCTOR_CACHE_DIR.")
    ap.add_argument("--devices", default=None, help="comma list of GPU ids: one worker per GPU")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU budget (default: the CPUs this process may use): Rust tracks and finalize, and "
                         "per worker up to min(16, threads // workers - 1) finishing processes (with --alt, its alt processes) and "
                         "max(1, min(8, threads // workers)) float16 conversion threads")
    ap.add_argument("--weights", default=None,
                    help="the weights file (default: $ANCESTRA_WEIGHTS, else the install's data dir); its sha256 must "
                         "equal the sealed one")
    ap.add_argument("-V", "--version", action="store_true", help="print the release stamp and exit")
    ap.add_argument("--worker", type=int, default=0, help=argparse.SUPPRESS)     # the multi-GPU re-exec, hidden
    ap.add_argument("--n-workers", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--no-merge", action="store_true", help=argparse.SUPPRESS)
    return ap


def _worker_cmd(argv, i, n):
    """One worker re-exec: the parent's argv plus ``--worker``, ``--n-workers`` and ``--no-merge``. ``--devices`` stays, so the child
    pins ``cuda:0``; the parent sets ``CUDA_VISIBLE_DEVICES``."""
    drop = {"--worker", "--n-workers", "--no-merge"}
    clean, skip = [], False
    for t in argv:
        if skip:
            skip = False
            continue
        if t.split("=", 1)[0] in drop:
            skip = "=" not in t
            continue
        clean.append(t)
    return [sys.executable, "-P", "-m", "ancestra.cli", *clean, "--worker", str(i), "--n-workers", str(n), "--no-merge"]


def _spawn_workers(argv, n, devs):
    """Start every worker, wait for all of them, and return the first non-zero exit code (0 when all succeed)."""
    procs = []
    for i in range(n):
        env = dict(os.environ)
        if devs:
            env["CUDA_VISIBLE_DEVICES"] = devs[i]
        procs.append(subprocess.Popen(_worker_cmd(argv, i, n), env=env))
    first = 0
    for rc in (p.wait() for p in procs):
        if rc and not first:
            first = rc
    return first


def _data_lines(path):
    """Non-empty lines that are not a stamp or a GFF3 directive. The writer's record count, recomputed."""
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"#") or not line.strip():
                continue
            n += 1
    return n


def _prov_field(done, plan, it):
    """The provenance field that is not this run's, or None. Device and GPU differ per worker, so they are recorded
    and not compared to one value; a resume does not compare RAM bounds."""
    name = _item_name(it)
    item = done.get("item")
    if (not isinstance(item, dict) or item.get("name") != name or item.get("chrom") != it["chrom"]
            or item.get("start") != it["start"] or item.get("end") != it["end"] or item.get("spec") != it["spec"]):
        return "item"
    if done.get("stamp") != plan["stamp"]:
        return "stamp"
    if done.get("recipe") != plan["params"]["recipe"]:
        return "recipe"
    tb = done.get("twobit")
    if (not isinstance(tb, dict) or tb.get("path") != plan["twobit"] or tb.get("sha256") != plan["twobit_sha256"]
            or tb.get("size") != os.path.getsize(plan["twobit"])):
        return "twobit"
    for field, want in (("ckpt_sha256", plan["checkpoint"]["sha256"]), ("params_sha256", plan["params_sha256"]),
                        ("weights_sha256", plan["weights_sha256"])):
        if done.get(field) != want:
            return field
    if done.get("track_features") != list(plan["track_features"]):
        return "track_features"
    if done.get("track_scale") != plan["params"]["tracks"]["scale"]:
        return "track_scale"
    if not isinstance(done.get("device"), str) or not isinstance(done.get("gpu"), str):
        return "device"
    backend = "triton" if done["device"].startswith("cuda") else "ref"
    if done.get("ssd_backend") != backend:
        return "ssd_backend"
    triton_version = done.get("triton_version")
    if (backend == "triton" and (not isinstance(triton_version, str) or not triton_version)
            or backend == "ref" and triton_version is not None):
        return "triton_version"
    rt = done.get("runtime")
    if not isinstance(rt, dict) or any(not isinstance(rt.get(k), str) for k in ("torch", "cuda", "numpy")):
        return "runtime"
    if isinstance(done.get("worker"), bool) or not isinstance(done.get("worker"), int):
        return "worker"
    if not isinstance(done.get("timings"), dict):
        return "timings"
    if done.get("compile") != plan.get("compile"):
        return "compile"
    if isinstance(done.get("peak_rss_kb"), bool) or not isinstance(done.get("peak_rss_kb"), int):
        return "peak_rss_kb"
    return None


def _stale_outputs(prefix, written):
    """Files this tool could have written at PREFIX.* and this run did not. Never deleted: named, hashed, recorded."""
    exts = ["gtf", "gff3", "bb"]
    exts += [f"{feature}.{strand}.{kind}" for feature in FEATURES for strand in ("plus", "minus")
             for kind in ("bw", "bedGraph")]
    exts += ["alt.bed", "alt.gtf", "alt.gff3", "alt.bb"]
    stale = {}
    for ext in exts:
        path = f"{prefix}.{ext}"
        if path in written or not os.path.isfile(path):
            continue
        try:
            digest = sha256_file(path)
        except OSError as e:
            die(f"{path}: {e.strerror}", EXIT_INTEGRITY)
        stale[path] = {"sha256": digest}
    return stale


def _warn_runtime(plan, done):
    expected = plan.get("sealed_runtime")
    if expected and any(d["runtime"]["torch"] != expected["torch"]
                        or d["runtime"]["numpy"] != expected["numpy"]
                        or d["gpu"] != plan["parity_gpu"] for d in done):
        print(f"warning: runtime differs from the release parity set ({plan['parity_gpu']}): on another GPU type "
              f"about 5% of transcripts differ at UTR ends, about 0.15% of CDS keys, and about 10-15% of "
              f"alternative rows; outputs are byte-reproducible on one GPU type", file=sys.stderr)


def finalize(plan):
    """Parent, after every worker exits 0. Parts come only from ``done_records`` — never a glob — and only the
    run's items are read. A mismatch exits 1 naming the item and the field; a stamp that does not re-read exits 1
    naming the file. ``PREFIX.prov.json`` is written last. A file this run could have written and did not is left
    in place, named on stderr and recorded under ``stale_outputs``."""
    t0 = time.perf_counter()
    alt = plan.get("alt")
    parts = plan["prefix"] + ".parts"
    if not os.path.isdir(parts):
        die(f"no parts directory: {parts}", EXIT_INTEGRITY)
    try:
        records = done_records(parts)
    except ValueError as e:
        die(str(e), EXIT_INTEGRITY)
    by_name = {}
    for done in records:
        by_name.setdefault(done["item"]["name"], []).append(done)
    chosen = []
    for it in plan["items"]:
        name = _item_name(it)
        named = by_name.get(name, [])
        if len(named) != 1:
            die(f"{name}: done does not match", EXIT_INTEGRITY)
        field = _prov_field(named[0], plan, it)
        if field:
            die(f"{name}: {field} does not match", EXIT_INTEGRITY)
        fault = _output_fault(named[0], alt is not None)
        if fault:
            path, field = fault
            die(f"{name}: {field} does not match" + (f" {path}" if path else ""), EXIT_INTEGRITY)
        tx = os.path.join(parts, named[0]["key"] + ".tx.tsv")
        if tx not in named[0]["outputs"]:
            die(f"{name}: outputs does not match {tx}", EXIT_INTEGRITY)
        if alt and not _alt_ok(named[0], alt, os.path.join(parts, named[0]["key"] + ".alt.tsv")):
            die(f"{name}: alt does not match", EXIT_INTEGRITY)
        chosen.append(named[0])
    _warn_runtime(plan, chosen)
    try:
        genome = _native.Genome(plan["twobit"])
        sizes = genome.chrom_sizes()
    except (OSError, ValueError) as e:
        die(f"{plan['twobit']}: {e}", EXIT_INTEGRITY)
    tx_parts = [os.path.join(parts, d["key"] + ".tx.tsv") for d in chosen]
    try:
        written = _native.write_outputs(tx_parts, plan["prefix"], plan["stamp"], list(plan["formats"]), sizes)
    except (OSError, ValueError) as e:
        die(str(e), EXIT_INTEGRITY)
    written = [(path, n, plan["stamp"]) for path, n in written]
    if alt:   # the alt parts the chosen .done records name, in their order; every alt file carries the alt stamp
        stamp_alt = plan["stamp"] + " alt=" + alt["sha256"]
        try:
            written += [(path, n, stamp_alt) for path, n in _native.write_alt_outputs(
                [os.path.join(parts, d["key"] + ".alt.tsv") for d in chosen], tx_parts, plan["prefix"], stamp_alt,
                list(plan["formats"]), sizes, alt["threshold"])]
        except (OSError, ValueError) as e:
            die(str(e), EXIT_INTEGRITY)
    outputs = {}
    for path, n, want in written:
        if not path.endswith(".bw"):
            try:
                got = _native.stamp_of(path)
            except (OSError, ValueError) as e:
                die(f"{path}: stamp does not match ({e})", EXIT_INTEGRITY)
            if got != want:
                die(f"{path}: stamp does not match", EXIT_INTEGRITY)
        if path.endswith((".bed", ".gtf", ".gff3", ".bedGraph")):
            if _data_lines(path) != n:
                die(f"{path}: n does not match", EXIT_INTEGRITY)
        outputs[path] = {"sha256": sha256_file(path), "n": n}
    if plan["tracks"] and chosen:
        bases = [os.path.join(parts, d["key"]) for d in chosen]  # the run's items; a stale part is not a base
        try:
            tracks = _native.finalize_tracks(parts, plan["prefix"], sizes, list(plan["track_features"]),
                                               plan["track_type"], plan["stamp"], int(plan["threads"] or 0), bases)
        except (OSError, ValueError) as e:
            die(str(e), EXIT_INTEGRITY)
        for path, meta in tracks.items():
            sha = sha256_file(path)
            if sha != meta["sha256"]:
                die(f"{path}: sha256 does not match", EXIT_INTEGRITY)
            n = int(meta["records"])
            if path.endswith(".bedGraph"):
                try:
                    got = _native.stamp_of(path)
                except (OSError, ValueError) as e:
                    die(f"{path}: stamp does not match ({e})", EXIT_INTEGRITY)
                if got != plan["stamp"]:
                    die(f"{path}: stamp does not match", EXIT_INTEGRITY)
                if _data_lines(path) != n:
                    die(f"{path}: n does not match", EXIT_INTEGRITY)
            elif not path.endswith(".bw"):
                die(f"{path}: outputs does not match", EXIT_INTEGRITY)
            outputs[path] = {"sha256": sha, "n": n, "feature": meta["feature"], "strand": str(meta["strand"])}
    sha = sha256_file(plan["twobit"])
    if sha != plan["twobit_sha256"]:
        die(f"{plan['twobit']}: sha256 does not match", EXIT_INTEGRITY)
    runtime = None
    workers = {}
    timings = {}
    ssd_backend = chosen[0]["ssd_backend"] if chosen else None
    triton_version = chosen[0]["triton_version"] if chosen else None
    for done in chosen:
        name = done["item"]["name"]
        if runtime is None:
            runtime = done["runtime"]
        elif done["runtime"] != runtime:
            die(f"{name}: runtime does not match", EXIT_INTEGRITY)
        if done["ssd_backend"] != ssd_backend:
            die(f"{name}: ssd_backend does not match", EXIT_INTEGRITY)
        if done["triton_version"] != triton_version:
            die(f"{name}: triton_version does not match", EXIT_INTEGRITY)
        w = done["worker"]
        slot = workers.get(w)
        if slot is None:
            workers[w] = {"worker": w, "device": done["device"], "gpu": done["gpu"],
                          "peak_rss_kb": done["peak_rss_kb"]}
        else:
            if slot["device"] != done["device"] or slot["gpu"] != done["gpu"]:
                die(f"{name}: device does not match", EXIT_INTEGRITY)
            if done["peak_rss_kb"] > slot["peak_rss_kb"]:
                slot["peak_rss_kb"] = done["peak_rss_kb"]
        timings[name] = done["timings"]
    stale = _stale_outputs(plan["prefix"], outputs)
    if stale:
        print("warning: stale outputs not written by this run: " + ", ".join(stale), file=sys.stderr)
    prov = {
        "stamp": plan["stamp"],
        "items": [{"name": _item_name(it), "chrom": it["chrom"], "start": it["start"],
                   "end": it["end"], "spec": it["spec"]} for it in plan["items"]],
        "recipe": plan["params"]["recipe"],
        "workers": [workers[k] for k in sorted(workers)],
        "runtime": runtime,
        "ssd_backend": ssd_backend,
        "triton_version": triton_version,
        "twobit": {"path": plan["twobit"], "size": os.path.getsize(plan["twobit"]), "sha256": sha},
        "outputs": outputs,
        "stale_outputs": stale,
        "timings": timings,
        "finalize_s": time.perf_counter() - t0,
        "peak_rss_kb": peak_rss_kb(),
        "ram": list(plan["ram_lines"]),
    }
    if plan.get("compile") is not None:
        prov["compile"] = plan["compile"]
    if alt:   # the counts per strand, summed over the items; the alt processes' peak and the largest window
        counts = {st: Counter() for st in "+-"}
        for done in chosen:
            for st in "+-":
                counts[st].update(done["alt"]["counts"][st])
        prov["alt"] = {"threshold": alt["threshold"], "alt_sha256": alt["sha256"], "workers_per_gpu": alt["n_alt"],
                       "vmhwm_kb": max((d["alt"]["vmhwm_kb"] for d in chosen), default=0),
                       "max_window_bp": max((d["alt"]["max_window_bp"] for d in chosen), default=0),
                       "counts": {st: dict(sorted(c.items())) for st, c in counts.items()}}
    _json_atomic(plan["prefix"] + ".prov.json", prov)


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.version:                                        # no genome, so the install stamp alone
        print(sealed_facts()["stamp0"])
        return
    missing = [flag for flag, value in (("-t/--twobit", a.twobit), ("-o/--out", a.out)) if not value]
    if missing:
        ap.error("the following arguments are required: " + ", ".join(missing))
    if a.out.endswith(os.sep) or not os.path.basename(a.out):
        die("-o/--out must name a file prefix, not a directory")
    if a.track_type is not None and not a.tracks:
        die("--track-type needs --tracks")
    if a.track_feature is not None and not a.tracks:
        die("--track-feature needs --tracks")
    if a.alt_threshold is not None and not a.alt:
        die("--alt-threshold needs --alt")
    threshold = 0.5 if a.alt_threshold is None else a.alt_threshold    # the registered default
    if not 0 < threshold <= 1:
        die(f"--alt-threshold {threshold} is not in (0, 1]")
    if a.threads is not None and a.threads < 1:
        die(f"--threads {a.threads} is not a positive integer")
    formats = parse_list((a.format or []) + [n for n, on in (("gtf", a.gtf), ("gff", a.gff), ("bb", a.bb)) if on],
                         FORMAT_SPEC)
    if "bed" not in formats:                       # the BED12 is always written, whatever else is asked for
        formats.insert(0, "bed")
    features = parse_list(a.track_feature, FEATURE_SPEC)
    devs = parse_devices(a.devices, check_visible=not a.no_merge)
    n_alt = 0
    if a.alt:      # per worker one finishing thread and n_alt alt processes; --n-workers < 1 is refused below
        workers, threads = len(devs) or max(a.n_workers, 1), a.threads or len(os.sched_getaffinity(0))
        if threads < 2 * workers:
            die(f"--alt needs --threads >= 2 per worker: {workers} workers, {threads} threads")
        n_alt = min(16, threads // workers - 1)
    if not os.path.isfile(a.twobit):
        die(f"--twobit must name an existing file: {a.twobit}")
    try:
        genome = _native.Genome(a.twobit)
        sizes = genome.chrom_sizes()
    except (OSError, ValueError) as e:
        die(f"ancestra: {a.twobit}: {e}", EXIT_INTEGRITY)
    specs = read_specs(a)                                    # items
    for spec, flag in specs:                                # a malformed range is a refused spec, never a traceback
        why = range_error(spec, sizes)
        if why:
            die(f"{flag}: {spec}: {why}")
    if specs:
        items, bad = parse_specs([spec for spec, _ in specs], sizes)
        bad_specs = {spec for spec, _ in bad}
        parsed_specs = [(item, spec, flag) for (spec, flag), item in
                        zip((pair for pair in specs if pair[0] not in bad_specs), items)]
    else:
        items, bad = [(chrom, chrom, 0, size) for chrom, size in sorted(sizes.items())], []
        parsed_specs = [(item, item[1], "--seqs") for item in items]
    check_one_item_per_chrom(parsed_specs)
    spec_of = {item[1]: spec for item, spec, _ in parsed_specs}
    for spec, why in bad:
        print(f"{spec}: {why}", file=sys.stderr)
    if bad and not a.allow_missing:
        print(f"{len(bad)} requested sequence(s) absent from {a.twobit} or out of range: refusing "
              f"(--allow-missing skips them)", file=sys.stderr)
        raise SystemExit(EXIT_MISSING)
    if not items:
        die("no sequence left to annotate after --allow-missing", EXIT_MISSING)
    preflight_names(a.twobit, genome, [it[1] for it in items], formats)    # names: the chroms of the run's items
    facts = sealed_facts()
    if a.alt:                                      # without --alt the file is never opened
        from ancestra.decode.alt import model_fault
        alt_path, alt_doc, alt_sha = sealed("alt.json")
        why = model_fault(alt_doc)
        if why:
            die(f"ancestra: sealed alt.json ({alt_path}): {why}", EXIT_INTEGRITY)
        if alt_sha != facts["alt_sha256"]:
            die(f"ancestra: sealed alt.json ({alt_path}) sha256 {alt_sha} != RELEASE.json alt_sha256 "
                f"{facts['alt_sha256']}: refusing --alt", EXIT_INTEGRITY)
        top = alt_doc["model"]["iso_y"][-1]
        if threshold > top and not a.no_merge:
            print(f"warning: --alt-threshold {threshold} is above the model's highest p {top}: no alternative "
                  f"can pass, so the alt track will be empty", file=sys.stderr)
    weights, weights_sha = resolve_weights(a.weights, facts)
    try:
        twobit_sha = sha256_file(a.twobit)
    except OSError as e:
        die(f"ancestra: {a.twobit}: {e.strerror}", EXIT_INTEGRITY)
    stamp = facts["stamp0"] + _native.stamp_genome(a.twobit, twobit_sha)
    n = len(devs) or a.n_workers
    if n < 1:
        die(f"--n-workers {a.n_workers} is not a positive integer")
    _, ram_lines = print_ram(items, n, devs, n_alt)        # RAM, still before any model is loaded
    plan = {
        "twobit": os.path.abspath(a.twobit), "twobit_sha256": twobit_sha, "prefix": a.out,
        "items": [{"chrom": chrom, "start": rs, "end": re, "spec": spec_of.get(chrom, chrom)}
                  for _, chrom, rs, re in items],
        "formats": formats, "tracks": bool(a.tracks), "track_features": features, "track_type": a.track_type or "bigwig",
        "devices": devs, "threads": a.threads, "weights": weights, "weights_sha256": weights_sha, "stamp": stamp,
        "params": facts["params"], "worker": a.worker, "n_workers": a.n_workers, "no_merge": bool(a.no_merge),
        # the worker reads the sealed facts straight out of the plan (never parses the stamp for them)
        "version": facts["version"], "checkpoint": facts["checkpoint"], "params_sha256": facts["params_sha256"],
        "defaults_digest": facts["defaults_digest"], "ram_lines": ram_lines,
        "sealed_runtime": facts["runtime"], "parity_gpu": facts["parity_gpu"],
    }
    if a.alt:
        plan["alt"] = {"threshold": threshold, "sha256": alt_sha, "model": alt_doc, "n_alt": n_alt}
    if a.compile:
        plan["compile"] = "reduce-overhead"
    if a.no_merge:                                          # a worker never finalizes
        run_items(plan)
        return
    if n > 1:
        rc = _spawn_workers(list(argv) if argv is not None else sys.argv[1:], n, devs)
        if rc:
            raise SystemExit(rc)                           # the first failing worker; 3, 4 and 1 keep their meaning
    else:
        # One device runs here. Pin it so _device_of's cuda:0 is that GPU, then put the environment back.
        plan["worker"], plan["n_workers"] = 0, 1
        old_dev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if devs:
            os.environ["CUDA_VISIBLE_DEVICES"] = devs[0]
        try:
            run_items(plan)
        finally:
            if devs:
                if old_dev is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_dev
    finalize(plan)


if __name__ == "__main__":
    main()
