//! Single-transcript CDS decoder over a candidate graph (CDS-only grammar).
//!
//! Max-product over a sparse segment graph, one strand at a time, in transcript
//! orientation. All scores are background-relative (background scores 0), so genes
//! chain without an explicit intergenic state and `Gene::score` is the gain over
//! calling the gene's span background.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strand {
    Plus,
    Minus,
}

pub struct Emissions<'a> {
    pub start: u64,
    /// genomic + strand ASCII, len L
    pub seq: &'a [u8],
    pub strand: Strand,
    /// [3, L] row-major: log p(bg), log p(CDS), log p(CDS intron) for THIS strand
    pub log_a: &'a [f32],
    /// [3, L] row-major: log p(phase)
    pub log_b: &'a [f32],
    /// [4, L] row-major probabilities: donor, acceptor, start, stop
    pub c: &'a [f32],
}

pub struct Params {
    /// candidate thresholds, same order as `c`
    pub c_thresh: [f32; 4],
    /// per-site-type additive score offset, same order as `c`; 0.0 is output-neutral
    pub c_bias: [f32; 4],
    pub w_c: f32,
    pub w_b: f32,
    /// weight of the phase-constant intron B feature; 0.0 until calibrated
    pub w_b_intron: f32,
    pub intron_min: u64,
    pub intron_l0: u64,
    /// per-base log penalty of the geometric tail, <= 0
    pub intron_geom_b: f32,
    pub exon_max: u64,
    pub gene_start_penalty: f32,
    /// CDS exons shorter than this pay `exon_short_penalty`: a prior, not a ban
    pub exon_soft_min: u64,
    pub exon_short_penalty: f32,
    pub max_candidates: usize,
    /// max weak sites (all four types together) one START..STOP gene may use; 0 = off
    pub weak_budget: u32,
    /// `c_thresh_weak[h] <= p < c_thresh[h]` is a weak candidate; >= c_thresh[h] disables h
    pub c_thresh_weak: [f32; 4],
    /// extra score (<= 0) on a weak site, on top of `w_c * ln p + c_bias`
    pub weak_penalty: [f32; 4],
    /// per-type cap on weak candidates, highest p kept; strong ones are never displaced
    pub max_weak_candidates: usize,
    /// additive score per CDS base: > 0 favours longer CDS, e.g. an upstream in-frame ATG; 0.0 is output-neutral
    pub cds_base_bias: f32,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            c_thresh: [0.9; 4], // calibrated release default
            c_bias: [0.0; 4],
            w_c: 8.0,
            w_b: 0.0, // calibrated release default
            w_b_intron: 0.0, // uncalibrated: keeps intron B out of the score
            intron_min: 30,
            intron_l0: 3000,
            intron_geom_b: -1e-4,
            exon_max: 20_000,
            gene_start_penalty: -10.0,
            exon_soft_min: 15,
            exon_short_penalty: -3.0,
            max_candidates: 1 << 20,
            weak_budget: 0,
            c_thresh_weak: [0.9; 4], // = c_thresh: no weak band unless asked for
            weak_penalty: [0.0; 4],
            max_weak_candidates: 1 << 20,
            cds_base_bias: 0.0,
        }
    }
}

impl Params {
    /// Every parameter that shaped a decode, so a caller can record exactly what ran.
    /// `Default` is the only source of defaults; `py::parse_params` starts from it.
    pub fn resolved(&self) -> Vec<(&'static str, f64)> {
        let Params {
            c_thresh,
            c_bias,
            w_c,
            w_b,
            w_b_intron,
            intron_min,
            intron_l0,
            intron_geom_b,
            exon_max,
            gene_start_penalty,
            exon_soft_min,
            exon_short_penalty,
            max_candidates,
            weak_budget,
            c_thresh_weak,
            weak_penalty,
            max_weak_candidates,
            cds_base_bias,
        } = *self;
        vec![
            ("c_thresh.donor", c_thresh[0] as f64),
            ("c_thresh.acceptor", c_thresh[1] as f64),
            ("c_thresh.start", c_thresh[2] as f64),
            ("c_thresh.stop", c_thresh[3] as f64),
            ("c_bias.donor", c_bias[0] as f64),
            ("c_bias.acceptor", c_bias[1] as f64),
            ("c_bias.start", c_bias[2] as f64),
            ("c_bias.stop", c_bias[3] as f64),
            ("w_c", w_c as f64),
            ("w_b", w_b as f64),
            ("w_b_intron", w_b_intron as f64),
            ("intron_min", intron_min as f64),
            ("intron_l0", intron_l0 as f64),
            ("intron_geom_b", intron_geom_b as f64),
            ("exon_max", exon_max as f64),
            ("gene_start_penalty", gene_start_penalty as f64),
            ("exon_soft_min", exon_soft_min as f64),
            ("exon_short_penalty", exon_short_penalty as f64),
            ("max_candidates", max_candidates as f64),
            ("weak_budget", weak_budget as f64),
            ("c_thresh_weak.donor", c_thresh_weak[0] as f64),
            ("c_thresh_weak.acceptor", c_thresh_weak[1] as f64),
            ("c_thresh_weak.start", c_thresh_weak[2] as f64),
            ("c_thresh_weak.stop", c_thresh_weak[3] as f64),
            ("weak_penalty.donor", weak_penalty[0] as f64),
            ("weak_penalty.acceptor", weak_penalty[1] as f64),
            ("weak_penalty.start", weak_penalty[2] as f64),
            ("weak_penalty.stop", weak_penalty[3] as f64),
            ("max_weak_candidates", max_weak_candidates as f64),
            ("cds_base_bias", cds_base_bias as f64),
        ]
    }

    /// Canonical JSON of `Params::default()` under the `default_params` keys (the build stamp's digest):
    /// keys sorted, no spaces (Rust `serde_json::to_string` over a BTreeMap; f32 values as f64, matching
    /// what `default_params()` hands Python). Python treats the digest as opaque (64-hex); float rendering
    /// differs across languages (e.g. `intron_geom_b`), so no cross-language string comparison.
    #[allow(dead_code)]
    pub fn defaults_canonical_json() -> String {
        let d = Params::default();
        // f32 -> f64 casts: matches what `default_params()` hands Python (pyo3 f32 -> float),
        // so this string equals Python's json.dumps(d, sort_keys=True, separators=(',', ':')).
        let f4 = |a: [f32; 4]| a.iter().map(|&v| v as f64).collect::<Vec<f64>>();
        let mut m = std::collections::BTreeMap::new();
        m.insert("c_bias", serde_json::json!(f4(d.c_bias)));
        m.insert("c_thresh", serde_json::json!(f4(d.c_thresh)));
        m.insert("c_thresh_weak", serde_json::json!(f4(d.c_thresh_weak)));
        m.insert("cds_base_bias", serde_json::json!(d.cds_base_bias as f64));
        m.insert("exon_max", serde_json::json!(d.exon_max));
        m.insert("exon_short_penalty", serde_json::json!(d.exon_short_penalty as f64));
        m.insert("exon_soft_min", serde_json::json!(d.exon_soft_min));
        m.insert("gene_start_penalty", serde_json::json!(d.gene_start_penalty as f64));
        m.insert("intron_geom_b", serde_json::json!(d.intron_geom_b as f64));
        m.insert("intron_l0", serde_json::json!(d.intron_l0));
        m.insert("intron_min", serde_json::json!(d.intron_min));
        m.insert("max_candidates", serde_json::json!(d.max_candidates));
        m.insert("max_weak_candidates", serde_json::json!(d.max_weak_candidates));
        m.insert("w_b", serde_json::json!(d.w_b as f64));
        m.insert("w_b_intron", serde_json::json!(d.w_b_intron as f64));
        m.insert("w_c", serde_json::json!(d.w_c as f64));
        m.insert("weak_budget", serde_json::json!(d.weak_budget));
        m.insert("weak_penalty", serde_json::json!(f4(d.weak_penalty)));
        serde_json::to_string(&m).expect("defaults JSON must serialize")
    }

    /// sha256 hex of `defaults_canonical_json()`.
    #[allow(dead_code)]
    pub fn defaults_digest() -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(Params::defaults_canonical_json().as_bytes());
        format!("{:x}", h.finalize())
    }
}

/// genomic half-open
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Exon {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone)]
pub struct Gene {
    pub strand: Strand,
    /// ascending genomic
    pub cds: Vec<Exon>,
    /// path log score minus the background path over its span
    pub score: f32,
    /// weak sites (below `c_thresh`) on the path; always 0 when `weak_budget` is 0
    pub n_weak: u32,
}

// --- semiring -------------------------------------------------------------
// `walk` runs in either semiring: Viterbi (+, max, with `relax` carrying the argmax) or
// sum-product (+, `lse`). The graph is identical; only the accumulation differs.

#[inline]
fn combine(a: f64, b: f64) -> f64 {
    a + b
}

#[inline]
fn choose(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// argmax-carrying `choose`: true when `cand` becomes the new best.
#[inline]
fn relax(best: &mut f64, cand: f64) -> bool {
    if choose(*best, cand) > *best {
        *best = cand;
        true
    } else {
        false
    }
}

/// log(exp(a) + exp(b)), exact at -inf; never below max(a, b), so running sums are monotone.
#[inline]
fn lse(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if lo == NEG {
        return hi;
    }
    hi + (lo - hi).exp().ln_1p()
}

// --- helpers --------------------------------------------------------------

const NONE: u32 = u32::MAX;
const NEG: f64 = f64::NEG_INFINITY;
const BAD: u8 = 255;
/// sum-product only: `Open::prev` of the merged ph 2 acceptor open (states 5..=20)
const ORIG_PH2: u32 = 21;
const BASE: [u8; 4] = [b'A', b'C', b'G', b'T'];

#[inline]
fn comp(b: u8) -> u8 {
    match b {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' => b'A',
        x => x,
    }
}

#[inline]
fn is_stop3(a: u8, b: u8, c: u8) -> bool {
    matches!(
        (a, b, c),
        (b'T', b'A', b'A') | (b'T', b'A', b'G') | (b'T', b'G', b'A')
    )
}

#[inline]
fn b4(x: u8) -> Option<u8> {
    match x {
        b'A' => Some(0),
        b'C' => Some(1),
        b'G' => Some(2),
        b'T' => Some(3),
        _ => None,
    }
}

/// Intron state = the pending codon bases at the donor: p=0 -> 0, p=1 -> 1..=4, p=2 -> 5..=20.
/// A non-ACGT pending base kills the intron edge (an N-tolerant 22nd state would keep it).
fn state_from(b: &[u8]) -> Option<u8> {
    Some(match b.len() {
        0 => 0,
        1 => 1 + b4(b[0])?,
        _ => 5 + 4 * b4(b[0])? + b4(b[1])?,
    })
}

/// Pending bases read off the genome. Only the spliced codon counts, so this is valid
/// only when the exon itself holds them (length >= p); otherwise carry them on `Open`.
fn state_of(seq: &[u8], u: usize, p: usize) -> Option<u8> {
    if u < p {
        return None;
    }
    state_from(&seq[u - p..u])
}

fn state_pending(st: usize) -> ([u8; 2], usize) {
    if st == 0 {
        ([0; 2], 0)
    } else if st < 5 {
        ([BASE[st - 1], 0], 1)
    } else {
        let k = st - 5;
        ([BASE[k / 4], BASE[k % 4]], 2)
    }
}

/// An exon opened at `a` with first-base phase `ph0`, waiting for a donor or a stop.
/// `base` folds the prefix sums at `a` in; `ns` is the first in-frame stop at or after `a`.
/// The exon may only end in `[elo, ehi]`: a start needs all three ATG bases, and a pending
/// base whose genomic junction codon is a stop is legal only as a 1-bp carry exon.
struct Open {
    a: u32,
    ns: u32,
    prev: u32,
    gene_s: u32,
    base: f64,
    /// the pending base when ph0 == 1, else 0: carries a codon across two introns
    pb0: u8,
    elo: u32,
    ehi: u32,
}

/// Splice family of a donor dinucleotide: GT -> 0, GC -> 1, AT -> 2, else `BAD`.
#[inline]
fn donor_fam(a: u8, b: u8) -> u8 {
    match (a, b) {
        (b'G', b'T') => 0,
        (b'G', b'C') => 1,
        (b'A', b'T') => 2,
        _ => BAD,
    }
}

/// Acceptor dinucleotide -> the donor families it may close: AG takes GT and GC, AC takes AT.
#[inline]
fn fam_ok(a: u8, b: u8) -> [bool; 3] {
    match (a, b) {
        (b'A', b'G') => [true, true, false],
        (b'A', b'C') => [false, false, true],
        _ => [false; 3],
    }
}

/// A donor with exons closed on it: one lane per distinct pending state, because the
/// junction-spanning stop test downstream reads those bases. The splice family is
/// fixed by the dinucleotide at `u`, so it is one field rather than a third index here;
/// the intron running maxima, which merge across positions, key on `st * 3 + fam`.
struct Donor {
    u: u32,
    fam: u8,
    e: [f64; 21],
    node: [u32; 21],
}

struct Node {
    a: u32,
    end: u32,
    prev: u32,
    gene_s: u32,
}

pub fn decode(e: &Emissions, p: &Params) -> Vec<Gene> {
    // budget 0 compiles with one constant layer, so the w loops fold away (no slowdown)
    if p.weak_budget == 0 {
        decode_layers::<true>(e, p)
    } else {
        decode_layers::<false>(e, p)
    }
}

fn decode_layers<const ONE: bool>(e: &Emissions, p: &Params) -> Vec<Gene> {
    match walk::<ONE, false>(e, p, 1.0) {
        Some(f) => f.backtrace(e),
        None => Vec::new(),
    }
}

/// The one forward walk over the candidate graph, in either semiring. `SUM = false` is the
/// Viterbi pass (max, with argmax nodes for the backtrace); `SUM = true` is the sum-product pass
/// (logsumexp), which keeps no argmax and instead retains what the FFBS sampler replays. Every candidate, filter and score is shared; only the accumulation differs.
/// `sc` (= 1 / temperature) scales every edge score in the sum-product pass only.
fn walk<const ONE: bool, const SUM: bool>(e: &Emissions, p: &Params, sc: f64) -> Option<Fwd> {
    let l = e.seq.len();
    assert_eq!(e.log_a.len(), 3 * l, "log_a must be [3, L]");
    assert_eq!(e.log_b.len(), 3 * l, "log_b must be [3, L]");
    assert_eq!(e.c.len(), 4 * l, "c must be [4, L]");
    if l < 6 {
        return None;
    }
    let minus = e.strand == Strand::Minus;
    // transcript orientation once, up front; coordinates map back at the end
    let src = |i: usize| if minus { l - 1 - i } else { i };
    let seq: Vec<u8> = (0..l)
        .map(|i| {
            let b = e.seq[src(i)].to_ascii_uppercase();
            if minus {
                comp(b)
            } else {
                b
            }
        })
        .collect();

    let wb = p.w_b as f64;
    let wbi = p.w_b_intron as f64;
    let wc = p.w_c as f64;
    let gb = if SUM { p.intron_geom_b as f64 * sc } else { p.intron_geom_b as f64 };
    let gsp = if SUM { p.gene_start_penalty as f64 * sc } else { p.gene_start_penalty as f64 };
    let imin = (p.intron_min.max(1) as usize).min(l);
    let l0 = (p.intron_l0 as usize).max(imin);
    let emax = (p.exon_max.max(1) as usize).min(l);
    let soft_min = p.exon_soft_min as usize;
    let short_pen = if SUM { p.exon_short_penalty as f64 * sc } else { p.exon_short_penalty as f64 };
    let cbb = p.cds_base_bias as f64;

    // prefix sums in f64: a 200 kb f32 accumulation loses more than a segment score
    let mut p_cds = vec![0f64; l + 1];
    let mut p_int = vec![0f64; l + 1];
    let mut qs = [vec![0f64; l + 1], vec![0f64; l + 1], vec![0f64; l + 1]];
    // rb[p] is the intron B feature: phase-constant over an intron, so it is a plain prefix
    // per phase, not the cycling CDS one. Null-centred on log(1/3) so a chance-level
    // head neither rewards nor penalises intron length.
    // Default w_b_intron=0 needs no intron-phase prefixes: save 24 bytes/base.
    let rb_len = if wbi == 0.0 { 0 } else { l + 1 };
    let mut rb = [vec![0f64; rb_len], vec![0f64; rb_len], vec![0f64; rb_len]];
    let null_b = (1.0f64 / 3.0).ln();
    // unknown (non-ACGT) base counts: a complete CDS may not contain one.
    // Strict reject, so a real gene over an assembly gap decodes as nothing rather
    // than as a flagged model; a sequence_incomplete flag on Gene and a
    // `Params` switch between reject and flag would keep it.
    let mut p_n = vec![0u32; l + 1];
    for i in 0..l {
        let g = src(i);
        let bgv = e.log_a[g] as f64;
        if SUM {
            p_cds[i + 1] = p_cds[i] + sc * ((e.log_a[l + g] as f64 - bgv) + cbb);
            p_int[i + 1] = p_int[i] + sc * (e.log_a[2 * l + g] as f64 - bgv);
        } else {
            p_cds[i + 1] = p_cds[i] + (e.log_a[l + g] as f64 - bgv) + cbb;
            p_int[i + 1] = p_int[i] + (e.log_a[2 * l + g] as f64 - bgv);
        }
        p_n[i + 1] = p_n[i] + b4(seq[i]).is_none() as u32;
        for r in 0..3 {
            if SUM {
                qs[r][i + 1] = qs[r][i] + sc * e.log_b[((i + r) % 3) * l + g] as f64;
                if wbi != 0.0 {
                    rb[r][i + 1] = rb[r][i] + sc * (e.log_b[r * l + g] as f64 - null_b);
                }
                continue;
            }
            qs[r][i + 1] = qs[r][i] + e.log_b[((i + r) % 3) * l + g] as f64;
            if wbi != 0.0 {
                rb[r][i + 1] = rb[r][i] + (e.log_b[r * l + g] as f64 - null_b);
            }
        }
    }
    let intron_phase = |ph: usize, i: usize| {
        if wbi == 0.0 { 0.0 } else { wbi * rb[ph][i] }
    };

    // next_stop[r][i]: next index >= i, == r (mod 3), holding a stop codon; l = none
    let mut nstop = [
        vec![l as u32; l + 1],
        vec![l as u32; l + 1],
        vec![l as u32; l + 1],
    ];
    for r in 0..3 {
        for i in (0..l).rev() {
            nstop[r][i] = if i % 3 == r && i + 3 <= l && is_stop3(seq[i], seq[i + 1], seq[i + 2]) {
                i as u32
            } else {
                nstop[r][i + 1]
            };
        }
    }

    // candidates: sequence-checked motif + head probability over threshold.
    // Start/stop codons must be contiguous in the genome; carrying
    // the partial start/stop across the junction with the same 21 pending-base states would lift that.
    // Keep only retained sites. Dense f64 tracks cost 32 bytes/base regardless of
    // candidate density; ordered lists preserve scores, caps and event order.
    // Weak sites (`c_thresh_weak <= p < c_thresh`) join only when a budget exists; each entry
    // carries its weak count (0/1), which moves a gene's w layer. Budget 0 keeps one layer.
    let kb = if ONE { 0 } else { p.weak_budget.min(u8::MAX as u32 - 1) as usize };
    let nw = kb + 1;
    // (position, score, weak count): 16 bytes, so the strong-only map collects in place
    let mut sites: [Vec<(u32, f64, u8)>; 4] = std::array::from_fn(|_| Vec::new());
    for h in 0..4 {
        let th = p.c_thresh[h];
        let thw = p.c_thresh_weak[h];
        let weak_on = kb > 0 && thw < th;
        let mut v: Vec<(usize, f32)> = Vec::new();
        let mut vw: Vec<(usize, f32)> = Vec::new();
        for i in 0..l {
            let ok = match h {
                0 => {
                    i + 2 <= l
                        && matches!(
                            (seq[i], seq[i + 1]),
                            (b'G', b'T') | (b'G', b'C') | (b'A', b'T')
                        )
                }
                1 => i >= 1 && matches!((seq[i - 1], seq[i]), (b'A', b'G') | (b'A', b'C')),
                2 => i + 3 <= l && seq[i] == b'A' && seq[i + 1] == b'T' && seq[i + 2] == b'G',
                _ => i + 3 <= l && is_stop3(seq[i], seq[i + 1], seq[i + 2]),
            };
            if !ok {
                continue;
            }
            let pr = e.c[h * l + src(i)];
            if pr >= th {
                v.push((i, pr));
            } else if weak_on && pr >= thw {
                vw.push((i, pr));
            }
        }
        if v.len() > p.max_candidates {
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            v.truncate(p.max_candidates);
            v.shrink_to_fit(); // do not retain the discarded candidates' allocation
            v.sort_by_key(|x| x.0);
        }
        let score = |pr: f32| {
            let s = wc * (pr as f64).max(1e-30).ln() + p.c_bias[h] as f64;
            if SUM { s * sc } else { s }
        };
        if vw.is_empty() {
            sites[h] = v.into_iter().map(|(i, pr)| (i as u32, score(pr), 0)).collect();
            continue;
        }
        if vw.len() > p.max_weak_candidates {
            vw.sort_by(|a, b| b.1.total_cmp(&a.1));
            vw.truncate(p.max_weak_candidates);
            vw.sort_by_key(|x| x.0);
        }
        // the two bands are disjoint in p, so a position is in at most one list: merge by position
        let wp = if SUM { p.weak_penalty[h] as f64 * sc } else { p.weak_penalty[h] as f64 };
        let mut out = Vec::with_capacity(v.len() + vw.len());
        let (mut x, mut y) = (0, 0);
        while x < v.len() || y < vw.len() {
            if y == vw.len() || (x < v.len() && v[x].0 < vw[y].0) {
                out.push((v[x].0 as u32, score(v[x].1), 0));
                x += 1;
            } else {
                out.push((vw[y].0 as u32, score(vw[y].1) + wp, 1));
                y += 1;
            }
        }
        sites[h] = out;
    }

    // Every in-gene container is split by w, the weak sites the gene has used so far, so each
    // scan reads one layer only: (K+1)x the state and work of a single-layer decode.
    let mut buckets: Vec<[Vec<Open>; 3]> = (0..nw).map(|_| Default::default()).collect();
    let mut head = vec![[0usize; 3]; nw];
    let mut arena: Vec<Node> = Vec::new();
    let mut donors: Vec<Vec<Donor>> = (0..nw).map(|_| Vec::new()).collect();
    // running maxima over folded donors, keyed [w][st * 3 + splice family]
    let mut m = vec![[NEG; 63]; nw];
    let mut marg = vec![[NONE; 63]; nw];
    let mut folded = vec![0usize; nw];
    let mut bg = vec![NEG; l + 1];
    let mut bg_src = vec![NONE; l + 1];
    let mut terms: Vec<(u32, f64, u8)> = Vec::new();
    bg[0] = 0.0;
    // sum-product only: the fold mass after each folded donor (FFBS bisects on it to pick a
    // folded donor), donor indices per splice family, and every gene-end term (end, log mass)
    let mut cum: Vec<Vec<[f64; 21]>> = (0..nw).map(|_| Vec::new()).collect();
    let mut fam_idx: Vec<[Vec<u32>; 3]> = (0..nw).map(|_| Default::default()).collect();
    let mut ends: Vec<(u32, f64)> = Vec::new();

    // open an exon at `a` with first-base phase `ph0` and incoming path score `s`.
    // bucket index rq = (ph0 - a) mod 3 is the phase-prefix offset AND fixes the
    // pending-base count at any later position i to (i + rq) mod 3.
    macro_rules! open {
        ($a:expr, $ph0:expr, $s:expr, $prev:expr, $gs:expr, $pb0:expr, $elo:expr, $ehi:expr, $w:expr) => {{
            let a: usize = $a;
            let ph0: usize = $ph0;
            let rq = (3 + ph0 - a % 3) % 3;
            let rc = (3 + a % 3 - ph0) % 3;
            buckets[$w][rq].push(Open {
                a: a as u32,
                ns: nstop[rc][a],
                prev: $prev,
                gene_s: $gs,
                base: $s - p_cds[a] - wb * qs[rq][a],
                pb0: $pb0,
                elo: $elo as u32,
                ehi: $ehi as u32,
            });
        }};
    }

    let mut site_pos = [0usize; 4];
    let mut next_site = sites.iter().filter_map(|v| v.first().map(|s| s.0 as usize)).min().unwrap_or(usize::MAX);
    for i in 0..l {
        if SUM {
            if i > 0 && bg[i - 1] > NEG {
                bg[i] = lse(bg[i], bg[i - 1]);
            }
        } else if i > 0 && bg[i - 1] > bg[i] {
            bg[i] = bg[i - 1];
            bg_src[i] = NONE;
        }
        // Only background propagation visits every base. Exon-window pruning and lagged
        // donor folding are monotone catch-ups keyed on i that only candidate events read,
        // so running them here folds the same donors in the same order as a per-base pass.
        if i != next_site {
            continue;
        }
        for w in 0..nw {
            let (buckets, head) = (&buckets[w], &mut head[w]);
            for r in 0..3 {
                while head[r] < buckets[r].len() && (buckets[r][head[r]].a as usize) + emax < i {
                    head[r] += 1;
                }
            }
        }
        // donors past the exact-enumeration window fold into the running maxima
        for w in 0..nw {
            let (donors, folded, m, marg) = (&donors[w], &mut folded[w], &mut m[w], &mut marg[w]);
            while *folded < donors.len() && (donors[*folded].u as usize) + l0 <= i {
                let f = donors[*folded].fam as usize;
                for st in 0..21 {
                    if donors[*folded].e[st] == NEG {
                        continue;
                    }
                    let v = combine(donors[*folded].e[st], -gb * donors[*folded].u as f64);
                    if SUM {
                        m[st * 3 + f] = lse(m[st * 3 + f], v);
                    } else if relax(&mut m[st * 3 + f], v) {
                        marg[st * 3 + f] = donors[*folded].node[st];
                    }
                }
                if SUM {
                    cum[w].push(std::array::from_fn(|st| m[st * 3 + f]));
                }
                *folded += 1;
            }
        }
        // cw[h]: weak sites this candidate spends (0 or 1)
        let mut cw = [0usize; 4];
        let cand: [f64; 4] = std::array::from_fn(|h| {
            match sites[h].get(site_pos[h]) {
                Some(&(pos, score, wk)) if pos as usize == i => {
                    site_pos[h] += 1;
                    cw[h] = if ONE { 0 } else { wk as usize };
                    score
                }
                _ => NEG,
            }
        });
        next_site = (0..4).filter_map(|h| sites[h].get(site_pos[h]).map(|s| s.0 as usize))
            .min().unwrap_or(usize::MAX);
        // gene start: exon opens at i with phase 0
        if cand[2] > NEG {
            open!(i, 0, bg[i] + gsp + cand[2], NONE, i as u32, 0, i + 3, l, cw[2]);
        }
        // acceptor: close an intron, open the next exon at i+1
        if cand[1] > NEG && i + 1 < l {
            let j = i;
            let a = j + 1;
            // junction codon = pending bases + the first `need` bases of the new exon. Only
            // settled here when the exon is long enough to hold them: a ph=1 pair may instead
            // be completed after a second intron, so its test moves to the close (`ehi`).
            let mut ok = [false; 21];
            ok[0] = true;
            for st in 1..21 {
                let (pb, ph) = state_pending(st);
                let need = 3 - ph;
                if a + need > l {
                    continue;
                }
                let mut cod = [0u8; 3];
                cod[..ph].copy_from_slice(&pb[..ph]);
                for k in 0..need {
                    cod[ph + k] = seq[a + k];
                }
                ok[st] = !is_stop3(cod[0], cod[1], cod[2]);
            }
            // only the intended donor families may close on this acceptor
            let famok = fam_ok(seq[j - 1], seq[j]);
            let lag = gb * (a as f64 - l0 as f64);
            let hi = a.saturating_sub(imin);
            // the intron B feature is constant at the intron's own phase, which the pending
            // state fixes, so each phase branch closes with its own tail
            let tail = |ph: usize| p_int[a] + intron_phase(ph, a) + cand[1];
            // one pass per incoming layer w; a weak acceptor lifts it to w + 1, so the top
            // layer cannot take one. Stack scratch per pass keeps the scans register-resident.
            for w in 0..nw - cw[1] {
                let wo = w + cw[1];
                // best incoming path per pending state: folded maxima, then the exact range
                let mut bs = [NEG; 21];
                let mut bsrc = [NONE; 21];
                for st in 0..21 {
                    for f in 0..3 {
                        if !famok[f] || m[w][st * 3 + f] == NEG {
                            continue;
                        }
                        if SUM {
                            bs[st] = lse(bs[st], combine(m[w][st * 3 + f], lag));
                        } else if relax(&mut bs[st], combine(m[w][st * 3 + f], lag)) {
                            bsrc[st] = marg[w][st * 3 + f];
                        }
                    }
                }
                // exact range: donors not yet folded, down to the minimum intron length
                let donors = &donors[w];
                let mut k = folded[w];
                while k < donors.len() && (donors[k].u as usize) <= hi {
                    if famok[donors[k].fam as usize] {
                        for st in 0..21 {
                            if SUM {
                                if donors[k].e[st] > NEG {
                                    bs[st] = lse(bs[st], donors[k].e[st]);
                                }
                            } else if donors[k].e[st] > NEG && relax(&mut bs[st], donors[k].e[st]) {
                                bsrc[st] = donors[k].node[st];
                            }
                        }
                    }
                    k += 1;
                }
                if SUM {
                    // `prev` records the open's origin for FFBS: NONE is a
                    // gene start, else the pending state closed here (0, 1..=4, or ORIG_PH2 for
                    // the merged ph 2 lanes)
                    if bs[0] > NEG {
                        open!(a, 0, combine(bs[0], tail(0)), 0, NONE, 0, a + 1, l, wo);
                    }
                    for st in 1..5 {
                        if bs[st] > NEG {
                            let ehi = if ok[st] { l } else { a + 1 };
                            open!(a, 1, combine(bs[st], tail(1)), st as u32, NONE, BASE[st - 1], a + 1, ehi, wo);
                        }
                    }
                    let mut b2 = NEG;
                    for st in 5..21 {
                        if ok[st] && bs[st] > NEG {
                            b2 = lse(b2, bs[st]);
                        }
                    }
                    if b2 > NEG {
                        open!(a, 2, combine(b2, tail(2)), ORIG_PH2, NONE, 0, a + 1, l, wo);
                    }
                    continue;
                }
                let gs = |nd: u32| arena[nd as usize].gene_s;
                // ph 0: nothing pending, the exon opens a codon
                if bsrc[0] != NONE {
                    open!(a, 0, combine(bs[0], tail(0)), bsrc[0], gs(bsrc[0]), 0, a + 1, l, wo);
                }
                // ph 1: the pending base rides on the exon, so the four states stay apart. A 1-bp
                // exon carries it to the next intron; longer ones complete the codon here.
                for st in 1..5 {
                    if bsrc[st] == NONE {
                        continue;
                    }
                    let ehi = if ok[st] { l } else { a + 1 };
                    open!(a, 1, combine(bs[st], tail(1)), bsrc[st], gs(bsrc[st]), BASE[st - 1], a + 1, ehi, wo);
                }
                // ph 2: the first exon base completes the codon, the states merge
                let (mut b2, mut s2) = (NEG, NONE);
                for st in 5..21 {
                    if ok[st] && bsrc[st] != NONE && relax(&mut b2, bs[st]) {
                        s2 = bsrc[st];
                    }
                }
                if s2 != NONE {
                    open!(a, 2, combine(b2, tail(2)), s2, gs(s2), 0, a + 1, l, wo);
                }
            }
        }
        // donor: close the open exons, one winner per distinct pending state. Histories
        // with different pending bases may not compete: the stop test that separates them is
        // only applied at the next acceptor.
        if cand[0] > NEG {
            let fam = donor_fam(seq[i], seq[i + 1]);
            let pend = [
                state_of(&seq, i, 0),
                state_of(&seq, i, 1),
                state_of(&seq, i, 2),
            ];
            // one record per output layer wd, fed by the exons of layer wd - cw[0] (a weak
            // donor lifts the layer by one); the top layer cannot take a weak donor
            for wd in cw[0]..nw {
                let wi = wd - cw[0];
                let mut rec = Donor {
                    u: i as u32,
                    fam,
                    e: [NEG; 21],
                    node: [NONE; 21],
                };
                let mut any = false;
                for rq in 0..3 {
                    let ph = (i + rq) % 3;
                    let t = p_cds[i] + wb * qs[rq][i];
                    let mut best = [NEG; 21];
                    let mut bo = [usize::MAX; 21];
                    let b = &buckets[wi][rq];
                    for k in head[wi][rq]..b.len() {
                        let o = &b[k];
                        if o.a as usize >= i {
                            break;
                        }
                        if i < o.elo as usize || i > o.ehi as usize {
                            continue; // start codon not whole, or junction codon a stop
                        }
                        if (o.ns as usize) + 3 <= i {
                            continue; // in-frame stop wholly inside the exon
                        }
                        if p_n[i] != p_n[o.a as usize] {
                            continue; // unknown base inside the CDS exon
                        }
                        let len = i - o.a as usize;
                        // an exon shorter than its pending count (ph 2, 1 bp) carries the codon
                        // across two introns: the pair is [carried base, this exon's base]
                        let st = if len < ph {
                            state_from(&[o.pb0, seq[o.a as usize]])
                        } else {
                            pend[ph]
                        };
                        let Some(st) = st else { continue };
                        let v = combine(o.base, t) + if len < soft_min { short_pen } else { 0.0 };
                        if SUM {
                            best[st as usize] = lse(best[st as usize], v);
                        } else if relax(&mut best[st as usize], v) {
                            bo[st as usize] = k;
                        }
                    }
                    // rq fixes ph, and ph partitions the 21 lanes, so passes never collide
                    for st in 0..21 {
                        if SUM {
                            if best[st] > NEG {
                                rec.e[st] = best[st] - p_int[i] - intron_phase(ph, i) + cand[0];
                                any = true;
                            }
                            continue;
                        }
                        if bo[st] == usize::MAX {
                            continue;
                        }
                        let o = &buckets[wi][rq][bo[st]];
                        rec.e[st] = best[st] - p_int[i] - intron_phase(ph, i) + cand[0];
                        rec.node[st] = arena.len() as u32;
                        arena.push(Node {
                            a: o.a,
                            end: i as u32,
                            prev: o.prev,
                            gene_s: o.gene_s,
                        });
                        any = true;
                    }
                }
                if any && fam != BAD {
                    if SUM {
                        fam_idx[wd][fam as usize].push(donors[wd].len() as u32);
                    }
                    donors[wd].push(rec);
                }
            }
        }
        // stop: close the CDS, stop codon included, back to background at i+3
        if cand[3] > NEG && i + 3 <= l {
            let rq = (3 - i % 3) % 3;
            let end = i + 3;
            let t = p_cds[end] + wb * qs[rq][end];
            // a gene may end from any layer with room for the stop; the best over w competes
            // (ties keep the lower layer: fewer weak sites)
            let mut best = NEG;
            let mut bo = (0, usize::MAX);
            for w in 0..nw - cw[3] {
                let b = &buckets[w][rq];
                for k in head[w][rq]..b.len() {
                    let o = &b[k];
                    if o.a as usize > i {
                        break;
                    }
                    let len = end - o.a as usize;
                    if o.ns as usize != i || len > emax {
                        continue; // this stop must be the first in-frame one
                    }
                    if end < o.elo as usize || end > o.ehi as usize {
                        continue; // start codon not whole, or junction codon a stop
                    }
                    if p_n[end] != p_n[o.a as usize] {
                        continue; // unknown base inside the CDS exon
                    }
                    let v = combine(o.base, t) + if len < soft_min { short_pen } else { 0.0 };
                    if SUM {
                        best = lse(best, v);
                    } else if relax(&mut best, v) {
                        bo = (w, k);
                    }
                }
            }
            if SUM {
                if best > NEG {
                    let total = combine(best, cand[3]);
                    bg[end] = lse(bg[end], total);
                    ends.push((end as u32, total));
                }
            } else if bo.1 != usize::MAX {
                let total = combine(best, cand[3]);
                if total > bg[end] {
                    let o = &buckets[bo.0][rq][bo.1];
                    bg[end] = total;
                    bg_src[end] = terms.len() as u32;
                    terms.push((arena.len() as u32, total, (bo.0 + cw[3]) as u8));
                    arena.push(Node {
                        a: o.a,
                        end: end as u32,
                        prev: o.prev,
                        gene_s: o.gene_s,
                    });
                }
            }
        }
    }
    if SUM {
        bg[l] = lse(bg[l], bg[l - 1]);
    } else if bg[l - 1] > bg[l] {
        bg[l] = bg[l - 1];
        bg_src[l] = NONE;
    }
    Some(Fwd {
        l,
        minus,
        nw,
        emax,
        imin,
        l0,
        soft_min,
        short_pen,
        wb,
        wbi,
        gb,
        gsp,
        seq,
        p_cds,
        p_int,
        qs,
        rb,
        p_n,
        sites,
        buckets,
        donors,
        bg,
        bg_src,
        terms,
        arena,
        cum,
        fam_idx,
        ends,
    })
}

impl Fwd {
    /// Viterbi backtrace: every gene on the best background path.
    fn backtrace(self, e: &Emissions) -> Vec<Gene> {
        let Fwd { l, minus, bg, bg_src, terms, arena, .. } = self;
        let mut out: Vec<Gene> = Vec::new();
        let mut q = l;
        while q > 0 {
            let s = bg_src[q];
            if s == NONE {
                q -= 1;
                continue;
            }
            let (node, total, n_weak) = terms[s as usize];
            let mut ex: Vec<(u64, u64)> = Vec::new();
            let mut n = node;
            let mut gs = 0usize;
            while n != NONE {
                let nd = &arena[n as usize];
                ex.push((nd.a as u64, nd.end as u64));
                gs = nd.gene_s as usize;
                n = nd.prev;
            }
            let mut cds: Vec<Exon> = ex
                .iter()
                .rev()
                .map(|&(a, b)| {
                    if minus {
                        Exon {
                            start: e.start + (l as u64 - b),
                            end: e.start + (l as u64 - a),
                        }
                    } else {
                        Exon {
                            start: e.start + a,
                            end: e.start + b,
                        }
                    }
                })
                .collect();
            if minus {
                cds.reverse();
            }
            out.push(Gene {
                strand: e.strand,
                cds,
                score: (total - bg[gs]) as f32,
                n_weak: n_weak as u32,
            });
            q = gs;
        }
        // traced back to front: plus needs reversing, minus is already ascending genomic
        if !minus {
            out.reverse();
        }
        out
    }
}

// --- sum-product: FFBS sampler, MBR -----------------------------------------------------------

/// What one forward walk leaves behind, in transcript orientation. The Viterbi pass reads
/// `bg_src`, `terms` and `arena`; the sum-product pass reads the retained opens (`base` is their
/// forward mass minus the prefix at `a`), the donor records (`e` is their forward mass), the fold
/// masses and the gene-end terms. The sampler replays the forward scans
/// through `scan_donor`, `scan_stop` and `acc`, which apply the forward's filters verbatim.
struct Fwd {
    l: usize,
    minus: bool,
    nw: usize,
    emax: usize,
    imin: usize,
    l0: usize,
    soft_min: usize,
    short_pen: f64,
    wb: f64,
    wbi: f64,
    gb: f64,
    gsp: f64,
    seq: Vec<u8>,
    p_cds: Vec<f64>,
    p_int: Vec<f64>,
    qs: [Vec<f64>; 3],
    rb: [Vec<f64>; 3],
    p_n: Vec<u32>,
    sites: [Vec<(u32, f64, u8)>; 4],
    buckets: Vec<[Vec<Open>; 3]>,
    donors: Vec<Vec<Donor>>,
    bg: Vec<f64>,
    bg_src: Vec<u32>,
    terms: Vec<(u32, f64, u8)>,
    arena: Vec<Node>,
    cum: Vec<Vec<[f64; 21]>>,
    fam_idx: Vec<[Vec<u32>; 3]>,
    ends: Vec<(u32, f64)>,
}

/// Pending-base count of a lane state: 0 -> 0, 1..=4 -> 1, 5..=20 -> 2.
#[inline]
fn ph_of(st: usize) -> usize {
    match st {
        0 => 0,
        1..=4 => 1,
        _ => 2,
    }
}


/// An acceptor candidate at `j`, as the forward pass sees it when it opens the exon at `a = j + 1`.
struct Acc {
    a: usize,
    /// weak sites the acceptor spends (0 or 1)
    cw: usize,
    famok: [bool; 3],
    /// junction codon not a stop, per pending state
    ok: [bool; 21],
    lag: f64,
    hi: usize,
    /// intron close score per pending count
    tail: [f64; 3],
}

impl Fwd {
    fn site(&self, h: usize, pos: usize) -> Option<(f64, usize)> {
        let v = &self.sites[h];
        let k = v.partition_point(|x| (x.0 as usize) < pos);
        match v.get(k) {
            Some(&(q, s, wk)) if q as usize == pos => Some((s, wk as usize)),
            _ => None,
        }
    }

    /// First open of a bucket still inside the exon window at `i` (the forward's `head`).
    fn head(&self, w: usize, rq: usize, i: usize) -> usize {
        self.buckets[w][rq].partition_point(|o| (o.a as usize) + self.emax < i)
    }

    fn iph(&self, ph: usize, i: usize) -> f64 {
        if self.wbi == 0.0 {
            0.0
        } else {
            self.wbi * self.rb[ph][i]
        }
    }


    /// Opens a donor at `i` closes from layer `wi`, lane offset `rq`: `f(open, state, edge)`, where
    /// the open's mass plus `edge` is the exon's closed mass before the donor terms.
    fn scan_donor(&self, wi: usize, rq: usize, i: usize, mut f: impl FnMut(usize, usize, f64)) {
        let ph = (i + rq) % 3;
        let t = self.p_cds[i] + self.wb * self.qs[rq][i];
        let pend = state_of(&self.seq, i, ph);
        let b = &self.buckets[wi][rq];
        for k in self.head(wi, rq, i)..b.len() {
            let o = &b[k];
            let a = o.a as usize;
            if a >= i {
                break;
            }
            if i < o.elo as usize || i > o.ehi as usize {
                continue;
            }
            if (o.ns as usize) + 3 <= i {
                continue;
            }
            if self.p_n[i] != self.p_n[a] {
                continue;
            }
            let len = i - a;
            let st = if len < ph { state_from(&[o.pb0, self.seq[a]]) } else { pend };
            let Some(st) = st else { continue };
            f(k, st as usize, t + if len < self.soft_min { self.short_pen } else { 0.0 });
        }
    }

    /// Opens a stop at `i` closes from layer `w`: `f(open, edge)`, edge before the stop score.
    fn scan_stop(&self, w: usize, i: usize, mut f: impl FnMut(usize, f64)) {
        let rq = (3 - i % 3) % 3;
        let end = i + 3;
        let t = self.p_cds[end] + self.wb * self.qs[rq][end];
        let b = &self.buckets[w][rq];
        for k in self.head(w, rq, i)..b.len() {
            let o = &b[k];
            let a = o.a as usize;
            if a > i {
                break;
            }
            let len = end - a;
            if o.ns as usize != i || len > self.emax {
                continue;
            }
            if end < o.elo as usize || end > o.ehi as usize {
                continue;
            }
            if self.p_n[end] != self.p_n[a] {
                continue;
            }
            f(k, t + if len < self.soft_min { self.short_pen } else { 0.0 });
        }
    }

    fn acc(&self, j: usize) -> Option<Acc> {
        let (c1, cw) = self.site(1, j)?;
        let a = j + 1;
        if a >= self.l {
            return None;
        }
        let seq = &self.seq;
        let mut ok = [false; 21];
        ok[0] = true;
        for st in 1..21 {
            let (pb, ph) = state_pending(st);
            let need = 3 - ph;
            if a + need > self.l {
                continue;
            }
            let mut cod = [0u8; 3];
            cod[..ph].copy_from_slice(&pb[..ph]);
            for k in 0..need {
                cod[ph + k] = seq[a + k];
            }
            ok[st] = !is_stop3(cod[0], cod[1], cod[2]);
        }
        Some(Acc {
            a,
            cw,
            famok: fam_ok(seq[j - 1], seq[j]),
            ok,
            lag: self.gb * (a as f64 - self.l0 as f64),
            // Donors at u >= a - 1 are never connected by the forward pass: the donor at u
            // is added after the acceptor in the same iteration, so with intron_min <= 1 the
            // raw bound a - imin would admit u = j = a - 1 here while the forward mass has
            // no such edge. Floor at 2 to replay exactly the forward's graph (Viterbi and
            // the forward pass read their own bound and are untouched).
            hi: a.saturating_sub(self.imin.max(2)),
            tail: std::array::from_fn(|ph| self.p_int[a] + self.iph(ph, a) + c1),
        })
    }

    /// Donors of layer `w` already folded when the acceptor at `j` is processed.
    fn nfold(&self, w: usize, j: usize) -> usize {
        self.donors[w].partition_point(|d| d.u as usize + self.l0 <= j)
    }

    /// Fold mass of family `f`, state `st` over the first `cnt` donors of layer `w`, and how many
    /// family-`f` donors that prefix holds.
    fn fold_mass(&self, w: usize, f: usize, st: usize, cnt: usize) -> (f64, usize) {
        let v = &self.fam_idx[w][f];
        let n = v.partition_point(|&d| (d as usize) < cnt);
        if n == 0 {
            (NEG, 0)
        } else {
            (self.cum[w][v[n - 1] as usize][st], n)
        }
    }



    fn span(&self, start: u64, a: usize, b: usize) -> (u64, u64) {
        if self.minus {
            (start + (self.l - b) as u64, start + (self.l - a) as u64)
        } else {
            (start + a as u64, start + b as u64)
        }
    }


    /// One whole-region path by backward sampling: genes in transcript order, each its exons
    /// (transcript coordinates, ascending).
    fn sample(&self, rng: &mut Rng) -> Vec<Vec<(usize, usize)>> {
        let mut genes = Vec::new();
        let mut buf: Vec<(f64, (usize, usize))> = Vec::new();
        let mut cands: Vec<(f64, (usize, usize, bool))> = Vec::new();
        let mut q = self.l;
        let mut idx = self.ends.len();
        loop {
            while idx > 0 && self.ends[idx - 1].0 as usize > q {
                idx -= 1;
            }
            if idx == 0 {
                break;
            }
            idx -= 1;
            // at background `end`, a gene ends here with probability term / bg[end]
            let (end, term) = self.ends[idx];
            let end = end as usize;
            if rng.f64() >= (term - self.bg[end]).exp() {
                continue;
            }
            let g = self.sample_gene(end, rng, &mut buf, &mut cands);
            q = g[0].0;
            genes.push(g);
        }
        genes.reverse();
        genes
    }

    fn sample_gene(
        &self,
        end: usize,
        rng: &mut Rng,
        buf: &mut Vec<(f64, (usize, usize))>,
        cands: &mut Vec<(f64, (usize, usize, bool))>,
    ) -> Vec<(usize, usize)> {
        let i = end - 3;
        let (_, cw3) = self.site(3, i).expect("gene end without a stop candidate");
        let mut rq = (3 - i % 3) % 3;
        buf.clear();
        for w in 0..self.nw - cw3 {
            self.scan_stop(w, i, |k, edge| buf.push((self.buckets[w][rq][k].base + edge, (w, k))));
        }
        let (mut w, mut k) = pick(buf, rng).expect("gene end without an open exon");
        let mut ex = vec![(self.buckets[w][rq][k].a as usize, end)];
        loop {
            let o = &self.buckets[w][rq][k];
            if o.prev == NONE {
                break; // gene start
            }
            let a = o.a as usize;
            let j = a - 1;
            let acc = self.acc(j).expect("acceptor open without an acceptor candidate");
            let win = w - acc.cw;
            let cnt = self.nfold(win, j);
            let states: Vec<usize> = match (rq + a) % 3 {
                0 => vec![0],
                1 => vec![o.prev as usize],
                _ => (5..21).filter(|&st| acc.ok[st]).collect(),
            };
            cands.clear();
            let dl = &self.donors[win];
            for &st in &states {
                for f in 0..3 {
                    if acc.famok[f] {
                        let (m, _) = self.fold_mass(win, f, st, cnt);
                        if m > NEG {
                            cands.push((m + acc.lag, (st, f, true)));
                        }
                    }
                }
                let mut kk = cnt;
                while kk < dl.len() && (dl[kk].u as usize) <= acc.hi {
                    if acc.famok[dl[kk].fam as usize] && dl[kk].e[st] > NEG {
                        cands.push((dl[kk].e[st], (st, kk, false)));
                    }
                    kk += 1;
                }
            }
            let (st, x, folded) = pick(cands, rng).expect("acceptor open without a donor");
            let d = if folded {
                // bisect the fold's running mass: donor t is drawn with its own share of it
                let v = &self.fam_idx[win][x];
                let n = v.partition_point(|&d| (d as usize) < cnt);
                let target = self.cum[win][v[n - 1] as usize][st] + rng.f64().ln();
                let t = v[..n].partition_point(|&d| self.cum[win][d as usize][st] < target);
                v[t.min(n - 1)] as usize
            } else {
                x
            };
            let u = dl[d].u as usize;
            let (_, cw0) = self.site(0, u).expect("donor record without a donor candidate");
            let wi = win - cw0;
            let rq2 = (ph_of(st) + 3 - u % 3) % 3;
            buf.clear();
            self.scan_donor(wi, rq2, u, |kk, s2, edge| {
                if s2 == st {
                    buf.push((self.buckets[wi][rq2][kk].base + edge, (wi, kk)));
                }
            });
            let (_, k2) = pick(buf, rng).expect("donor state without an open exon");
            w = wi;
            rq = rq2;
            k = k2;
            ex.push((self.buckets[w][rq][k].a as usize, u));
        }
        ex.reverse();
        ex
    }

    /// Transcript exons -> ascending genomic exons.
    fn genomic(&self, start: u64, ex: &[(usize, usize)]) -> Vec<Exon> {
        let mut v: Vec<Exon> = ex
            .iter()
            .map(|&(a, b)| {
                let (s, e) = self.span(start, a, b);
                Exon { start: s, end: e }
            })
            .collect();
        v.sort_by_key(|x| x.start);
        v
    }
}

/// Deterministic xoshiro256** (seeded through splitmix64); no dependency, stable across builds.
pub(crate) struct Rng([u64; 4]);

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut sm = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        };
        Rng([sm(), sm(), sm(), sm()])
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// uniform on the open interval (0, 1)
    pub(crate) fn f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }
}

/// Draw one payload with probability proportional to exp(log weight).
fn pick<T: Copy>(c: &[(f64, T)], rng: &mut Rng) -> Option<T> {
    let mx = c.iter().map(|x| x.0).fold(NEG, f64::max);
    if mx == NEG {
        return None;
    }
    let tot: f64 = c.iter().map(|x| (x.0 - mx).exp()).sum();
    let mut r = rng.f64() * tot;
    for &(w, t) in c {
        let p = (w - mx).exp();
        if r < p {
            return Some(t);
        }
        r -= p;
    }
    c.iter().rev().find(|x| x.0 > NEG).map(|x| x.1) // rounding at the top end
}


/// Floor on the sum-product temperature: below this the 1/T scaling overflows f64
/// (posteriors of inf at T = 1e-300 on a 30-kb region) instead of concentrating.
pub const MIN_TEMPERATURE: f64 = 1e-3;

fn check_temperature(temperature: f64) -> Result<f64, String> {
    if !temperature.is_finite() || temperature < MIN_TEMPERATURE {
        return Err(format!("temperature must be finite and >= {MIN_TEMPERATURE}, got {temperature}"));
    }
    Ok(1.0 / temperature)
}


#[derive(Debug, Clone, Copy)]
pub struct MbrOpts {
    /// FFBS paths drawn per region
    pub n_samples: usize,
    /// divides every edge score before the sum-product (1 = the model as is)
    pub temperature: f64,
    /// a locus's mode chain is primary when its posterior is >= tau
    pub tau: f64,
    pub seed: u64,
}

impl Default for MbrOpts {
    fn default() -> MbrOpts {
        MbrOpts { n_samples: 200, temperature: 1.0, tau: 0.5, seed: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct MbrChain {
    /// ascending genomic CDS blocks
    pub cds: Vec<Exon>,
    /// sample frequency
    pub posterior: f64,
    /// the chain's own path score at T = 1, in the same sense as `Gene::score`
    /// (the gene-path log terms over the chain's span: start, exons, introns, stop)
    pub score: f32,
    /// weak sites on the achieving path, like `Gene::n_weak`
    pub n_weak: u32,
}

/// A same-strand CDS-overlap component of the sampled chains.
#[derive(Debug, Clone)]
pub struct MbrLocus {
    pub strand: Strand,
    /// genomic span of every chain in the locus
    pub start: u64,
    pub end: u64,
    /// every sampled chain, by posterior (descending; ties by coordinates): `chains[0]` is the mode
    pub chains: Vec<MbrChain>,
    /// fraction of samples with no chain in this locus
    pub p_none: f64,
    /// mode posterior >= tau: the mode is emitted in the primary output, else the whole locus
    /// goes to the low-confidence track
    pub primary: bool,
}

#[derive(Debug, Clone)]
pub struct Mbr {
    pub log_z: f64,
    pub n_samples: usize,
    /// ascending by start
    pub loci: Vec<MbrLocus>,
}

/// Sample-path seed: one independent stream per (seed, strand, sample index).
fn sample_seed(seed: u64, minus: bool, s: usize) -> u64 {
    seed ^ (if minus { 0xa5a5_5a5a_d00d_f00d } else { 0 }) ^ (s as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Expected-F1 (MBR) decoding of one region: forward sum-product, `n_samples` FFBS paths, chain
/// posteriors = sample frequencies, loci = CDS-overlap components; a locus is primary when its mode
/// chain's posterior is >= tau, else it goes whole to the low-confidence track. Nothing is dropped.
/// Each chain also carries its own path score at T = 1 (see `chain_score`), so the MBR gene dicts
/// are the gene representation `decode` returns plus `posterior` and `locus`.
pub fn decode_mbr(e: &Emissions, p: &Params, o: &MbrOpts) -> Result<Mbr, String> {
    use std::collections::HashMap;
    assert!(o.n_samples > 0, "n_samples must be > 0");
    assert!((0.0..=1.0).contains(&o.tau), "tau must be in [0, 1]");
    let sc = check_temperature(o.temperature)?;
    let f = if p.weak_budget == 0 { walk::<true, true>(e, p, sc) } else { walk::<false, true>(e, p, sc) };
    let Some(f) = f else {
        return Ok(Mbr { log_z: 0.0, n_samples: o.n_samples, loci: vec![] });
    };
    let mut ids: HashMap<Vec<Exon>, usize> = HashMap::new();
    let mut chains: Vec<Vec<Exon>> = Vec::new();
    let mut count: Vec<usize> = Vec::new();
    let mut per_sample: Vec<Vec<usize>> = Vec::with_capacity(o.n_samples);
    for s in 0..o.n_samples {
        let mut rng = Rng::new(sample_seed(o.seed, f.minus, s));
        let mut mine = Vec::new();
        for g in f.sample(&mut rng) {
            let cds = f.genomic(e.start, &g);
            let id = *ids.entry(cds.clone()).or_insert_with(|| {
                chains.push(cds);
                count.push(0);
                chains.len() - 1
            });
            count[id] += 1;
            mine.push(id);
        }
        per_sample.push(mine);
    }
    // Chain scores at T = 1 (see `chain_score`): the same sense as `Gene::score`.
    let scores: Vec<(f32, u32)> = chains
        .iter()
        .map(|c| chain_score(e, p, f.l, f.minus, &to_transcript(c, e.start, f.l, f.minus)))
        .collect();
    // loci: union chains whose CDS blocks overlap, sweeping by span start
    let n = chains.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let span = |c: &[Exon]| (c[0].start, c[c.len() - 1].end);
    let overlap = |x: &[Exon], y: &[Exon]| {
        let (mut i, mut j) = (0, 0);
        while i < x.len() && j < y.len() {
            if x[i].start < y[j].end && y[j].start < x[i].end {
                return true;
            }
            if x[i].end <= y[j].end {
                i += 1;
            } else {
                j += 1;
            }
        }
        false
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| chains[x].cmp(&chains[y]));
    let mut active: Vec<usize> = Vec::new();
    for &c in &order {
        let (s, _) = span(&chains[c]);
        active.retain(|&x| span(&chains[x]).1 > s);
        for &x in &active {
            if overlap(&chains[x], &chains[c]) {
                let (rx, rc) = (root(&mut parent, x), root(&mut parent, c));
                if rx != rc {
                    parent[rx.max(rc)] = rx.min(rc);
                }
            }
        }
        active.push(c);
    }
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for c in 0..n {
        let r = root(&mut parent, c);
        members.entry(r).or_default().push(c);
    }
    let mut hits: HashMap<usize, usize> = HashMap::new();
    for mine in &per_sample {
        let mut rs: Vec<usize> = mine.iter().map(|&c| root(&mut parent, c)).collect();
        rs.sort_unstable();
        rs.dedup();
        for r in rs {
            *hits.entry(r).or_insert(0) += 1;
        }
    }
    let ns = o.n_samples as f64;
    let mut loci: Vec<MbrLocus> = members
        .into_iter()
        .map(|(r, mut cs)| {
            cs.sort_by(|&x, &y| count[y].cmp(&count[x]).then_with(|| chains[x].cmp(&chains[y])));
            let start = cs.iter().map(|&c| span(&chains[c]).0).min().unwrap();
            let end = cs.iter().map(|&c| span(&chains[c]).1).max().unwrap();
            let chains: Vec<MbrChain> = cs
                .iter()
                .map(|&c| MbrChain {
                    cds: chains[c].clone(),
                    posterior: count[c] as f64 / ns,
                    score: scores[c].0,
                    n_weak: scores[c].1,
                })
                .collect();
            MbrLocus {
                strand: e.strand,
                start,
                end,
                primary: chains[0].posterior >= o.tau,
                chains,
                p_none: 1.0 - hits[&r] as f64 / ns,
            }
        })
        .collect();
    loci.sort_by(|x, y| (x.start, x.end).cmp(&(y.start, y.end)).then_with(|| x.chains[0].cds.cmp(&y.chains[0].cds)));
    Ok(Mbr { log_z: f.bg[f.l], n_samples: o.n_samples, loci })
}

/// Ascending genomic CDS blocks to transcript intervals.
pub(crate) fn to_transcript(cds: &[Exon], start: u64, l: usize, minus: bool) -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = cds
        .iter()
        .map(|x| {
            let (s, e) = ((x.start - start) as usize, (x.end - start) as usize);
            if minus { (l - e, l - s) } else { (s, e) }
        })
        .collect();
    if minus {
        v.reverse();
    }
    v
}

/// One scored site on a chain: type 0 donor, 1 acceptor, 2 start, 3 stop.
/// `pos` is the genomic coordinate the decoder indexes (`e.start + g(t)`),
/// `p` the head probability there, `motif_ok` the sequence-motif check,
/// `above_thresh` whether the site is a candidate (strong, or weak-rescued).
pub struct ChainSite {
    pub typ: u8,
    pub pos: u64,
    pub p: f32,
    pub motif_ok: bool,
    pub above_thresh: bool,
}

/// Additive components of [`chain_score`], in the same sense (T = 1 path score).
/// `sites` parallels the chain's sites in transcript order (start, donor/acceptor
/// pairs, stop). A site below threshold does NOT force NEG here: it is recorded
/// with `above_thresh = false` and contributes its unthresholded
/// `w_c * ln p + c_bias` score, so infeasible chains still decompose;
/// [`chain_score`] keeps the old NEG behaviour.
pub struct ChainTerms {
    pub gene_start: f64,
    pub site_start: f64,
    pub site_donor: f64,
    pub site_acceptor: f64,
    pub site_stop: f64,
    pub exon_region: f64,
    pub exon_phase: f64,
    pub cds_base_bias: f64,
    pub intron_region: f64,
    pub intron_phase: f64,
    pub intron_geom: f64,
    pub exon_short: f64,
    pub sites: Vec<ChainSite>,
    pub n_weak: u32,
}

impl ChainTerms {
    pub fn total(&self) -> f64 {
        self.gene_start
            + self.site_start
            + self.site_donor
            + self.site_acceptor
            + self.site_stop
            + self.exon_region
            + self.exon_phase
            + self.cds_base_bias
            + self.intron_region
            + self.intron_phase
            + self.intron_geom
            + self.exon_short
    }
}

/// Additive decomposition of [`chain_score`] over transcript exons `tex`.
/// See [`ChainTerms`]: infeasible sites contribute their unthresholded score.
pub fn chain_terms(e: &Emissions, p: &Params, l: usize, minus: bool, tex: &[(usize, usize)]) -> ChainTerms {
    let g = |t: usize| if minus { l - 1 - t } else { t };
    let la = |r: usize, t: usize| e.log_a[r * l + g(t)] as f64;
    let lb = |r: usize, t: usize| e.log_b[r * l + g(t)] as f64;
    let null_b = (1.0f64 / 3.0).ln();
    let kb = p.weak_budget.min(254) as usize;
    // Motif bases on demand in transcript orientation, as `walk` orients them: only the
    // 2-3 bases at each site are read, with no per-call allocation proportional to L
    // (`decode_mbr` calls `chain_score` once per sampled chain on 50 Mb chunks).
    let src = |i: usize| if minus { l - 1 - i } else { i };
    let base_at = |t: usize| {
        let b = e.seq[src(t)].to_ascii_uppercase();
        if minus { comp(b) } else { b }
    };
    let weak_rescue = |h: usize, pr: f32| kb > 0 && p.c_thresh_weak[h] < p.c_thresh[h] && pr >= p.c_thresh_weak[h];
    // (unthresholded score, weak count, above_thresh)
    let site3 = |h: usize, t: usize| -> (f64, u32, bool, f32) {
        let pr = e.c[h * l + g(t)];
        let s = p.w_c as f64 * (pr as f64).max(1e-30).ln() + p.c_bias[h] as f64;
        if pr >= p.c_thresh[h] {
            (s, 0, true, pr)
        } else if weak_rescue(h, pr) {
            (s + p.weak_penalty[h] as f64, 1, true, pr)
        } else {
            (s, 0, false, pr)
        }
    };
    let motif_at = |t: usize, kind: u8| -> bool {
        match kind {
            2 => t + 3 <= l && base_at(t) == b'A' && base_at(t + 1) == b'T' && base_at(t + 2) == b'G',
            0 => t + 2 <= l && matches!((base_at(t), base_at(t + 1)), (b'G', b'T') | (b'G', b'C') | (b'A', b'T')),
            1 => t >= 1 && matches!((base_at(t - 1), base_at(t)), (b'A', b'G') | (b'A', b'C')),
            _ => t + 3 <= l && is_stop3(base_at(t), base_at(t + 1), base_at(t + 2)),
        }
    };
    let gpos = |t: usize| e.start + g(t) as u64;
    let imin = (p.intron_min.max(1) as usize).min(l);
    let l0 = (p.intron_l0 as usize).max(imin);
    let mut t = ChainTerms {
        gene_start: p.gene_start_penalty as f64,
        site_start: 0.0, site_donor: 0.0, site_acceptor: 0.0, site_stop: 0.0,
        exon_region: 0.0, exon_phase: 0.0, cds_base_bias: 0.0,
        intron_region: 0.0, intron_phase: 0.0, intron_geom: 0.0, exon_short: 0.0,
        sites: Vec::new(), n_weak: 0,
    };
    let mut k = 0usize; // CDS bases so far, in transcript order
    let (s0, w0, ok0, pr0) = site3(2, tex[0].0);
    t.site_start += s0;
    t.n_weak += w0;
    t.sites.push(ChainSite { typ: 2, pos: gpos(tex[0].0), p: pr0, motif_ok: motif_at(tex[0].0, 2), above_thresh: ok0 });
    for (n, &(a, b)) in tex.iter().enumerate() {
        if n > 0 {
            let (u, a2) = (tex[n - 1].1, a);
            let (sd, wd, okd, prd) = site3(0, u);
            let (sa, wa, oka, pra) = site3(1, a2 - 1);
            t.site_donor += sd;
            t.site_acceptor += sa;
            t.n_weak += wd + wa;
            t.sites.push(ChainSite { typ: 0, pos: gpos(u), p: prd, motif_ok: motif_at(u, 0), above_thresh: okd });
            t.sites.push(ChainSite { typ: 1, pos: gpos(a2 - 1), p: pra, motif_ok: motif_at(a2 - 1, 1), above_thresh: oka });
            let ph = k % 3;
            for x in u..a2 {
                t.intron_region += la(2, x) - la(0, x);
                t.intron_phase += p.w_b_intron as f64 * (lb(ph, x) - null_b);
            }
            if a2 - u > l0 {
                t.intron_geom += p.intron_geom_b as f64 * (a2 - u - l0) as f64;
            }
        }
        for x in a..b {
            t.exon_region += la(1, x) - la(0, x);
            t.cds_base_bias += p.cds_base_bias as f64;
            t.exon_phase += p.w_b as f64 * lb(k % 3, x);
            k += 1;
        }
        if b - a < p.exon_soft_min as usize {
            t.exon_short += p.exon_short_penalty as f64;
        }
    }
    let te = tex[tex.len() - 1].1 - 3;
    let (s3, w3, ok3, pr3) = site3(3, te);
    t.site_stop += s3;
    t.n_weak += w3;
    t.sites.push(ChainSite { typ: 3, pos: gpos(te), p: pr3, motif_ok: motif_at(te, 3), above_thresh: ok3 });
    t
}

/// The T = 1 gene-path score of one CDS chain (transcript exons `tex`), plus its weak-site
/// count: the same sense as `Gene::score` / `Gene::n_weak`. The chain fixes every site and
/// segment, so this is the same sum the decoder's forward pass accumulates at sc = 1 (gene-start
/// penalty and start site, exon emission sums with short penalties, intron sums with the
/// geometric tail past `l0`, stop site); `Gene::score` subtracts the background the gene-start
/// open started from, which this sum never adds, so no background walk is needed. Site bands
/// are fixed per position (strong and weak bands are disjoint in p), so no maximization over
/// weak assignments is needed either. The chain was sampled through every one of these sites,
/// so each lookup hits a candidate.
fn chain_score(e: &Emissions, p: &Params, l: usize, minus: bool, tex: &[(usize, usize)]) -> (f32, u32) {
    let t = chain_terms(e, p, l, minus, tex);
    if t.sites.iter().any(|s| !s.above_thresh) {
        return (NEG as f32, t.n_weak);
    }
    (t.total() as f32, t.n_weak)
}

