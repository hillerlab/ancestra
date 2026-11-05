//! Output layer: parts → BED12, GTF, GFF3 and bigBed, the sequence-name rule
//! and the run stamp.
//!
//! A *part* is one `<item>.tx.tsv` row: the 24 tab-separated columns of the
//! bigBed autoSql, in that order. `write_outputs` writes the native BED12 plus,
//! on request, the GTF and GFF3 built by genepred 0.0.17's
//! `to_gxf_with_options::<Gtf|Gff>` under the release options, and the bigBed
//! `PREFIX.bb` (BED12+12) written by bigtools 0.5.8 with the autoSql — its
//! table comment is the run stamp — followed by the guarded `definedFieldCount`
//! patch. Every text output carries the stamp line `#S` as its line 1 — line 2
//! of the GFF3, after the `##gff-version 3` GFF3 requires first — and every
//! write is atomic: a unique tmp file, `sync_all`, `rename`, and no partial
//! output on any error. The bigBed carries its stamp in its autoSql table
//! comment, so `stamp_of` reads it back from the header.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::os::unix::ffi::OsStrExt;
use std::sync::atomic::{AtomicU64, Ordering};

use bigtools::beddata::BedParserStreamingIterator;
use bigtools::BigBedWrite;
use crate::twobit::TwoBit;
use genepred::{BedFormat, GeneLine, GenePred, Gff, Gtf, GxfOptions, Strand, TranscriptParent};

/// Columns of a part row: the 24 fields of the bigBed autoSql, in order.
const COLS: usize = 24;
const ALT_COLS: usize = 21;
/// Longest allowed sequence name, in bytes.
const MAX_NAME_BYTES: usize = 238;
/// bigBed's magic at byte 0 (`0x8789F2EB`, little-endian): a stamp lives in its
/// autoSql table comment instead.
const BIGBED_MAGIC: [u8; 4] = [0xEB, 0xF2, 0x89, 0x87];
const ZOOM_HEADER_BYTES: usize = 10 * 24;
const SUMMARY_BYTES: usize = 40;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn bad(src: &Path, line: usize, msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{}:{line}: {msg}", src.display()))
}

pub(crate) fn bad_input(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Not every failure of this module names its file. A part row that does not
/// parse names the file and its line (`bad`, `InvalidData`), and so does a
/// non-UTF-8 line; an I/O error on a named path is an `InvalidInput` whose
/// message starts with that path (`bad_path`), as are `stamp_of`'s file checks,
/// `atomic_write`'s failures and the item refusals (each names its part; the
/// two-part ones name both and start with `sequence` or `name`). The refusals
/// that name no file are `check_stamp`'s stamp text, the 2bit's sha256, the
/// `--format` list and the finalize sequence-name pass.
pub(crate) fn bad_path(path: &Path, e: io::Error) -> io::Error {
    bad_input(format!("{}: {e}", path.display()))
}

/// One part row: the 24 columns of the bigBed autoSql, in that order. The BED12
/// core is columns 1–12; columns 13–24 are the metadata the bigBed carries.
#[derive(Debug, Clone)]
pub struct Part {
    pub chrom: String,
    pub chrom_start: u32,
    pub chrom_end: u32,
    pub name: String,
    pub score: u32,
    pub strand: Strand,
    pub thick_start: u32,
    pub thick_end: u32,
    pub reserved: u32,
    pub block_count: u32,
    pub block_sizes: Vec<u32>,
    pub block_starts: Vec<u32>,
    pub gene_id: String,
    /// The three `float` columns of the autoSql, each carried as its own part
    /// text (checked finite at parse). The bigBed holds the item finish's
    /// spelling byte for byte, which an f64 re-serialization could not
    /// reproduce: Rust prints `0.0` as `0` and `1e-30` as `0.000…001`.
    pub gene_score: String,
    pub cds_prob: String,
    pub start_prob: String,
    pub start_extended: u32,
    pub start_shift: u32,
    pub utr5_blocks: u32,
    pub utr3_blocks: u32,
    pub utr_decoder: String,
    pub release: String,
    pub ckpt_sha256: String,
    pub params_sha256: String,
    /// `path:line` the row was read from, for every later error message.
    pub src: String,
}

impl Part {
    /// Exon blocks as absolute 0-based half-open intervals, ascending.
    fn blocks(&self) -> Vec<(u64, u64)> {
        (0..self.block_count as usize)
            .map(|i| {
                let s = u64::from(self.chrom_start) + u64::from(self.block_starts[i]);
                (s, s + u64::from(self.block_sizes[i]))
            })
            .collect()
    }

    /// Coding exon pieces: the exon blocks clipped to the CDS
    /// `[thickStart, thickEnd)`, ascending. Empty when the CDS holds no exon base.
    fn coding_pieces(&self) -> Vec<(u64, u64)> {
        let (ts, te) = (u64::from(self.thick_start), u64::from(self.thick_end));
        self.blocks()
            .into_iter()
            .filter_map(|(s, e)| {
                let (cs, ce) = (s.max(ts), e.min(te));
                (cs < ce).then_some((cs, ce))
            })
            .collect()
    }

    /// one `GenePred` for the GXF builders: absolute blocks, 0-based half-open.
    fn to_genepred(&self) -> GenePred {
        let blocks = self.blocks();
        GenePred {
            chrom: self.chrom.as_bytes().to_vec(),
            start: u64::from(self.chrom_start),
            end: u64::from(self.chrom_end),
            name: Some(self.name.as_bytes().to_vec()),
            strand: Some(self.strand),
            thick_start: Some(u64::from(self.thick_start)),
            thick_end: Some(u64::from(self.thick_end)),
            block_count: Some(self.block_count),
            block_starts: Some(blocks.iter().map(|(s, _)| *s).collect()),
            block_ends: Some(blocks.iter().map(|(_, e)| *e).collect()),
            extras: Default::default(),
        }
    }
}

/// Reads every row of one part file, in file order.
pub fn read_parts(path: &Path) -> io::Result<Vec<Part>> {
    let data = fs::read(path).map_err(|e| bad_path(path, e))?;
    let text = std::str::from_utf8(&data).map_err(|e| {
        // the first invalid byte's own line: count the '\n' before it
        let line = data[..e.valid_up_to()].iter().filter(|b| **b == b'\n').count() + 1;
        bad(path, line, "part is not UTF-8 text".to_owned())
    })?;
    text.lines().enumerate().map(|(i, line)| parse_row(path, i + 1, line)).collect()
}

fn parse_row(src: &Path, line_no: usize, line: &str) -> io::Result<Part> {
    let f: Vec<&str> = line.split('\t').collect();
    if f.len() != COLS {
        return Err(bad(src, line_no, format!("expected {COLS} columns of the bigBed autoSql, found {}", f.len())));
    }
    let uint = |i: usize, what: &str| -> io::Result<u32> {
        f[i].parse::<u32>()
            .map_err(|_| bad(src, line_no, format!("column {} ({what}) is not an unsigned integer: '{}'", i + 1, f[i])))
    };
    let float = |i: usize, what: &str| -> io::Result<String> {
        let v = f[i]
            .parse::<f64>()
            .map_err(|_| bad(src, line_no, format!("column {} ({what}) is not a float: '{}'", i + 1, f[i])))?;
        if !v.is_finite() {
            return Err(bad(src, line_no, format!("column {} ({what}) is not finite: '{}'", i + 1, f[i])));
        }
        // the column's own text: what the bigBed carries, byte for byte
        Ok(f[i].to_owned())
    };
    let sha = |i: usize, what: &str| -> io::Result<String> {
        if f[i].len() == 64 && f[i].bytes().all(|b| b.is_ascii_hexdigit()) {
            Ok(f[i].to_owned())
        } else {
            Err(bad(src, line_no, format!("column {} ({what}) is not a sha256: '{}'", i + 1, f[i])))
        }
    };

    let chrom = f[0];
    if chrom.is_empty() {
        return Err(bad(src, line_no, "column 1 (chrom) is empty".to_owned()));
    }
    let chrom_start = uint(1, "chromStart")?;
    let chrom_end = uint(2, "chromEnd")?;
    if chrom_start >= chrom_end {
        return Err(bad(src, line_no, format!("column 2 (chromStart {chrom_start}) must be below column 3 (chromEnd {chrom_end})")));
    }
    let name = f[3];
    if name.is_empty() {
        return Err(bad(src, line_no, "column 4 (name) is empty".to_owned()));
    }
    let score = uint(4, "score")?;
    if score > 1000 {
        return Err(bad(src, line_no, format!("column 5 (score {score}) is outside 0-1000")));
    }
    let strand = match f[5] {
        "+" => Strand::Forward,
        "-" => Strand::Reverse,
        other => return Err(bad(src, line_no, format!("column 6 (strand) must be + or -, got '{other}'"))),
    };
    let thick_start = uint(6, "thickStart")?;
    let thick_end = uint(7, "thickEnd")?;
    if thick_start < chrom_start || thick_end > chrom_end || thick_start >= thick_end {
        return Err(bad(src, line_no, format!("the CDS [{thick_start}, {thick_end}) must sit inside the transcript [{chrom_start}, {chrom_end})")));
    }
    let reserved = uint(8, "reserved")?;
    if reserved != 0 {
        return Err(bad(src, line_no, format!("column 9 (reserved {reserved}) must be 0: itemRgb is always 0")));
    }
    let block_count = uint(9, "blockCount")?;
    if block_count == 0 {
        return Err(bad(src, line_no, "column 10 (blockCount) must be at least 1".to_owned()));
    }
    let block_sizes = int_list(src, line_no, 11, "blockSizes", f[10], block_count)?;
    let block_starts = int_list(src, line_no, 12, "chromStarts", f[11], block_count)?;
    let gene_id = f[12];
    if gene_id != name {
        return Err(bad(src, line_no, format!("column 13 (geneId '{gene_id}') must equal column 4 (name '{name}'): one transcript per gene")));
    }
    let (gene_score, cds_prob, start_prob) = (float(13, "geneScore")?, float(14, "cdsProb")?, float(15, "startProb")?);
    let (start_extended, start_shift) = (uint(16, "startExtended")?, uint(17, "startShift")?);
    let (utr5_blocks, utr3_blocks) = (uint(18, "utr5Blocks")?, uint(19, "utr3Blocks")?);
    let utr_decoder = f[20].to_owned();
    let release = f[21].to_owned();
    let (ckpt_sha256, params_sha256) = (sha(22, "ckptSha256")?, sha(23, "paramsSha256")?);

    let part = Part {
        chrom: chrom.to_owned(),
        chrom_start,
        chrom_end,
        name: name.to_owned(),
        score,
        strand,
        thick_start,
        thick_end,
        reserved,
        block_count,
        block_sizes,
        block_starts,
        gene_id: gene_id.to_owned(),
        gene_score,
        cds_prob,
        start_prob,
        start_extended,
        start_shift,
        utr5_blocks,
        utr3_blocks,
        utr_decoder,
        release,
        ckpt_sha256,
        params_sha256,
        src: format!("{}:{line_no}", src.display()),
    };
    check_blocks(&part, src, line_no)?;
    check_cds_geometry(&part, src, line_no)?;
    Ok(part)
}

/// Parses one `int[blockCount]` list: BED12's comma form, one optional trailing
/// comma, no other empty item, and exactly `block_count` integers.
fn int_list(src: &Path, line_no: usize, col: usize, what: &str, field: &str, count: u32) -> io::Result<Vec<u32>> {
    let field = field.strip_suffix(',').unwrap_or(field);
    let mut out = Vec::with_capacity(count as usize);
    for item in field.split(',') {
        if item.is_empty() {
            return Err(bad(src, line_no, format!("column {col} ({what}) has an empty item: '{field}'")));
        }
        let v = item
            .parse::<u32>()
            .map_err(|_| bad(src, line_no, format!("column {col} ({what}) has a bad integer: '{item}'")))?;
        out.push(v);
    }
    if out.len() != count as usize {
        return Err(bad(src, line_no, format!("column {col} ({what}) holds {} integers, but blockCount is {count}", out.len())));
    }
    Ok(out)
}

/// Blocks tile the transcript span exactly: ascending, non-overlapping, none
/// empty, the first starting at chromStart and the last ending at chromEnd.
fn check_blocks(p: &Part, src: &Path, line_no: usize) -> io::Result<()> {
    if p.block_starts[0] != 0 {
        return Err(bad(src, line_no, "column 12 (chromStarts) must start at 0".to_owned()));
    }
    let (start, end) = (u64::from(p.chrom_start), u64::from(p.chrom_end));
    let mut prev_end = start;
    for (i, (s, e)) in p.blocks().into_iter().enumerate() {
        if e <= s || s < prev_end || e > end {
            return Err(bad(src, line_no, format!("block {i} [{s}, {e}) is empty, out of order or outside the transcript")));
        }
        prev_end = e;
    }
    if prev_end != end {
        return Err(bad(src, line_no, "column 12 (chromStarts) must end at chromEnd - chromStart".to_owned()));
    }
    Ok(())
}

/// One BED12 line, hand-written (score from the part, itemRgb 0, trailing
/// commas on the two block lists) so genepred's BED writer is never involved.
fn bed12_line(p: &Part) -> String {
    let join = |v: &[u32]| -> String {
        let mut s = String::new();
        for x in v {
            s.push_str(&x.to_string());
            s.push(',');
        }
        s
    };
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0\t{}\t{}\t{}",
        p.chrom,
        p.chrom_start,
        p.chrom_end,
        p.name,
        p.score,
        if p.strand == Strand::Forward { "+" } else { "-" },
        p.thick_start,
        p.thick_end,
        p.block_count,
        join(&p.block_sizes),
        join(&p.block_starts),
    )
}

/// The release's `GxfOptions`: source `ancestra`, UTR rows, GTF 2.2 shaping
/// and GFF3 percent-encoding, both kinds.
fn gxf_options() -> GxfOptions {
    GxfOptions {
        source: Some("ancestra"),
        utr_rows: true,
        gtf22: true,
        gff_escape: true,
        ..Default::default()
    }
}

/// the CDS geometry, checked on every row of every part whatever the
/// format (`--format` must cost no GPU time): the coding pieces — exon
/// blocks clipped to `[thickStart, thickEnd)` — must start at thickStart, end
/// at thickEnd, hold a CDS whose length is a multiple of 3 (a whole number of
/// codons), and have both genomic extremes at least a complete codon. A codon
/// is never split on either strand: the two genomic extremes are the
/// transcript-order extremes for + and for −, so a shorter one would make
/// genepred write a truncated codon row, and a transcript with no coding exon
/// piece at all would be written as non-coding. Every failure names the
/// transcript and its source line.
fn check_cds_geometry(p: &Part, src: &Path, line_no: usize) -> io::Result<()> {
    let pieces = p.coding_pieces();
    let (ts, te) = (p.thick_start, p.thick_end);
    let (Some(first), Some(last)) = (pieces.first(), pieces.last()) else {
        return Err(bad(src, line_no, format!(
            "transcript {} has no coding exon piece: the CDS [{}, {}) falls outside every block",
            p.name, ts, te
        )));
    };
    if first.0 != u64::from(ts) {
        return Err(bad(src, line_no, format!(
            "transcript {}: the first coding exon piece starts at {}, not at thickStart {}",
            p.name, first.0, ts
        )));
    }
    if last.1 != u64::from(te) {
        return Err(bad(src, line_no, format!(
            "transcript {}: the last coding exon piece ends at {}, not at thickEnd {}",
            p.name, last.1, te
        )));
    }
    for edge in [first, last] {
        if edge.1 - edge.0 < 3 {
            return Err(bad(src, line_no, format!(
                "transcript {} has a {}-bp terminal coding exon piece: a codon is never split",
                p.name,
                edge.1 - edge.0
            )));
        }
    }
    let coding: u64 = pieces.iter().map(|(s, e)| e - s).sum();
    if !coding.is_multiple_of(3) {
        return Err(bad(src, line_no, format!(
            "transcript {}: the CDS [{}, {}) is {coding} bp, not a multiple of 3",
            p.name, ts, te
        )));
    }
    Ok(())
}

/// One 21-column alternative part row. The BED12 geometry lives in `bed` so
/// the primary writer's coordinate checks and serializers stay shared.
#[derive(Debug, Clone)]
struct AltPart {
    bed: Part,
    rank: u32,
    sources: String,
    p: String,
    utr: String,
    release: String,
    ckpt_sha256: String,
    params_sha256: String,
    alt_sha256: String,
}

fn parse_alt_row(
    src: &Path,
    line_no: usize,
    line: &str,
    threshold: f64,
    primaries: &BTreeMap<String, Part>,
) -> io::Result<AltPart> {
    let f: Vec<&str> = line.split('\t').collect();
    if f.len() != ALT_COLS {
        return Err(bad(
            src,
            line_no,
            format!("expected {ALT_COLS} BED12+9 columns, found {}", f.len()),
        ));
    }
    let uint = |i: usize, what: &str| -> io::Result<u32> {
        f[i].parse::<u32>().map_err(|_| {
            bad(
                src,
                line_no,
                format!(
                    "column {} ({what}) is not an unsigned integer: '{}'",
                    i + 1,
                    f[i]
                ),
            )
        })
    };
    let sha = |i: usize, what: &str| -> io::Result<String> {
        if f[i].len() == 64 && f[i].bytes().all(|b| b.is_ascii_hexdigit()) {
            Ok(f[i].to_owned())
        } else {
            Err(bad(
                src,
                line_no,
                format!(
                    "column {} ({what}) is not a 64-hex sha256: '{}'",
                    i + 1,
                    f[i]
                ),
            ))
        }
    };
    let (chrom, name, gene_id) = (f[0], f[3], f[12]);
    let chrom_start = uint(1, "chromStart")?;
    let chrom_end = uint(2, "chromEnd")?;
    if chrom.is_empty() || chrom_start >= chrom_end {
        return Err(bad(
            src,
            line_no,
            "chrom must be nonempty and chromStart below chromEnd".to_owned(),
        ));
    }
    if name.is_empty() {
        return Err(bad(src, line_no, "column 4 (name) is empty".to_owned()));
    }
    let score = uint(4, "score")?;
    let strand = match f[5] {
        "+" => Strand::Forward,
        "-" => Strand::Reverse,
        other => {
            return Err(bad(
                src,
                line_no,
                format!("column 6 (strand) must be + or -, got '{other}'"),
            ))
        }
    };
    let (thick_start, thick_end, reserved, block_count) = (
        uint(6, "thickStart")?,
        uint(7, "thickEnd")?,
        uint(8, "reserved")?,
        uint(9, "blockCount")?,
    );
    if thick_start < chrom_start || thick_end > chrom_end || thick_start >= thick_end {
        return Err(bad(src, line_no, format!("the CDS [{thick_start}, {thick_end}) must sit inside the transcript [{chrom_start}, {chrom_end})")));
    }
    if reserved != 0 {
        return Err(bad(
            src,
            line_no,
            format!("column 9 (reserved {reserved}) must be 0: itemRgb is always 0"),
        ));
    }
    if block_count == 0 {
        return Err(bad(
            src,
            line_no,
            "column 10 (blockCount) must be at least 1".to_owned(),
        ));
    }
    let block_sizes = int_list(src, line_no, 11, "blockSizes", f[10], block_count)?;
    let block_starts = int_list(src, line_no, 12, "chromStarts", f[11], block_count)?;
    let primary = primaries.get(gene_id).ok_or_else(|| {
        bad(
            src,
            line_no,
            format!("geneId '{gene_id}' names no primary in this item's part"),
        )
    })?;
    if primary.chrom != chrom || primary.strand != strand {
        return Err(bad(
            src,
            line_no,
            format!("geneId '{gene_id}' does not match this row's chrom and strand"),
        ));
    }
    let rank = uint(13, "rank")?;
    if rank.to_string() != f[13] {
        return Err(bad(
            src,
            line_no,
            format!("column 14 (rank) must be canonical decimal: '{}'", f[13]),
        ));
    }
    if name != format!("{gene_id}.{rank}") {
        return Err(bad(
            src,
            line_no,
            format!("name '{name}' must be '{gene_id}.{rank}'"),
        ));
    }
    if !matches!(f[14], "ffbs" | "struct" | "ffbs,struct") {
        return Err(bad(
            src,
            line_no,
            format!(
                "column 15 (sources) must be ffbs, struct or ffbs,struct, got '{}'",
                f[14]
            ),
        ));
    }
    let p = f[15].parse::<f64>().map_err(|_| {
        bad(
            src,
            line_no,
            format!(
                "column 16 (p) is not finite or outside [{threshold}, 1]: '{}'",
                f[15]
            ),
        )
    })?;
    if !p.is_finite() || p < threshold || p > 1.0 {
        return Err(bad(
            src,
            line_no,
            format!(
                "column 16 (p) is not finite or outside [{threshold}, 1]: '{}'",
                f[15]
            ),
        ));
    }
    let rounded = (1000.0 * p + 0.5).floor() as u32;
    if score != rounded {
        return Err(bad(
            src,
            line_no,
            format!("column 5 (score {score}) must equal floor(1000 * p + 0.5) = {rounded}"),
        ));
    }
    if !matches!(f[16], "5,3" | "5" | "3" | "none") {
        return Err(bad(
            src,
            line_no,
            format!("column 17 (utr) must be 5,3, 5, 3 or none, got '{}'", f[16]),
        ));
    }
    let (ckpt_sha256, params_sha256, alt_sha256) = (
        sha(18, "ckptSha256")?,
        sha(19, "paramsSha256")?,
        sha(20, "altSha256")?,
    );
    if f[17].is_empty() {
        return Err(bad(src, line_no, "column 18 (release) is empty".to_owned()));
    }
    let bed = Part {
        chrom: chrom.to_owned(),
        chrom_start,
        chrom_end,
        name: name.to_owned(),
        score,
        strand,
        thick_start,
        thick_end,
        reserved,
        block_count,
        block_sizes,
        block_starts,
        gene_id: gene_id.to_owned(),
        gene_score: String::new(),
        cds_prob: String::new(),
        start_prob: String::new(),
        start_extended: 0,
        start_shift: 0,
        utr5_blocks: 0,
        utr3_blocks: 0,
        utr_decoder: String::new(),
        release: f[17].to_owned(),
        ckpt_sha256: ckpt_sha256.clone(),
        params_sha256: params_sha256.clone(),
        src: format!("{}:{line_no}", src.display()),
    };
    check_blocks(&bed, src, line_no)?;
    check_cds_geometry(&bed, src, line_no)?;
    Ok(AltPart {
        bed,
        rank,
        sources: f[14].to_owned(),
        p: f[15].to_owned(),
        utr: f[16].to_owned(),
        release: f[17].to_owned(),
        ckpt_sha256,
        params_sha256,
        alt_sha256,
    })
}

fn alt_cds_key(p: &Part) -> (String, u8, Vec<(u64, u64)>) {
    (
        p.chrom.clone(),
        if p.strand == Strand::Forward { 0 } else { 1 },
        p.coding_pieces(),
    )
}

/// Reads paired parts in run-item order and validates item-local links, ranks,
/// and CDS uniqueness before a writer opens any output.
fn alt_rows(
    alt_parts: &[PathBuf],
    primary_parts: &[PathBuf],
    threshold: f64,
) -> io::Result<(Vec<AltPart>, BTreeMap<String, Part>)> {
    if alt_parts.len() != primary_parts.len() {
        return Err(bad_input(format!(
            "{} alt parts but {} primary parts",
            alt_parts.len(),
            primary_parts.len()
        )));
    }
    if !threshold.is_finite() || !(threshold > 0.0 && threshold <= 1.0) {
        return Err(bad_input(format!(
            "threshold must be finite and in (0, 1], got {threshold}"
        )));
    }
    let mut all_alt = Vec::new();
    let mut all_primary = BTreeMap::new();
    for (alt_path, primary_path) in alt_parts.iter().zip(primary_parts) {
        let primary_rows = read_parts(primary_path)?;
        let mut primaries = BTreeMap::new();
        let mut primary_keys = BTreeSet::new();
        for p in primary_rows {
            if primaries.insert(p.name.clone(), p.clone()).is_some() {
                return Err(bad_input(format!(
                    "{}: duplicate primary name '{}'",
                    p.src, p.name
                )));
            }
            primary_keys.insert(alt_cds_key(&p));
            if all_primary.insert(p.name.clone(), p.clone()).is_some() {
                return Err(bad_input(format!(
                    "{}: primary name '{}' occurs in more than one item",
                    p.src, p.name
                )));
            }
        }
        let bytes = fs::read(alt_path).map_err(|e| bad_path(alt_path, e))?;
        let text = std::str::from_utf8(&bytes).map_err(|e| {
            let line = bytes[..e.valid_up_to()]
                .iter()
                .filter(|b| **b == b'\n')
                .count()
                + 1;
            bad(alt_path, line, "part is not UTF-8 text".to_owned())
        })?;
        let mut ranks = BTreeMap::<String, BTreeMap<u32, usize>>::new();
        let mut alt_keys = BTreeSet::new();
        for (i, line) in text.lines().enumerate() {
            let line_no = i + 1;
            let alt = parse_alt_row(alt_path, line_no, line, threshold, &primaries)?;
            if alt.rank > 3 {
                return Err(bad(
                    alt_path,
                    line_no,
                    format!("rank {} exceeds the per-gene cap of 3", alt.rank),
                ));
            }
            let gene_ranks = ranks.entry(alt.bed.gene_id.clone()).or_default();
            if gene_ranks.insert(alt.rank, line_no).is_some() {
                return Err(bad(
                    alt_path,
                    line_no,
                    format!("rank {} for gene '{}' occurs more than once", alt.rank, alt.bed.gene_id),
                ));
            }
            let key = alt_cds_key(&alt.bed);
            if primary_keys.contains(&key) {
                return Err(bad(
                    alt_path,
                    line_no,
                    format!(
                        "CDS key for '{}' equals a same-strand primary in this item",
                        alt.bed.name
                    ),
                ));
            }
            if !alt_keys.insert(key) {
                return Err(bad(
                    alt_path,
                    line_no,
                    format!(
                        "CDS key for '{}' duplicates an alternative on this strand of the item",
                        alt.bed.name
                    ),
                ));
            }
            all_alt.push(alt);
        }
        for (gene_id, gene_ranks) in ranks {
            for (i, (&rank, &line_no)) in gene_ranks.iter().enumerate() {
                let expected = i as u32 + 1;
                if rank != expected {
                    return Err(bad(alt_path, line_no, format!("ranks for gene '{gene_id}' must be 1..n contiguous; expected {expected}, found {rank}")));
                }
            }
        }
    }
    Ok((all_alt, all_primary))
}

/// GTF or GFF3 lines of one part, via genepred 0.0.17 under the release
/// options. `gtf22` trims the stop codon out of the GTF CDS rows, so the
/// 3-bp stop-only terminal coding exon of the toy pair leaves no GTF CDS row.
fn gxf_lines<K>(p: &Part) -> Vec<Vec<u8>>
where
    K: BedFormat,
{
    p.to_genepred().to_gxf_with_options::<K>(&gxf_options(), None)
}

/// One line read by `stamp_of`: `BufRead::lines` reports a non-UTF-8 line as
/// `InvalidData` with no position of its own, so it is named by its line.
fn text_line(path: &Path, line_no: usize, r: io::Result<String>) -> io::Result<String> {
    r.map_err(|e| {
        if e.kind() == io::ErrorKind::InvalidData {
            bad(path, line_no, "not UTF-8 text".to_owned())
        } else {
            bad_path(path, e)
        }
    })
}

/// the payload of any text output — the `#S` line without its `#`.
/// Line 1 of the BED12, GTF and every bedGraph; line 2 of the GFF3, after the
/// `##gff-version 3` GFF3 requires first. A bigBed's stamp is its autoSql table
/// comment and is read from the file's header (the layout). Streamed line by
/// line: a multi-GB bedGraph is re-read by finalize without holding it in RAM.
pub fn stamp_of(path: &Path) -> io::Result<String> {
    let mut file = File::open(path).map_err(|e| bad_path(path, e))?;
    let mut magic = [0u8; 4];
    let n = file.read(&mut magic).map_err(|e| bad_path(path, e))?;
    drop(file);
    let payload = if n == 4 && magic == BIGBED_MAGIC {
        bigbed_stamp(path)?
    } else {
        let reader = BufReader::new(File::open(path).map_err(|e| bad_path(path, e))?);
        let mut lines = reader.lines();
        let first = lines
            .next()
            .ok_or_else(|| bad_input(format!("{}: empty file has no stamp line", path.display())))
            .and_then(|l| text_line(path, 1, l))?;
        let (stamp_line_no, stamp_line) = if first.starts_with("##gff-version") {
            let second = lines
                .next()
                .ok_or_else(|| bad_input(format!("{}: a GFF3 needs its stamp on line 2", path.display())))
                .and_then(|l| text_line(path, 2, l))?;
            (2, second)
        } else {
            (1, first)
        };
        let payload = stamp_line
            .strip_prefix('#')
            .ok_or_else(|| bad_input(format!(
                "{}: line {stamp_line_no} is not the stamp line: a text output's stamp starts with '#'",
                path.display()
            )))?
            .to_owned();
        for (i, line) in lines.enumerate() {
            let line = text_line(path, stamp_line_no + i + 1, line)?;
            if line.starts_with('#') {
                return Err(bad_input(format!(
                    "{}: line {} starts with '#': a text output holds exactly one stamp line, GFF3's line 1 excepted",
                    path.display(),
                    stamp_line_no + i + 1
                )));
            }
        }
        payload
    };
    // the stamp is one line of printable ASCII without `"`, whatever file it came from
    if payload.is_empty()
        || payload.bytes().any(|b| !(0x20..=0x7E).contains(&b))
        || payload.contains('"')
    {
        return Err(bad_input(format!(
            "{}: the stamp must be one line of printable ASCII without '\"'",
            path.display()
        )));
    }
    Ok(payload)
}

/// S is one line of printable ASCII without `"`; the file line is
/// `#` + S. Checked before any output is opened.
pub(crate) fn check_stamp(stamp: &str) -> io::Result<()> {
    if stamp.is_empty() || stamp.bytes().any(|b| !(0x20..=0x7E).contains(&b)) || stamp.contains('"') {
        return Err(bad_input(format!(
            "the stamp must be one line of printable ASCII without '\"', got '{stamp}'"
        )));
    }
    Ok(())
}

/// the run stamp's ` genome=` field — S = the install stamp + ` genome=<2bit
/// basename>:<sha256>`, the basename percent-encoded (upper-case `%XX`)
/// outside `[A-Za-z0-9._-]`, so S is one line of printable ASCII without `"`
/// whatever the file is called (`a b"c.2bit` → `genome=a%20b%22c.2bit:`).
/// The raw basename bytes are encoded, and the sha is checked, so a
/// non-UTF-8 or malformed path or sha fails loudly instead of writing a
/// payload no one can verify.
pub fn stamp_genome(path: &Path, sha256: &str) -> io::Result<String> {
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(bad_input(format!("the 2bit's sha256 must be 64 lower-case hex digits, got '{sha256}'")));
    }
    let base = path
        .file_name()
        .filter(|b| !b.is_empty() && *b != "." && *b != "..")
        .ok_or_else(|| {
            bad_input(format!(
                "{}: the 2bit path needs a file name that is not empty, '.' or '..'",
                path.display()
            ))
        })?;
    let mut enc = String::new();
    for &b in base.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-') {
            enc.push(char::from(b));
        } else {
            enc.push_str(&format!("%{b:02X}"));
        }
    }
    Ok(format!(" genome={enc}:{sha256}"))
}

/// Python's `ascii()`: the quotes follow Python's rule — single quotes, or
/// double quotes when the name holds `'` and no `"` (prints refused names
/// with it so that no name can break the message).
pub(crate) fn ascii_escape(name: &str) -> String {
    let quote = if name.contains('\'') && !name.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in name.chars() {
        match c {
            // Python escapes the delimiter only in single-quoted mode (in
            // double-quoted mode the string holds no `"` at all)
            '\'' if quote == '\'' => out.push_str("\\'"),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (0x20..0x7f).contains(&(c as u32)) => out.push(c),
            c if (c as u32) <= 0xff => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) <= 0xffff => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out.push(quote);
    out
}

/// One name's refusal under Sequence names, with the message shape.
fn name_refusal(name: &str, gtf: bool) -> Option<String> {
    let unsafe_reason = |detail: String| {
        format!(
            "sequence name {} is unsafe for BED, bigBed and bedGraph ({detail}): names are 1-{MAX_NAME_BYTES} bytes of \
             0x21-0x7E and do not start with '#', 'track' or 'browser'",
            ascii_escape(name)
        )
    };
    let bytes = name.as_bytes();
    if bytes.is_empty() {
        return Some(unsafe_reason("empty".to_owned()));
    }
    if bytes.len() > MAX_NAME_BYTES {
        return Some(unsafe_reason(format!("{} bytes, at most {MAX_NAME_BYTES}", bytes.len())));
    }
    if let Some((i, b)) = bytes.iter().enumerate().find(|(_, b)| !(0x21..=0x7E).contains(*b)) {
        return Some(unsafe_reason(format!("byte 0x{b:02X} at offset {i}")));
    }
    for prefix in ["#", "track", "browser"] {
        if name.starts_with(prefix) {
            return Some(unsafe_reason(format!("starts with '{prefix}'")));
        }
    }
    if gtf && name.contains(['"', ';', '#']) {
        return Some(format!(
            "sequence name {} contains '\"', ';' or '#' (GTF attribute values are quoted and end at ';', and '#' starts a \
             comment)",
            ascii_escape(name)
        ));
    }
    None
}

/// Sequence names: the per-name rule, listed once per distinct name (the CLI
/// lists every refused name of the run). Empty when every name is safe.
pub(crate) fn name_refusals(names: &[String], gtf: bool) -> Vec<(String, String)> {
    let mut seen = BTreeSet::new();
    let mut refused = Vec::new();
    for name in names {
        if !seen.insert(name.as_str()) {
            continue; // every distinct name is listed once
        }
        if let Some(reason) = name_refusal(name, gtf) {
            refused.push((ascii_escape(name), reason));
        }
    }
    refused
}

/// Sequence names, every run, before any model is loaded: the rule on the
/// run's chroms, a duplicate anywhere in the 2bit's index (read from the raw
/// index, `TwoBit::names`, so a name the open-time HashMap collapsed is still
/// counted twice), and — with `gtf` — the names GTF cannot carry. Returns every
/// refused name (ascii-escaped) with its reason; empty when the names are safe.
pub fn check_names(genome: &TwoBit, chroms: &[String], gtf: bool) -> Vec<(String, String)> {
    let mut refused = name_refusals(chroms, gtf);
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for name in genome.names() {
        *counts.entry(name).or_insert(0) += 1;
    }
    for (name, count) in counts {
        if count > 1 {
            refused.push((
                ascii_escape(name),
                format!(
                    "sequence name {} occurs {count} times in the index: refusing an ambiguous 2bit",
                    ascii_escape(name)
                ),
            ));
        }
    }
    refused
}

/// `prefix.ext`, never `Path::with_extension` (a dot in the prefix would eat it).
pub(crate) fn out_path(prefix: &str, ext: &str) -> PathBuf {
    PathBuf::from(format!("{prefix}.{ext}"))
}

/// The line of a `Part::src` (`path:line`), for the message that must name the
/// part once and still point at the row the name was first seen on.
fn line_of(src: &str) -> &str {
    match src.rsplit_once(':') {
        Some((_, line)) => line,
        None => src,
    }
}

/// Reserve the unique tmp file for `path`: `path` with a unique suffix
/// appended (≤ 64 tries), so the rename onto `path` is atomic. Shared by
/// `atomic_write` (the body writes through the returned file) and
/// `atomic_write_path` (a library writes the file itself: the bigWig).
fn reserve_tmp(path: &Path) -> io::Result<(PathBuf, File)> {
    (0..64)
        .find_map(|_| {
            let mut name = path.file_name()?.to_os_string();
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            name.push(format!(".tmp.{}.{}", std::process::id(), id));
            let tmp = path.with_file_name(name);
            match OpenOptions::new().write(true).create_new(true).open(&tmp) {
                Ok(file) => Some(Ok((tmp, file))),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => None,
                Err(e) => Some(Err(e)),
            }
        })
        .ok_or_else(|| bad_input("could not reserve an output temp file".to_owned()))?
}

/// Writes `path` atomically: a unique tmp file, `sync_all`, `rename`; on any
/// error the tmp file is removed and no partial output is left. `body` writes
/// the content, returns the number of data records, and is handed the tmp path
/// it writes to: the bigBed's patch and read-back re-open their own file,
/// which the borrowed writer alone cannot do (bigtools takes its writer by
/// value).
pub(crate) fn atomic_write<F>(path: &Path, body: F) -> io::Result<usize>
where
    F: FnOnce(&mut BufWriter<File>, &Path) -> io::Result<usize>,
{
    // the output names itself in every failure, whatever the stage
    let write = || -> io::Result<usize> {
        let (tmp_path, file) = reserve_tmp(path)?;
        let written = (|| {
            let mut writer = BufWriter::new(file);
            let n = body(&mut writer, &tmp_path)?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
            Ok(n)
        })();
        let n = match written {
            Ok(n) => match fs::rename(&tmp_path, path) {
                Ok(()) => n,
                Err(e) => {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(e);
                }
            },
            Err(e) => {
                let _ = fs::remove_file(&tmp_path);
                return Err(e);
            }
        };
        Ok(n)
    };
    write().map_err(|e| bad_path(path, e))
}

/// The same atomic write for an output a library writes itself (the
/// bigWig, built by bigtools): `body` receives the reserved tmp path, writes
/// the whole file there and returns its data-record count; the file is renamed
/// onto `path` on success and removed on any error.
pub(crate) fn atomic_write_path<F>(path: &Path, body: F) -> io::Result<usize>
where
    F: FnOnce(&Path) -> io::Result<usize>,
{
    let write = || -> io::Result<usize> {
        let (tmp_path, file) = reserve_tmp(path)?;
        match body(&tmp_path) {
            Ok(n) => {
                if let Err(e) = file.sync_all() {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(e);
                }
                match fs::rename(&tmp_path, path) {
                    Ok(()) => Ok(n),
                    Err(e) => {
                        let _ = fs::remove_file(&tmp_path);
                        Err(e)
                    }
                }
            }
            Err(e) => {
                let _ = fs::remove_file(&tmp_path);
                Err(e)
            }
        }
    };
    write().map_err(|e| bad_path(path, e))
}

/// Items and the BED12 row order: the BED12 keeps the Viterbi order — the
/// stable sort on (chrom bytes, Viterbi CDS start) — with n the 1-based rank
/// within each chrom, so every part holds one
/// chrom's rows in Viterbi order and the parts follow chrom byte order. An
/// item is one sequence range with at most one item per chrom, so a chrom
/// spread over two parts, a part holding two chroms, a repeated name and a
/// non-increasing or non-canonical n within a chrom are all refused here,
/// before any output is opened. Returns the rows in BED12 order.
fn bed12_rows(parts: &[PathBuf]) -> io::Result<Vec<Part>> {
    let mut per_part: Vec<(String, Vec<Part>, PathBuf)> = Vec::new();
    for part in parts {
        let rows = read_parts(part)?;
        if rows.is_empty() {
            continue; // an item with no transcript writes no row
        }
        let chrom = rows[0].chrom.clone();
        for r in &rows {
            if r.chrom != chrom {
                // `r.src` is `part:line`, so the part is named once
                return Err(bad_input(format!(
                    "{}: the part holds chroms {} and {}: one part is one item, and an item is one sequence range",
                    r.src,
                    ascii_escape(&chrom),
                    ascii_escape(&r.chrom)
                )));
            }
        }
        per_part.push((chrom, rows, part.clone()));
    }
    let mut first_part: BTreeMap<&str, &Path> = BTreeMap::new();
    for (chrom, rows, part) in &per_part {
        if let Some(prev) = first_part.insert(chrom.as_str(), part) {
            // the same part twice names itself once (with the line); two parts
            // name each one, ascii-escaped so no path can break the message
            return Err(bad_input(if prev == part {
                format!(
                    "{}: the part is listed twice: at most one item per sequence, so one part per chrom",
                    rows[0].src
                )
            } else {
                format!(
                    "sequence {} is in parts {} and {}: at most one item per sequence, so one part per chrom",
                    ascii_escape(chrom),
                    ascii_escape(&prev.display().to_string()),
                    ascii_escape(&part.display().to_string())
                )
            }));
        }
    }
    let mut first_name: BTreeMap<&str, (&Path, &Part)> = BTreeMap::new();
    for (_, rows, part) in &per_part {
        for r in rows {
            if let Some((prev_part, prev)) = first_name.insert(r.name.as_str(), (part.as_path(), r)) {
                // one part named once: the repeat's own `path:line`, then the
                // line the name was first seen on
                return Err(bad_input(if prev_part == part.as_path() {
                    format!(
                        "{}: name {} repeats line {}: one transcript per gene, so every name is unique",
                        r.src,
                        ascii_escape(&r.name),
                        line_of(&prev.src)
                    )
                } else {
                    format!(
                        "name {} occurs in parts {} and {}: one transcript per gene, so every name is unique",
                        ascii_escape(&r.name),
                        ascii_escape(&prev_part.display().to_string()),
                        ascii_escape(&part.display().to_string())
                    )
                }));
            }
        }
    }
    for (chrom, rows, _) in &per_part {
        let prefix = format!("ancestra_{chrom}_");
        let mut prev_n: Option<u32> = None;
        for r in rows {
            // n is canonical: the 1-based rank spelled in decimal, no sign, no
            // leading zero and no zero (`_0`, `_01` and `_+1` are not names
            // `merge_parts` can produce)
            let n = r.name.strip_prefix(&prefix).and_then(|n| {
                let v = n.parse::<u32>().ok()?;
                (v >= 1 && v.to_string() == n).then_some(v)
            });
            match (n, prev_n) {
                (Some(n), Some(p)) if n <= p => {
                    return Err(bad_input(format!(
                        "{}: name {} follows {} in the same chrom: n is the 1-based Viterbi rank within \
                         the chrom, so it must increase",
                        r.src,
                        ascii_escape(&r.name),
                        ascii_escape(&format!("ancestra_{chrom}_{p}"))
                    )));
                }
                (Some(n), _) => prev_n = Some(n),
                (None, _) => {
                    return Err(bad_input(format!(
                        "{}: name {} is not {} with n a rank within its chrom",
                        r.src,
                        ascii_escape(&r.name),
                        ascii_escape(&format!("{prefix}<n>"))
                    )));
                }
            }
        }
    }
    // parts in chrom byte order (stable); rows keep each part's Viterbi order
    per_part.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    Ok(per_part.into_iter().flat_map(|(_, rows, _)| rows).collect())
}

/// Reads the parts (in the run's item order) and writes the outputs: the BED12
/// always, the GTF, GFF3 and bigBed when `formats` asks. The BED12 rows keep
/// the Viterbi order; the GTF, GFF3 and bigBed follow the bigBed order
/// (chrom bytes, chromStart, chromEnd, name), and the bigBed's chrom sizes come
/// from the caller — the 2bit's, which only the bigBed reads. Returns each
/// output path with its data-record count (transcripts for the BED12 and the
/// bigBed, feature lines for the GXF files).
pub fn write_outputs(
    parts: &[PathBuf],
    prefix: &str,
    stamp: &str,
    formats: &[&str],
    chrom_sizes: &BTreeMap<String, u64>,
) -> io::Result<Vec<(PathBuf, usize)>> {
    check_stamp(stamp)?;
    let (mut want_gtf, mut want_gff, mut want_bb) = (false, false, false);
    for f in formats {
        match *f {
            "bed" => {}
            "gtf" => want_gtf = true,
            "gff" => want_gff = true,
            "bb" => want_bb = true,
            other => {
                return Err(bad_input(format!(
                    "unknown format '{other}' (choose from bed, gtf, gff, bb)"
                )))
            }
        }
    }
    // Every part is read, ordered, name-checked and every GXF line built
    // before any output is opened, so a bad part, a bad name or a split codon
    // leaves the output directory untouched.
    let rows = bed12_rows(parts)?;
    // `check_names` applies both rules; finalize calls it again before
    // writing, now without a 2bit: the per-name rule on every distinct chrom,
    // with gtf = whether GTF is requested. It is the *sequence*-name rule, and
    // `bed12_rows` has just proved every transcript name is
    // `ancestra_<chrom>_<n>`: a name holds a refused byte, prefix or GTF
    // character exactly when its chrom does, and the 238-byte rule bounds the
    // chrom, never the name (caps `name` at 255 bytes, which
    // "ancestra_" + 238 + "_" + n holds for n < 10⁷).
    let chroms: Vec<String> = rows.iter().map(|r| r.chrom.clone()).collect();
    let refused = name_refusals(&chroms, want_gtf);
    if !refused.is_empty() {
        return Err(bad_input(format!(
            "the run's sequence names are refused: {}",
            refused.iter().map(|(_, reason)| reason.as_str()).collect::<Vec<_>>().join("; ")
        )));
    }
    // the GXF files and the bigBed are written in the bigBed order.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| {
        (rows[a].chrom.as_bytes(), rows[a].chrom_start, rows[a].chrom_end, rows[a].name.as_bytes())
            .cmp(&(rows[b].chrom.as_bytes(), rows[b].chrom_start, rows[b].chrom_end, rows[b].name.as_bytes()))
    });
    let gtf_rows = want_gtf.then(|| order.iter().map(|&i| gxf_lines::<Gtf>(&rows[i])).collect::<Vec<_>>());
    let gff_rows = want_gff.then(|| order.iter().map(|&i| gxf_lines::<Gff>(&rows[i])).collect::<Vec<_>>());
    let bb = want_bb.then(|| bigbed_rows(&rows, &order, chrom_sizes)).transpose()?;
    let mut written = Vec::new();
    {
        let path = out_path(prefix, "bed");
        let n = atomic_write(&path, |w, _| {
            writeln!(w, "#{stamp}")?;
            for p in &rows {
                writeln!(w, "{}", bed12_line(p))?;
            }
            Ok(rows.len())
        })?;
        written.push((path, n));
    }
    for (gxf_rows, ext, header) in [
        (gtf_rows.as_deref(), "gtf", format!("#{stamp}\n")),
        (gff_rows.as_deref(), "gff3", format!("##gff-version 3\n#{stamp}\n")),
    ] {
        let Some(rows) = gxf_rows else { continue };
        let path = out_path(prefix, ext);
        let n = atomic_write(&path, |w, _| {
            w.write_all(header.as_bytes())?;
            let mut count = 0;
            for row in rows {
                for line in row {
                    w.write_all(line)?;
                    w.write_all(b"\n")?;
                    count += 1;
                }
            }
            Ok(count)
        })?;
        written.push((path, n));
    }
    if let Some((rows, sizes)) = bb {
        let path = out_path(prefix, "bb");
        let n = write_bigbed(&path, &rows, &sizes, stamp)?;
        written.push((path, n));
    }
    Ok(written)
}

fn alt_bed17_line(a: &AltPart) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        bed12_line(&a.bed),
        a.bed.gene_id,
        a.rank,
        a.sources,
        a.p,
        a.utr
    )
}

fn alt_bed21_line(a: &AltPart) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}",
        alt_bed17_line(a),
        a.release,
        a.ckpt_sha256,
        a.params_sha256,
        a.alt_sha256
    )
}

fn alt_gxf_lines<K: BedFormat>(
    a: &AltPart,
    gene_map: &HashMap<String, String>,
    gff: bool,
) -> Vec<Vec<u8>> {
    let options = GxfOptions {
        gene_line: GeneLine::Omit,
        transcript_parent: TranscriptParent::Include,
        ..gxf_options()
    };
    let mut lines = a
        .bed
        .to_genepred()
        .to_gxf_with_options::<K>(&options, Some(gene_map));
    let transcript = lines
        .first_mut()
        .expect("genepred always writes a transcript row");
    if gff {
        transcript.extend_from_slice(
            format!(
                "rank={};sources={};p={};utr={};",
                a.rank, a.sources, a.p, a.utr
            )
            .as_bytes(),
        );
    } else {
        transcript.extend_from_slice(
            format!(
                " rank \"{}\"; sources \"{}\"; p \"{}\"; utr \"{}\";",
                a.rank, a.sources, a.p, a.utr
            )
            .as_bytes(),
        );
    }
    lines
}

fn alt_gff_gene_lines(alts: &[AltPart], primaries: &BTreeMap<String, Part>) -> BTreeMap<String, Vec<u8>> {
    let mut spans = BTreeMap::<String, (Part, u32, u32)>::new();
    for a in alts {
        let primary = primaries
            .get(&a.bed.gene_id)
            .expect("alt_rows checked geneId");
        let entry = spans
            .entry(a.bed.gene_id.clone())
            .or_insert_with(|| (primary.clone(), primary.chrom_start, primary.chrom_end));
        entry.1 = entry.1.min(a.bed.chrom_start);
        entry.2 = entry.2.max(a.bed.chrom_end);
    }
    let mut rows = BTreeMap::new();
    for (gene_id, (mut primary, start, end)) in spans {
        primary.chrom_start = start;
        primary.chrom_end = end;
        let pred = primary.to_genepred();
        let map = HashMap::from([(gene_id.clone(), format!("gene-{gene_id}"))]);
        let options = GxfOptions {
            gene_line: GeneLine::Include,
            ..gxf_options()
        };
        let line = pred
            .to_gxf_with_options::<Gff>(&options, Some(&map))
            .remove(0);
        rows.insert(gene_id, line);
    }
    rows
}

/// Writes the thin alternative track from paired item parts. BED is always
/// emitted; requested GXF and bigBed rows follow the primary writer's order.
pub fn write_alt_outputs(
    alt_parts: &[PathBuf],
    primary_parts: &[PathBuf],
    prefix: &str,
    stamp: &str,
    formats: &[&str],
    chrom_sizes: &BTreeMap<String, u64>,
    threshold: f64,
) -> io::Result<Vec<(PathBuf, usize)>> {
    check_stamp(stamp)?;
    let (mut want_gtf, mut want_gff, mut want_bb) = (false, false, false);
    for f in formats {
        match *f {
            "bed" => {}
            "gtf" => want_gtf = true,
            "gff" => want_gff = true,
            "bb" => want_bb = true,
            other => {
                return Err(bad_input(format!(
                    "unknown format '{other}' (choose from bed, gtf, gff, bb)"
                )))
            }
        }
    }
    // Keep the primary file's canonical-name, part-order and uniqueness
    // checks in the shared reader; alt_rows then pairs those parts by item.
    let _ = bed12_rows(primary_parts)?;
    let (alts, primaries) = alt_rows(alt_parts, primary_parts, threshold)?;
    let chroms: Vec<String> = alts.iter().map(|a| a.bed.chrom.clone()).collect();
    let refused = name_refusals(&chroms, want_gtf);
    if !refused.is_empty() {
        return Err(bad_input(format!(
            "the run's sequence names are refused: {}",
            refused
                .iter()
                .map(|(_, r)| r.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        )));
    }
    let mut order: Vec<usize> = (0..alts.len()).collect();
    order.sort_by(|&a, &b| {
        let (a, b) = (&alts[a].bed, &alts[b].bed);
        (
            a.chrom.as_bytes(),
            a.chrom_start,
            a.chrom_end,
            a.name.as_bytes(),
        )
            .cmp(&(
                b.chrom.as_bytes(),
                b.chrom_start,
                b.chrom_end,
                b.name.as_bytes(),
            ))
    });
    // Stable chrom-byte sort matches bed12_rows' part order and keeps each
    // alt part's row order intact.
    let mut bed_order: Vec<usize> = (0..alts.len()).collect();
    bed_order.sort_by(|&a, &b| {
        alts[a]
            .bed
            .chrom
            .as_bytes()
            .cmp(alts[b].bed.chrom.as_bytes())
    });
    let gtf_map: HashMap<String, String> = alts
        .iter()
        .map(|a| (a.bed.name.clone(), a.bed.gene_id.clone()))
        .collect();
    let gff_map: HashMap<String, String> = alts
        .iter()
        .map(|a| (a.bed.name.clone(), format!("gene-{}", a.bed.gene_id)))
        .collect();
    let gtf_rows = want_gtf.then(|| {
        order
            .iter()
            .map(|&i| alt_gxf_lines::<Gtf>(&alts[i], &gtf_map, false))
            .collect::<Vec<_>>()
    });
    let gff_rows = want_gff.then(|| {
        order
            .iter()
            .map(|&i| alt_gxf_lines::<Gff>(&alts[i], &gff_map, true))
            .collect::<Vec<_>>()
    });
    let gene_rows = want_gff.then(|| alt_gff_gene_lines(&alts, &primaries));
    let bb = if want_bb {
        let core: Vec<Part> = alts.iter().map(|a| a.bed.clone()).collect();
        Some(bigbed_rows(&core, &order, chrom_sizes)?.1)
    } else {
        None
    };
    let mut written = Vec::new();
    let path = out_path(prefix, "alt.bed");
    let n = atomic_write(&path, |w, _| {
        writeln!(w, "#{stamp}")?;
        for &i in &bed_order {
            writeln!(w, "{}", alt_bed17_line(&alts[i]))?;
        }
        Ok(alts.len())
    })?;
    written.push((path, n));
    if let Some(rows) = gtf_rows {
        let path = out_path(prefix, "alt.gtf");
        let n = atomic_write(&path, |w, _| {
            writeln!(w, "#{stamp}")?;
            let mut n = 0;
            for row in rows.iter().flatten() {
                w.write_all(row)?;
                w.write_all(b"\n")?;
                n += 1;
            }
            Ok(n)
        })?;
        written.push((path, n));
    }
    if let Some(rows) = gff_rows {
        let path = out_path(prefix, "alt.gff3");
        let genes = gene_rows.as_ref().unwrap();
        let n = atomic_write(&path, |w, _| {
            writeln!(w, "##gff-version 3\n#{stamp}")?;
            let mut n = 0;
            let mut emitted = BTreeSet::new();
            for (&i, row) in order.iter().zip(&rows) {
                let gene_id = &alts[i].bed.gene_id;
                if emitted.insert(gene_id.clone()) {
                    w.write_all(genes.get(gene_id).unwrap())?;
                    w.write_all(b"\n")?;
                    n += 1;
                }
                for line in row {
                    w.write_all(line)?;
                    w.write_all(b"\n")?;
                    n += 1;
                }
            }
            Ok(n)
        })?;
        written.push((path, n));
    }
    if let Some(sizes) = bb {
        let rows: Vec<&AltPart> = order.iter().map(|&i| &alts[i]).collect();
        let path = out_path(prefix, "alt.bb");
        let n = write_alt_bigbed(&path, &rows, &sizes, stamp)?;
        written.push((path, n));
    }
    Ok(written)
}

fn write_alt_bigbed(
    path: &Path,
    rows: &[&AltPart],
    chrom_sizes: &HashMap<String, u32>,
    stamp: &str,
) -> io::Result<usize> {
    if rows.is_empty() {
        return write_empty_bigbed_with(path, stamp, &alt_autosql(stamp), ALT_COLS);
    }
    let mut text = String::new();
    for a in rows {
        text.push_str(&alt_bed21_line(a));
        text.push('\n');
    }
    atomic_write(path, |w, tmp| {
        let out = w.get_ref().try_clone()?;
        let mut bbw = BigBedWrite::new(out, chrom_sizes.clone());
        bbw.options.inmemory = true;
        bbw.autosql = Some(alt_autosql(stamp));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .build()
            .map_err(|e| bad_input(format!("the bigBed's tokio runtime: {e}")))?;
        bbw.write(
            BedParserStreamingIterator::from_bed_file(Cursor::new(text.into_bytes()), false),
            runtime,
        )
        .map_err(|e| bad_input(format!("bigtools BigBedWrite: {e}")))?;
        let mut f = OpenOptions::new().read(true).write(true).open(tmp)?;
        patch_defined_field_count(&mut f)?;
        drop(f);
        check_bigbed(tmp, rows.len(), stamp, ALT_COLS)?;
        Ok(rows.len())
    })
}

/// the autoSql, byte for byte (field names, types and comments), with the run
/// stamp S as the table comment. bigtools parses it to learn the field count
/// (24 here: `parse_autosql` counts the fields of the single `table`
/// declaration) and writes it into the file, so the table comment carries the
/// stamp into every bigBed. The one `{}` is where S goes; the text holds no
/// other brace, and a stamp holds no `"` (`check_stamp`), so the comment is one
/// quoted string.
const AUTOSQL: &str = r#"table ancestraTx
"{}"
    (
    string chrom;            "Sequence name (2bit)"
    uint   chromStart;       "Transcript start, 0-based"
    uint   chromEnd;         "Transcript end, half-open"
    string name;             "ancestra_<chrom>_<n>"
    uint   score;            "round(1000 * cdsProb), half up, 0-1000"
    char[1] strand;          "+ or -"
    uint   thickStart;       "CDS start (stop codon included on -)"
    uint   thickEnd;         "CDS end (stop codon included on +)"
    uint   reserved;         "itemRgb, always 0"
    int    blockCount;       "Exon count"
    int[blockCount] blockSizes;  "Exon sizes"
    int[blockCount] chromStarts; "Exon starts relative to chromStart"
    string geneId;           "Gene id (= name; one transcript per gene)"
    float  geneScore;        "Viterbi gene score over background (nats), unclipped"
    float  cdsProb;          "Mean p(CDS) over the CDS bases, stop codon included (f16-derived, may exceed 1 by ~1e-7)"
    float  startProb;        "START-head probability at the A of the start codon"
    uint   startExtended;    "1 if the START extension moved the start codon upstream"
    uint   startShift;       "bp the start codon moved (0 if not)"
    uint   utr5Blocks;       "5' UTR exon pieces (0: no 5' UTR)"
    uint   utr3Blocks;       "3' UTR exon pieces (0: no 3' UTR)"
    string utrDecoder;       "UTR decoders: 5' UTR decoder, 3' UTR DP attach"
    string release;          "ancestra version"
    string ckptSha256;       "sha256 of the checkpoint"
    string paramsSha256;     "sha256 of the sealed params file"
    )
"#;

/// BED12+9 autoSql for the optional alternative track; `{}` is the alt stamp.
const ALT_AUTOSQL: &str = r#"table ancestraAlt
"{}"
    (
    string chrom;            "Sequence name (2bit)"
    uint   chromStart;       "Transcript start, 0-based"
    uint   chromEnd;         "Transcript end, half-open"
    string name;             "<geneId>.<rank>"
    uint   score;            "round(1000 * p), half up, 0-1000"
    char[1] strand;          "+ or -"
    uint   thickStart;       "CDS start (stop codon included on -)"
    uint   thickEnd;         "CDS end (stop codon included on +)"
    uint   reserved;         "itemRgb, always 0"
    int    blockCount;       "Exon count"
    int[blockCount] blockSizes;  "Exon sizes"
    int[blockCount] chromStarts; "Exon starts relative to chromStart"
    string geneId;           "Name of the primary transcript (PREFIX.bed) this alternative belongs to"
    uint   rank;             "1..3 within the gene: p descending, then canonical identity"
    string sources;          "Candidate sources: ffbs, struct or ffbs,struct"
    float  p;                "Calibrated P(exact CDS match), from the model in alt.json"
    string utr;              "UTR sides copied from the primary: 5,3 | 5 | 3 | none"
    string release;          "ancestra version"
    string ckptSha256;       "sha256 of the checkpoint"
    string paramsSha256;     "sha256 of the sealed params file"
    string altSha256;        "sha256 of the sealed alt model file"
    )
"#;

fn alt_autosql(stamp: &str) -> String {
    ALT_AUTOSQL.replacen("{}", stamp, 1)
}

/// the autoSql with the run stamp as the table comment. The one `{}` in the
/// template is where S goes (`replacen`, so a stamp holding braces of its own
/// cannot touch the rest of the text).
fn autosql(stamp: &str) -> String {
    AUTOSQL.replacen("{}", stamp, 1)
}

/// the bigBed's text for one part — the 24 columns of the autoSql in
/// order. The first 12 are the BED12's own columns in its own spelling
/// (`bed12_line`: score, itemRgb 0 and the two block lists with their trailing
/// commas), so a bigBed's bed 1-12 columns are the BED12's bytes; the last 12
/// are the metadata columns, the part's own text. This is the "sorted
/// BED12+12 text" the bigBed carries and its read-back must equal.
fn bed24_line(p: &Part) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        bed12_line(p),
        p.gene_id,
        p.gene_score,
        p.cds_prob,
        p.start_prob,
        p.start_extended,
        p.start_shift,
        p.utr5_blocks,
        p.utr3_blocks,
        p.utr_decoder,
        p.release,
        p.ckpt_sha256,
        p.params_sha256,
    )
}

/// the bigBed rows: `order` already holds the bigBed order (chrom bytes,
/// chromStart, chromEnd, name), which the GXF files share. The caller's chrom
/// sizes (the 2bit's, `Genome.chrom_sizes`) come back as bigtools' u32 map,
/// checked against the rows: bigtools refuses a chrom it has no size for and a
/// start past the chrom's end, but not an end past it, and a length above
/// bigtools' u32 type is refused here. Every refusal names the part row, and
/// all of them run before any output is opened.
fn bigbed_rows<'a>(
    rows: &'a [Part],
    order: &[usize],
    chrom_sizes: &BTreeMap<String, u64>,
) -> io::Result<(Vec<&'a Part>, HashMap<String, u32>)> {
    if order.is_empty() {
        return Ok((Vec::new(), HashMap::new()));
    }
    let mut sizes: HashMap<String, u32> = HashMap::with_capacity(chrom_sizes.len());
    for (chrom, &len) in chrom_sizes {
        let len = u32::try_from(len).map_err(|_| {
            bad_input(format!(
                "chrom sizes: {} is {len} bp, above bigtools' u32 maximum {}",
                ascii_escape(chrom),
                u32::MAX
            ))
        })?;
        sizes.insert(chrom.clone(), len);
    }
    let out: Vec<&Part> = order.iter().map(|&i| &rows[i]).collect();
    for p in &out {
        let Some(&len) = sizes.get(&p.chrom) else {
            return Err(bad_input(format!(
                "{}: sequence {} is not in the chrom sizes (the 2bit's): the bigBed needs its length",
                p.src,
                ascii_escape(&p.chrom)
            )));
        };
        if u64::from(p.chrom_end) > u64::from(len) {
            return Err(bad_input(format!(
                "{}: transcript {} ends at {}, past the end of {} ({} bp)",
                p.src,
                ascii_escape(&p.name),
                p.chrom_end,
                ascii_escape(&p.chrom),
                len
            )));
        }
    }
    Ok((out, sizes))
}

/// the bigBed `PREFIX.bb` (BED12+12). The rows are already in the bigBed
/// order (chrom bytes, chromStart, chromEnd, name), which bigtools' own
/// out-of-order check re-verifies. bigtools 0.5.8's `BigBedWrite` writes the
/// sorted BED12+12 text with the autoSql; then the guarded
/// `definedFieldCount` patch and the re-open check run on the tmp file, so the
/// renamed output is the verified one. Atomic like every other output.
fn write_bigbed(
    path: &Path,
    rows: &[&Part],
    chrom_sizes: &HashMap<String, u32>,
    stamp: &str,
) -> io::Result<usize> {
    if rows.is_empty() {
        return write_empty_bigbed(path, stamp);
    }
    // the sorted BED12+12 text: what the bigBed carries, and what its read-back
    // must equal
    let mut text = String::new();
    for p in rows {
        text.push_str(&bed24_line(p));
        text.push('\n');
    }
    let n = atomic_write(path, |w, tmp| {
        // bigtools takes its writer by value: a dup of the same file (one
        // inode, one offset) leaves `atomic_write` the owner of the tmp path
        let out = w.get_ref().try_clone()?;
        let mut bbw = BigBedWrite::new(out, chrom_sizes.clone());
        bbw.options.inmemory = true;
        bbw.autosql = Some(autosql(stamp));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .build()
            .map_err(|e| bad_input(format!("the bigBed's tokio runtime: {e}")))?;
        // the order check is bigtools' own: chroms out of byte order or starts
        // out of order are refused by it, not only by the sort above
        bbw.write(
            BedParserStreamingIterator::from_bed_file(Cursor::new(text.into_bytes()), false),
            runtime,
        )
        .map_err(|e| bad_input(format!("bigtools BigBedWrite: {e}")))?;
        // the guarded patch, then the re-open: both read the finished file
        let mut f = OpenOptions::new().read(true).write(true).open(tmp)?;
        patch_defined_field_count(&mut f)?;
        drop(f);
        check_bigbed(tmp, rows.len(), stamp, COLS)?;
        Ok(rows.len())
    })?;
    Ok(n)
}

fn write_empty_bigbed(path: &Path, stamp: &str) -> io::Result<usize> {
    write_empty_bigbed_with(path, stamp, &autosql(stamp), COLS)
}

fn write_empty_bigbed_with(
    path: &Path,
    stamp: &str,
    sql: &str,
    fields: usize,
) -> io::Result<usize> {
    let bytes = empty_bigbed_bytes_with(sql, fields);
    atomic_write(path, |w, tmp| {
        w.write_all(&bytes)?;
        w.flush()?;
        check_bigbed(tmp, 0, stamp, fields)?;
        Ok(0)
    })
}

fn empty_bigbed_bytes_with(sql: &str, fields: usize) -> Vec<u8> {
    let autosql_offset = (64 + ZOOM_HEADER_BYTES) as u64;
    let summary_offset = autosql_offset + sql.len() as u64 + 1;
    let full_data_offset = summary_offset + SUMMARY_BYTES as u64;
    let chrom_tree = crate::trackout::empty_chrom_tree();
    let chrom_tree_offset = full_data_offset + 8;
    let full_index_offset = chrom_tree_offset + chrom_tree.len() as u64;
    let cir_tree = crate::trackout::empty_cir_tree(full_index_offset);

    let mut out = vec![0u8; 64];
    out[..4].copy_from_slice(&BIGBED_MAGIC);
    out[4..6].copy_from_slice(&4u16.to_le_bytes());
    out[8..16].copy_from_slice(&chrom_tree_offset.to_le_bytes());
    out[16..24].copy_from_slice(&full_data_offset.to_le_bytes());
    out[24..32].copy_from_slice(&full_index_offset.to_le_bytes());
    out[32..34].copy_from_slice(&(fields as u16).to_le_bytes());
    out[34..36].copy_from_slice(&12u16.to_le_bytes());
    out[36..44].copy_from_slice(&autosql_offset.to_le_bytes());
    out[44..52].copy_from_slice(&summary_offset.to_le_bytes());
    out.extend_from_slice(&[0u8; ZOOM_HEADER_BYTES]);
    out.extend_from_slice(sql.as_bytes());
    out.push(0);
    out.extend_from_slice(&[0u8; SUMMARY_BYTES]);
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&chrom_tree);
    out.extend_from_slice(&cir_tree);
    out.extend_from_slice(&BIGBED_MAGIC);
    out
}

/// The fields of a bigBed header this writer reads, in place. bigtools 0.5.8's
/// `write_info` lays the first 64 bytes out as the magic u32 at byte 0, the
/// version u16 at 4, zoomLevels u16 at 6, chromTreeOffset u64 at 8,
/// fullDataOffset u64 at 16 (where the total item count sits), fullIndexOffset
/// u64 at 24, fieldCount u16 at 32, definedFieldCount u16 at 34, autoSqlOffset
/// u64 at 36 and totalSummaryOffset u64 at 44.
#[derive(Debug, Clone, Copy)]
struct BbHeader {
    field_count: u16,
    defined_field_count: u16,
    full_data_offset: u64,
    auto_sql_offset: u64,
}

/// Reads a bigBed's header and checks its magic.
fn bb_header(path: &Path) -> io::Result<BbHeader> {
    let mut file = File::open(path).map_err(|e| bad_path(path, e))?;
    let mut h = [0u8; 44];
    file.read_exact(&mut h).map_err(|e| {
        bad_path(
            path,
            io::Error::new(e.kind(), format!("the bigBed header is shorter than 44 bytes: {e}")),
        )
    })?;
    if h[..4] != BIGBED_MAGIC {
        return Err(bad_input(format!(
            "{}: byte 0 is {:02X}{:02X}{:02X}{:02X}, not the bigBed magic 0x8789F2EB",
            path.display(),
            h[0],
            h[1],
            h[2],
            h[3]
        )));
    }
    Ok(BbHeader {
        field_count: u16::from_le_bytes([h[32], h[33]]),
        defined_field_count: u16::from_le_bytes([h[34], h[35]]),
        full_data_offset: u64::from_le_bytes(h[16..24].try_into().unwrap()),
        auto_sql_offset: u64::from_le_bytes(h[36..44].try_into().unwrap()),
    })
}

/// The autoSql of a bigBed: the NUL-terminated string at `offset`, read in
/// place (a multi-GB bigBed is never copied). Bounded read: an autoSql is a few
/// KB, so a file whose autoSql runs past 64 KiB without its NUL is refused.
fn bb_autosql(path: &Path, offset: u64) -> io::Result<String> {
    if offset == 0 {
        return Err(bad_input(format!(
            "{}: the bigBed holds no autoSql: there is no stamp",
            path.display()
        )));
    }
    let mut file = File::open(path).map_err(|e| bad_path(path, e))?;
    file.seek(SeekFrom::Start(offset)).map_err(|e| bad_path(path, e))?;
    let mut buf = Vec::new();
    (&file)
        .take(1 << 16)
        .read_to_end(&mut buf)
        .map_err(|e| bad_path(path, e))?;
    let end = buf.iter().position(|&b| b == 0).ok_or_else(|| {
        bad_input(format!(
            "{}: the autoSql is not NUL-terminated within 64 KiB",
            path.display()
        ))
    })?;
    buf.truncate(end);
    std::str::from_utf8(&buf)
        .map(str::to_owned)
        .map_err(|_| bad_input(format!("{}: the autoSql is not UTF-8 text", path.display())))
}

/// The table comment of an autoSql: its first quoted string (`table <name>`
/// then `"<comment>"`, autoSql.doc). A stamp holds no `"` (`check_stamp`), and
/// neither does any comment in the text, so the first quoted span is it.
fn bb_table_comment(path: &Path, autosql: &str) -> io::Result<String> {
    let start = autosql.find('"').ok_or_else(|| {
        bad_input(format!(
            "{}: the autoSql holds no table comment: there is no stamp",
            path.display()
        ))
    })?;
    let rest = &autosql[start + 1..];
    let end = rest.find('"').ok_or_else(|| {
        bad_input(format!(
            "{}: the autoSql's table comment is not a closed quoted string",
            path.display()
        ))
    })?;
    Ok(rest[..end].to_owned())
}

/// a bigBed's stamp is its autoSql table comment.
fn bigbed_stamp(path: &Path) -> io::Result<String> {
    let header = bb_header(path)?;
    let autosql = bb_autosql(path, header.auto_sql_offset)?;
    bb_table_comment(path, &autosql)
}

/// bigtools 0.5.8 writes `definedFieldCount = fieldCount` (24 primary fields or
/// 21 alt fields) and offers no separate option. Check the magic and supported
/// field count, patch the u16 at byte 34 to 12, then re-open the file.
fn patch_defined_field_count(file: &mut File) -> io::Result<()> {
    let mut h = [0u8; 36];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut h)?;
    let field_count = u16::from_le_bytes([h[32], h[33]]);
    if h[..4] != BIGBED_MAGIC || ![COLS as u16, ALT_COLS as u16].contains(&field_count) {
        return Err(bad_input(format!(
            "the bigBed header is not a supported bigtools 0.5.8 output: byte 0 is \
             {:02X}{:02X}{:02X}{:02X} and fieldCount (byte 32) is {field_count}: refusing to patch",
            h[0],
            h[1],
            h[2],
            h[3]
        )));
    }
    file.seek(SeekFrom::Start(34))?;
    file.write_all(&12u16.to_le_bytes())?;
    Ok(())
}

/// the re-open: the finished bigBed, read back. The patch must read back as
/// `definedFieldCount` 12 with its caller-specified `fieldCount`, the autoSql's
/// table comment must be the run stamp (finalize's re-read) and the header's
/// item count (`fullDataOffset`'s u64, what bigtools' `item_count` reads) must
/// be the row count.
fn check_bigbed(path: &Path, records: usize, stamp: &str, fields: usize) -> io::Result<()> {
    let header = bb_header(path)?;
    if header.field_count != fields as u16 || header.defined_field_count != 12 {
        return Err(bad_input(format!(
            "{}: the bigBed header holds fieldCount {} and definedFieldCount {}: the patch must read \
             back as {fields} and 12",
            path.display(),
            header.field_count,
            header.defined_field_count
        )));
    }
    if bb_table_comment(path, &bb_autosql(path, header.auto_sql_offset)?)? != stamp {
        return Err(bad_input(format!(
            "{}: the bigBed's autoSql table comment is not the run stamp",
            path.display()
        )));
    }
    let mut file = File::open(path).map_err(|e| bad_path(path, e))?;
    file.seek(SeekFrom::Start(header.full_data_offset))
        .map_err(|e| bad_path(path, e))?;
    let mut count = [0u8; 8];
    file.read_exact(&mut count).map_err(|e| bad_path(path, e))?;
    let items = u64::from_le_bytes(count);
    if items != records as u64 {
        return Err(bad_input(format!(
            "{}: the bigBed holds {items} records, not the {records} rows of its parts",
            path.display()
        )));
    }
    Ok(())
}
