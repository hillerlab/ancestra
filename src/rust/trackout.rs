//! Finalize the track run parts into bigWig (default) or bedGraph.
//!
//! `tracks::write_track_runs` streams one item's per-base runs into
//! `<item>.<feature>.<strand>.runs` parts (format version 2, `tracks.rs`). This module
//! gathers, for every requested (feature, strand), the caller's item bases
//! (`<base>.<feature>.<strand>.runs`, not a directory listing: a stale run of
//! an item outside the run is never opened) and streams them — in chrom byte
//! order, through `tracks::RunReader`, never a re-parse — into one output per pair:
//!
//! - **bigWig** via bigtools 0.5.8 `BigWigWrite` fed by
//!   `BedParserStreamingIterator::wrap_infallible_iter` (order checks on,
//!   default zooms). The value is `q as f32 / scale as f32`.
//! - **bedGraph** written directly: line 1 the stamp line `#S`, its only
//!   `#` line (the feature and the strand live in the file name and in the
//!   `PREFIX.prov.json` entry), then `chrom start end value` with the value as
//!   the exact decimal `q / scale` (scale 1000: `0.123`, `1.000`; scale 100:
//!   `0.12`).
//!
//! A track with no records is still written, as a valid empty file: the
//! bedGraph is the stamp line alone, and the bigWig is the file bigtools' own
//! writer produces for zero data (see `write_empty_bigwig`).
//!
//! Every write is atomic, like `out.rs`: a reserved unique tmp file,
//! `sync_all`, `rename`, and no partial output on any error. Every part of
//! every track is read and checked *before* any output is opened, so a bad
//! part, a bad scale or a bad name leaves the output directory untouched.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use bigtools::beddata::BedParserStreamingIterator;
use bigtools::{BigWigWrite, Value};
use sha2::{Digest, Sha256};
use tokio::runtime::Builder as RuntimeBuilder;

use crate::out::{
    ascii_escape, atomic_write, atomic_write_path, bad_input, bad_path, check_stamp, name_refusals,
    out_path,
};
use crate::tracks::{self, RunHeader, RunReader};

/// The strand tokens of a part name and of an output name.
const STRANDS: [(&str, char); 2] = [("plus", '+'), ("minus", '-')];

/// bigWig's magic `0x888FFC26`, little-endian: the empty track's header and the
/// trailer bigtools writes after the file body.
const BIGWIG_MAGIC: [u8; 4] = [0x26, 0xFC, 0x8F, 0x88];
/// The B+ tree magic of a bigWig's chrom tree (`0x78CA8C91`).
const CHROM_TREE_MAGIC: [u8; 4] = [0x91, 0x8C, 0xCA, 0x78];
/// The R-tree magic of a bigWig's index (`0x2468ACE0`).
const CIR_TREE_MAGIC: [u8; 4] = [0xE0, 0xAC, 0x68, 0x24];
/// Chrom tree bytes bigtools writes for zero chroms: the 32-byte header
/// (`write_chrom_tree`'s fields) and the 4-byte empty leaf.
const EMPTY_CHROM_TREE_LEN: usize = 36;
/// CIR-tree bytes bigtools writes for zero sections: the 48-byte header and the
/// 4-byte empty root leaf.
const EMPTY_CIR_TREE_LEN: usize = 52;
/// The zoom headers bigtools always reserves (`MAX_ZOOM_LEVELS * 24`).
const ZOOM_HEADER_BYTES: usize = 10 * 24;
/// The blank total summary bigtools reserves (`write_pre`'s `[0; 40]`).
const SUMMARY_BYTES: usize = 40;

/// One output's provenance entry for `PREFIX.prov.json` (for a track also
/// its feature and strand).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackOutput {
    /// Data records in the file (0 for an empty track).
    pub records: usize,
    /// sha256 of the written file: a bigWig has no text field, so this is what
    /// binds it to S.
    pub sha256: String,
    pub feature: String,
    pub strand: char,
}

/// `--track-type`: exactly one value, bigwig the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackKind {
    BigWig,
    BedGraph,
}

impl TrackKind {
    fn parse(kind: &str) -> io::Result<Self> {
        match kind {
            "bigwig" => Ok(Self::BigWig),
            "bedgraph" => Ok(Self::BedGraph),
            other => Err(bad_input(format!(
                "track type must be bigwig or bedgraph, got '{other}'"
            ))),
        }
    }

    /// The output's extension (`PREFIX.<feature>.<plus|minus>.bw` or
    /// `.bedGraph`).
    fn extension(self) -> &'static str {
        match self {
            Self::BigWig => "bw",
            Self::BedGraph => "bedGraph",
        }
    }
}

/// One part of a track: where it is and what its header says (its `RunHeader`
/// was validated by `RunReader::open`).
#[derive(Clone, Debug)]
struct PartRef {
    path: PathBuf,
    chrom: String,
    item_start: u32,
    item_end: u32,
    scale: u32,
}

/// The parts of one (feature, strand) track, checked and ordered by chrom bytes.
#[derive(Clone, Debug)]
struct Track {
    feature: String,
    strand: char,
    /// Exactly one part per chrom (one item per chrom, so two parts of one
    /// chrom are refused — which also refuses any overlap), in chrom byte
    /// order: the order the records are streamed in.
    parts: Vec<PartRef>,
    /// The one scale of every part (the precision is sealed in
    /// params.json). 0 for a track with no parts, where no value is written.
    scale: u32,
    /// Records over all parts, counted by the validation pass.
    records: usize,
}

/// Streams one track's records, in chrom byte order, as bigtools' infallible
/// bedGraph values (`wrap_infallible_iter`). The record layout was already
/// checked by the validation pass, so a failure here can only be I/O: it is
/// recorded in `error` and ends the stream, and `write_bigwig` fails loudly on
/// it — so a read error never leaves a truncated output behind.
struct TrackValues<'a> {
    parts: &'a [PartRef],
    scale: u32,
    next_part: usize,
    reader: Option<RunReader>,
    error: &'a RefCell<Option<io::Error>>,
}

impl<'a> Iterator for TrackValues<'a> {
    type Item = (&'a str, Value);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let Some(reader) = self.reader.as_mut() else {
                // the next part, in chrom byte order
                let part = self.parts.get(self.next_part)?;
                self.next_part += 1;
                match RunReader::open(&part.path) {
                    Ok(reader) => self.reader = Some(reader),
                    Err(e) => {
                        *self.error.borrow_mut() = Some(e);
                        return None;
                    }
                }
                continue;
            };
            match reader.next() {
                Some(Ok((start, end, q))) => {
                    let index = self.next_part - 1;
                    return Some((
                        self.parts[index].chrom.as_str(),
                        Value {
                            start,
                            end,
                            value: q as f32 / self.scale as f32,
                        },
                    ));
                }
                Some(Err(e)) => {
                    *self.error.borrow_mut() = Some(e);
                    self.reader = None;
                    return None;
                }
                None => self.reader = None,
            }
        }
    }
}

/// Gathers, checks and reads every part of one (feature, strand) track. The
/// parts are `<item>.<feature>.<strand>.runs` (tracks.rs `write_track_runs`),
/// and each part's header must match its file name. A track with no part is
/// still valid (and still written, empty).
fn gather_track(
    parts_dir: &Path,
    feature: &str,
    strand_token: &str,
    strand: char,
    chrom_sizes: &HashMap<String, u32>,
    item_bases: &[PathBuf],
) -> io::Result<Track> {
    let mut parts: Vec<PartRef> = Vec::new();
    for base in item_bases {
        // The parent names the run's items. A directory listing would also take
        // a stale `<other>.<feature>.<strand>.runs` left from an earlier run.
        // The base's own file name is one component inside the canonical
        // parts_dir: `.` / `..` and a separator are refused before anything is
        // opened, and so is a parent that is not that directory (`base/..`).
        let name = base.file_name().unwrap_or_default();
        if name.is_empty() || name == "." || name == ".." {
            return Err(bad_input(format!(
                "{}: item base file name '{}' is refused",
                base.display(),
                name.to_string_lossy()
            )));
        }
        if name.to_string_lossy().chars().any(std::path::is_separator) {
            return Err(bad_input(format!(
                "{}: item base file name contains a path separator",
                base.display()
            )));
        }
        let parent = base
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent = parent.canonicalize().map_err(|e| {
            bad_input(format!(
                "{}: item base is outside {}: {e}",
                base.display(),
                parts_dir.display()
            ))
        })?;
        if parent != parts_dir {
            return Err(bad_input(format!(
                "{}: item base is outside {}",
                base.display(),
                parts_dir.display()
            )));
        }
        let path = PathBuf::from(format!(
            "{}.{feature}.{strand_token}.runs",
            base.display()
        ));
        let meta = match std::fs::symlink_metadata(&path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(bad_input(format!(
                    "{}: no run part for {feature} on strand {strand}",
                    path.display()
                )));
            }
            Err(e) => return Err(bad_path(&path, e)),
        };
        if meta.file_type().is_symlink() {
            return Err(bad_input(format!(
                "{}: run part {} is a symlink",
                base.display(),
                path.display()
            )));
        }
        if !meta.is_file() {
            return Err(bad_input(format!(
                "{}: no run part for {feature} on strand {strand}",
                path.display()
            )));
        }
        let canon = path.canonicalize().map_err(|e| bad_path(&path, e))?;
        if !canon.starts_with(parts_dir) {
            return Err(bad_input(format!(
                "{}: run part {} is outside {}",
                base.display(),
                canon.display(),
                parts_dir.display()
            )));
        }
        // the reader validates the version-2 header (and names the part on any error)
        let reader = RunReader::open(&path)?;
        let RunHeader {
            chrom,
            item_start,
            item_end,
            scale,
            feature: header_feature,
            strand: header_strand,
        } = reader.header().clone();
        if header_feature != feature || header_strand != strand {
            return Err(bad_input(format!(
                "{}: the part's header says {} on strand {}, but its file name says {feature} on {strand}",
                path.display(),
                header_feature,
                header_strand
            )));
        }
        parts.push(PartRef {
            path,
            chrom,
            item_start,
            item_end,
            scale,
        });
    }

    // one scale for the whole track: the parts of one track share the sealed
    // precision, and a mix would make the written value ambiguous
    let scale = parts.first().map_or(0, |p| p.scale);
    if let Some(first) = parts.first() {
        for part in &parts[1..] {
            if part.scale != scale {
                return Err(bad_input(format!(
                    "{}: scale {} does not match scale {} of {}: one track has one scale",
                    part.path.display(),
                    part.scale,
                    scale,
                    first.path.display()
                )));
            }
        }
    }

    // one item per chrom: two parts of one chrom are refused, which also
    // refuses any overlap of two items of one chrom
    let mut by_chrom: BTreeMap<&str, &PartRef> = BTreeMap::new();
    for part in &parts {
        if let Some(prev) = by_chrom.insert(part.chrom.as_str(), part) {
            return Err(bad_input(format!(
                "sequence {} is in parts {} and {}: one item per sequence, so one part per chrom",
                ascii_escape(&prev.chrom),
                prev.path.display(),
                part.path.display()
            )));
        }
    }
    // chrom byte order: the order the records are streamed in, and the order
    // bigtools' order checks require
    parts.sort_by(|a, b| a.chrom.as_bytes().cmp(b.chrom.as_bytes()));

    // the chroms of a track are written verbatim (into the chrom tree and into
    // the bedGraph's first column), so the name rule applies here too (gtf
    // is never requested for tracks)
    let names: Vec<String> = parts.iter().map(|p| p.chrom.clone()).collect();
    let refused = name_refusals(&names, false);
    if !refused.is_empty() {
        return Err(bad_input(format!(
            "the run's sequence names are refused: {}",
            refused
                .iter()
                .map(|(_, reason)| reason.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        )));
    }
    // every item must sit inside its sequence: the 2bit header's sizes are the
    // only source of a chrom's length, so a part beyond it means a part from
    // another run or a wrong chrom_sizes
    for part in &parts {
        let Some(size) = chrom_sizes.get(&part.chrom) else {
            return Err(bad_input(format!(
                "{}: sequence {} is not in the chrom sizes: the 2bit header provides every sequence of the run",
                part.path.display(),
                ascii_escape(&part.chrom)
            )));
        };
        if u64::from(part.item_end) > u64::from(*size) {
            return Err(bad_input(format!(
                "{}: item {}:[{}, {}) ends past the {size} base sequence",
                part.path.display(),
                ascii_escape(&part.chrom),
                part.item_start,
                part.item_end
            )));
        }
    }

    // read every record of every part before any output is opened (out.rs's
    // discipline): RunReader checks each record's order and span as it is read
    let mut records = 0usize;
    for part in &parts {
        for record in RunReader::open(&part.path)? {
            record?;
            records += 1;
        }
    }

    Ok(Track {
        feature: feature.to_owned(),
        strand,
        parts,
        scale,
        records,
    })
}

/// bigWig: bigtools `BigWigWrite` fed by the parts in chrom byte order,
/// through `BedParserStreamingIterator::wrap_infallible_iter` (order checks on,
/// `allow_out_of_order_chroms = false`; default zooms). `threads` (0: the
/// runtime's own default, all cores) sets the tokio worker count the CLI's
/// `--threads` means. A track with no records is written as the valid empty
/// file bigtools' own writer would produce for zero data.
fn write_bigwig(
    path: &Path,
    track: &Track,
    chrom_sizes: &HashMap<String, u32>,
    threads: usize,
) -> io::Result<usize> {
    if track.records == 0 {
        return write_empty_bigwig(path);
    }
    let mut builder = RuntimeBuilder::new_multi_thread();
    if threads > 0 {
        builder.worker_threads(threads);
    }
    // bigtools consumes the runtime, so each track builds its own; that costs
    // only its worker threads
    let runtime = builder.build().map_err(|e| {
        bad_input(format!(
            "the tokio runtime for {} failed: {e}",
            path.display()
        ))
    })?;
    let error: RefCell<Option<io::Error>> = RefCell::new(None);
    let source = BedParserStreamingIterator::wrap_infallible_iter(
        TrackValues {
            parts: &track.parts,
            scale: track.scale,
            next_part: 0,
            reader: None,
            error: &error,
        },
        false,
    );
    let mut writer =
        BigWigWrite::create_file(path, chrom_sizes.clone()).map_err(|e| bad_path(path, e))?;
    writer.options.inmemory = true;
    writer.write(source, runtime).map_err(|e| {
        bad_path(
            path,
            io::Error::new(io::ErrorKind::InvalidData, e.to_string()),
        )
    })?;
    // a read error mid-stream truncated the file: never publish it
    if let Some(error) = error.into_inner() {
        return Err(error);
    }
    Ok(track.records)
}

/// bedGraph: line 1 the stamp line `#S`, its only `#` line (the feature and
/// the strand are in the file name and in the `PREFIX.prov.json` entry; the
/// precision is sealed in params.json), then `chrom start end value` — the
/// exact decimal `q / scale` (scale 1000 `0.123`, `1.000`; scale 100
/// `0.12`). bigtools' bedGraph parser has no header skip, so the text is
/// written directly (the bigWig→bedGraph conversion is the round-trip check).
fn write_bedgraph(path: &Path, track: &Track, stamp: &str) -> io::Result<usize> {
    // a bedGraph value is written from the integers alone, so the scale must be
    // a power of ten: as many decimals as the scale has zeros (an empty track
    // has no part, hence no scale, and no value to write)
    let decimals = if track.records == 0 {
        0
    } else {
        decimal_places(track.scale)?
    };
    let scale = track.scale;
    atomic_write(path, |writer: &mut std::io::BufWriter<File>, _tmp: &Path| {
        writeln!(writer, "#{stamp}")?;
        for part in &track.parts {
            for record in RunReader::open(&part.path)? {
                let (start, end, q) = record?;
                let (whole, fraction) = (u32::from(q) / scale, u32::from(q) % scale);
                if decimals == 0 {
                    writeln!(writer, "{}\t{start}\t{end}\t{whole}", part.chrom)?;
                } else {
                    writeln!(
                        writer,
                        "{}\t{start}\t{end}\t{whole}.{fraction:0decimals$}",
                        part.chrom
                    )?;
                }
            }
        }
        Ok(track.records)
    })
}

/// The decimals of a bedGraph value: the zeros of the scale. A scale that
/// is not 10^k has no exact fixed-point text, so it is refused rather than
/// rounded.
fn decimal_places(scale: u32) -> io::Result<usize> {
    let mut decimals = 0usize;
    let mut rest = scale;
    while rest > 0 && rest.is_multiple_of(10) {
        rest /= 10;
        decimals += 1;
    }
    if rest != 1 {
        return Err(bad_input(format!(
            "track scale {scale} is not a power of ten: the bedGraph value text is exact only for a decimal scale"
        )));
    }
    Ok(decimals)
}

/// The 0-item chrom tree bigtools' `write_chrom_tree` writes for zero data
/// chroms: the 32-byte B+ tree header (magic, block size `max(256, 0)`, the
/// longest key 0, the item size 8, the item count 0, reserved) and the 4-byte
/// empty leaf.
pub(crate) fn empty_chrom_tree() -> [u8; EMPTY_CHROM_TREE_LEN] {
    let mut tree = Vec::with_capacity(EMPTY_CHROM_TREE_LEN);
    tree.extend_from_slice(&CHROM_TREE_MAGIC);
    tree.extend_from_slice(&256u32.to_le_bytes()); // block size
    tree.extend_from_slice(&0u32.to_le_bytes()); // the longest key: no chrom
    tree.extend_from_slice(&8u32.to_le_bytes()); // id + size
    tree.extend_from_slice(&0u64.to_le_bytes()); // item count
    tree.extend_from_slice(&0u64.to_le_bytes()); // reserved
    tree.extend_from_slice(&[1, 0]); // isLeaf, reserved
    tree.extend_from_slice(&0u16.to_le_bytes()); // items in the leaf
    tree.try_into().expect("the empty chrom tree's length")
}

/// The 0-item cir-tree root bigtools' `write_rtreeindex` writes for zero
/// sections: its 48-byte header (magic, block size, the section count 0, the
/// root's bounding box, the end of the primary data, the items per slot,
/// reserved) and the 4-byte empty leaf.
pub(crate) fn empty_cir_tree(end_of_data: u64) -> [u8; EMPTY_CIR_TREE_LEN] {
    let mut tree = Vec::with_capacity(EMPTY_CIR_TREE_LEN);
    tree.extend_from_slice(&CIR_TREE_MAGIC);
    tree.extend_from_slice(&256u32.to_le_bytes()); // block size
    tree.extend_from_slice(&0u64.to_le_bytes()); // section count
    tree.extend_from_slice(&0u32.to_le_bytes()); // start chrom
    tree.extend_from_slice(&0u32.to_le_bytes()); // start base
    tree.extend_from_slice(&0u32.to_le_bytes()); // end chrom
    tree.extend_from_slice(&0u32.to_le_bytes()); // end base
    tree.extend_from_slice(&end_of_data.to_le_bytes()); // end of the primary data
    tree.extend_from_slice(&1024u32.to_le_bytes()); // items per slot
    tree.extend_from_slice(&0u32.to_le_bytes()); // reserved
    tree.extend_from_slice(&[1, 0]); // isLeaf, reserved
    tree.extend_from_slice(&0u16.to_le_bytes()); // items in the leaf
    tree.try_into().expect("the empty cir tree's length")
}

/// A track with no records is still written, and bigtools cannot write it:
/// its `BedParserStreamingIterator` refuses an empty bedGraph ("Input bedGraph
/// is empty.") and a source that yields no values deadlocks in
/// `bbiwrite::get_rtreeindex`, whose node build-up loop never reaches its
/// single-node base case with zero sections. So the file is written here in the
/// layout bigtools' own writer produces for zero data (`write_pre`,
/// `write_mid` and `write_info` of `bbiwrite.rs`): the blank 64-byte header, the
/// blank zoom headers, the blank total summary, the u64 data count 0 at
/// fullDataOffset, a 0-item chrom tree, a 0-item cir-tree root and the trailing
/// magic — 444 bytes. Checked: bigtools' reader opens it as a bigWig with no
/// chrom and a zero summary (`test_empty_track_is_a_valid_empty_file`), and
/// kent's `bigWigToBedGraph` converts it to nothing. kent's `validateFiles`
/// refuses it, because a bigWig with no chrom holds no chromosome list to
/// check — the only kent tool that does; `bigWigInfo` and `bigWigToBedGraph`
/// read it.
fn write_empty_bigwig(path: &Path) -> io::Result<usize> {
    let mut file = File::create(path).map_err(|e| bad_path(path, e))?;
    file.write_all(&empty_bigwig_bytes())
        .map_err(|e| bad_path(path, e))?;
    file.sync_all().map_err(|e| bad_path(path, e))?;
    Ok(0)
}

/// The bytes of the empty bigWig (see `write_empty_bigwig`).
fn empty_bigwig_bytes() -> Vec<u8> {
    // `write_pre`'s body, in its order: the blank total summary, then the u64
    // data count at fullDataOffset
    let mut body = Vec::with_capacity(444);
    body.extend_from_slice(&[0u8; SUMMARY_BYTES]);
    body.extend_from_slice(&0u64.to_le_bytes()); // no data sections
    let chrom_tree_offset = u64::try_from(64 + ZOOM_HEADER_BYTES + body.len()).unwrap();
    body.extend_from_slice(&empty_chrom_tree());
    let full_index_offset = chrom_tree_offset + EMPTY_CHROM_TREE_LEN as u64;
    body.extend_from_slice(&empty_cir_tree(full_index_offset));

    let mut out = vec![0u8; 64];
    out[0..4].copy_from_slice(&BIGWIG_MAGIC);
    out[4..6].copy_from_slice(&4u16.to_le_bytes()); // version
    out[6..8].copy_from_slice(&0u16.to_le_bytes()); // zoom levels: none
    out[8..16].copy_from_slice(&chrom_tree_offset.to_le_bytes());
    let full_data_offset = (64 + ZOOM_HEADER_BYTES + SUMMARY_BYTES) as u64;
    out[16..24].copy_from_slice(&full_data_offset.to_le_bytes());
    out[24..32].copy_from_slice(&full_index_offset.to_le_bytes());
    // 32..34 fieldCount and 34..36 definedFieldCount stay 0: a bigWig carries
    // no bed fields
    // 36..44 autoSqlOffset stays 0: a bigWig has no autoSql
    let total_summary_offset = (64 + ZOOM_HEADER_BYTES) as u64;
    out[44..52].copy_from_slice(&total_summary_offset.to_le_bytes());
    // 52..56 uncompressBufSize stays 0: the empty file is not compressed
    // 56..64 reserved stays 0
    out.extend_from_slice(&[0u8; ZOOM_HEADER_BYTES]); // the blank zoom headers
    out.extend_from_slice(&body);
    out.extend_from_slice(&BIGWIG_MAGIC); // the trailer
    out
}

/// The sha256 of a written output, for `PREFIX.prov.json`.
fn sha256_of(path: &Path) -> io::Result<String> {
    let mut file = File::open(path).map_err(|e| bad_path(path, e))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|e| bad_path(path, e))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Finalizes the track run parts of `item_bases` (paths without the
/// `.<feature>.<strand>.runs` suffix; one per run item) into
/// `PREFIX.<feature>.<plus|minus>.bw` (default) or `.bedGraph`, for every
/// requested feature on both strands, and returns each output's path with its
/// provenance entry. `parts_dir` is the directory those bases live in.
///
/// `chrom_sizes` is the 2bit header's map (names verbatim): a length above
/// bigtools' u32 is refused, every part's chrom must be in it and every
/// item must sit inside its sequence. `kind` is `bigwig` or `bedgraph`,
/// `threads` the tokio worker count (0: the runtime default). Every part is
/// read and checked before any output is opened, so a failure anywhere leaves
/// no output behind.
#[allow(clippy::too_many_arguments)] // the parent's list, added to the release arguments
pub fn finalize_tracks(
    parts_dir: &Path,
    prefix: &str,
    chrom_sizes: &BTreeMap<String, u64>,
    features: &[String],
    kind: &str,
    stamp: &str,
    threads: usize,
    item_bases: &[PathBuf],
) -> io::Result<Vec<(PathBuf, TrackOutput)>> {
    if !parts_dir.is_dir() {
        return Err(bad_path(
            parts_dir,
            io::Error::new(io::ErrorKind::NotFound, "no parts directory"),
        ));
    }
    // Once: every base and every run part is contained in this directory, not
    // in a lexical prefix of it (`..`, a symlink).
    let parts_dir = parts_dir.canonicalize().map_err(|e| bad_path(parts_dir, e))?;
    check_stamp(stamp)?;
    let kind = TrackKind::parse(kind)?;
    let selected = tracks::feature_names(features)?;
    // bigtools' chrom tree and sections are u32
    let mut sizes: HashMap<String, u32> = HashMap::with_capacity(chrom_sizes.len());
    for (name, size) in chrom_sizes {
        let size32 = u32::try_from(*size).map_err(|_| {
            bad_input(format!(
                "sequence {} is {size} bases long, above bigtools' u32 limit",
                ascii_escape(name)
            ))
        })?;
        sizes.insert(name.clone(), size32);
    }

    // pass 1: gather, check and read every part of every track
    let mut tracks: Vec<Track> = Vec::with_capacity(selected.len() * STRANDS.len());
    for feature in &selected {
        for (strand_token, strand) in STRANDS {
            tracks.push(gather_track(
                &parts_dir,
                feature,
                strand_token,
                strand,
                &sizes,
                item_bases,
            )?);
        }
    }
    // a bedGraph's value text is exact only for a decimal scale: checked
    // here, before any output is opened
    if kind == TrackKind::BedGraph {
        for track in &tracks {
            if track.records > 0 {
                decimal_places(track.scale)?;
            }
        }
    }

    // pass 2: write, atomically, one output per (feature, strand)
    let mut written = Vec::with_capacity(tracks.len());
    for track in &tracks {
        let path = out_path(
            prefix,
            &format!(
                "{}.{}.{}",
                track.feature,
                if track.strand == '+' { "plus" } else { "minus" },
                kind.extension()
            ),
        );
        let records = match kind {
            TrackKind::BigWig => {
                atomic_write_path(&path, |tmp| write_bigwig(tmp, track, &sizes, threads))?
            }
            TrackKind::BedGraph => write_bedgraph(&path, track, stamp)?,
        };
        let sha256 = sha256_of(&path)?;
        written.push((
            path,
            TrackOutput {
                records,
                sha256,
                feature: track.feature.clone(),
                strand: track.strand,
            },
        ));
    }
    Ok(written)
}
