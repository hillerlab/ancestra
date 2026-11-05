//! Streamed per-base probability runs.
//!
//! File format version 2 (all integers little-endian): `APRRUNS\0`, u16 version,
//! u32 scale, u64 item start, u64 item end, u32 chrom-byte length, u16
//! feature-byte length, u8 strand (0 `+`, 1 `-`), u8 reserved (zero), then
//! UTF-8 chrom and ASCII feature bytes. Records follow as 10-byte `(u32 start,
//! u32 end, u16 q)` tuples with absolute chrom coordinates and half-open bounds.
//! Part presence does not mark item completion; the item's `.done` record does.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

const MAGIC: &[u8; 8] = b"APRRUNS\0";
const VERSION: u16 = 2;
const HEADER_LEN: usize = 38;
const RECORD_LEN: usize = 10;
const MAX_CHROM_BYTES: usize = 238;
static EXP16: OnceLock<Vec<f64>> = OnceLock::new();
static IDENTITY16: OnceLock<Vec<f64>> = OnceLock::new();
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Exact binary16-to-binary64 conversion, including subnormals and non-finites.
fn f16_value(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;
    match exponent {
        0 if fraction == 0 => sign * 0.0,
        0 => sign * (fraction as f64) * 2f64.powi(-24),
        0x1f if fraction == 0 => sign * f64::INFINITY,
        0x1f => f64::NAN,
        e => sign * ((1024 + fraction) as f64) * 2f64.powi(e - 25),
    }
}

fn identity16() -> &'static [f64] {
    IDENTITY16.get_or_init(|| (0..=u16::MAX).map(f16_value).collect())
}

/// Shared f64 exp table indexed by the raw f16 bits (also used for BED scores).
pub fn exp16_table() -> &'static [f64] {
    EXP16.get_or_init(|| identity16().iter().map(|x| x.exp()).collect())
}

/// f64 identity table indexed by the raw f16 bits.
pub fn identity16_table() -> &'static [f64] {
    identity16()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Feature {
    Cds,
    Exon,
    FiveUtr,
    ThreeUtr,
    Donor,
    Acceptor,
    Start,
    Stop,
}

const FEATURES: [Feature; 8] = [
    Feature::Cds,
    Feature::Exon,
    Feature::FiveUtr,
    Feature::ThreeUtr,
    Feature::Donor,
    Feature::Acceptor,
    Feature::Start,
    Feature::Stop,
];

impl Feature {
    fn name(self) -> &'static str {
        match self {
            Self::Cds => "cds",
            Self::Exon => "exon",
            Self::FiveUtr => "fiveutr",
            Self::ThreeUtr => "threeutr",
            Self::Donor => "donor",
            Self::Acceptor => "acceptor",
            Self::Start => "start",
            Self::Stop => "stop",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Cds => 0,
            Self::Exon => 1,
            Self::FiveUtr => 2,
            Self::ThreeUtr => 3,
            Self::Donor => 4,
            Self::Acceptor => 5,
            Self::Start => 6,
            Self::Stop => 7,
        }
    }

    fn parse(name: &str) -> Option<Self> {
        FEATURES.into_iter().find(|f| f.name() == name)
    }
}

#[derive(Clone, Copy)]
enum BaseFeature {
    Cds,
    Exon,
    FiveUtr,
    ThreeUtr,
    Donor,
    Acceptor,
}

/// The run-part reader and the finalizer both need the same feature selection
/// (`all`, `utr`, `codon` and the eight names, in canonical order), so it lives here.
fn select_features(names: &[String]) -> io::Result<Vec<Feature>> {
    let mut selected = [false; 8];
    for name in names {
        match name.as_str() {
            "all" => selected.fill(true),
            "utr" => {
                selected[Feature::FiveUtr.index()] = true;
                selected[Feature::ThreeUtr.index()] = true;
            }
            "codon" => {
                selected[Feature::Start.index()] = true;
                selected[Feature::Stop.index()] = true;
            }
            _ => {
                let Some(feature) = Feature::parse(name) else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unknown track feature: {name}"),
                    ));
                };
                selected[feature.index()] = true;
            }
        }
    }
    if !selected.iter().any(|&v| v) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "features must not be empty",
        ));
    }
    Ok(FEATURES
        .into_iter()
        .filter(|f| selected[f.index()])
        .collect())
}

/// The canonical feature names of `names` (`all` / `utr` / `codon` expanded), in the
/// order `write_track_runs` writes them: the finalize path names a track
/// exactly as the run-part path spells it.
pub(crate) fn feature_names(names: &[String]) -> io::Result<Vec<String>> {
    Ok(select_features(names)?.iter().map(|f| f.name().to_owned()).collect())
}

fn finite_input(
    bits: u16,
    channel: &str,
    chrom: &str,
    item_start: u64,
    pos: u64,
    strand: usize,
) -> io::Result<()> {
    if bits & 0x7c00 == 0x7c00 {
        let sign = if strand == 0 { "+" } else { "-" };
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("non-finite {channel} input for item {chrom}:{item_start} at {chrom}:{pos} ({sign})"),
        ));
    }
    Ok(())
}

fn quantize(p: f64, scale: u32) -> u16 {
    (p.clamp(0.0, 1.0) * f64::from(scale) + 0.5).floor() as u16
}

struct TempRun {
    path: PathBuf,
    tmp_path: PathBuf,
    writer: Option<BufWriter<File>>,
    strand: usize,
    active: Option<(u64, u16)>,
    n_records: usize,
    committed: bool,
}

impl TempRun {
    fn create(
        path: PathBuf,
        chrom: &str,
        item_start: u64,
        item_end: u64,
        scale: u32,
        feature: Feature,
        strand: usize,
    ) -> io::Result<Self> {
        let chrom_bytes = chrom.as_bytes();
        let feature_bytes = feature.name().as_bytes();
        let chrom_len = u32::try_from(chrom_bytes.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "chrom name is too long"))?;
        let feature_len = u16::try_from(feature_bytes.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "feature name is too long"))?;
        let (tmp_path, file) = (0..64)
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
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "could not reserve track temp file",
                )
            })??;
        let mut writer = BufWriter::new(file);
        let header_result = (|| {
            writer.write_all(MAGIC)?;
            writer.write_all(&VERSION.to_le_bytes())?;
            writer.write_all(&scale.to_le_bytes())?;
            writer.write_all(&item_start.to_le_bytes())?;
            writer.write_all(&item_end.to_le_bytes())?;
            writer.write_all(&chrom_len.to_le_bytes())?;
            writer.write_all(&feature_len.to_le_bytes())?;
            writer.write_all(&[strand as u8, 0])?;
            writer.write_all(chrom_bytes)?;
            writer.write_all(feature_bytes)
        })();
        if let Err(error) = header_result {
            drop(writer);
            let _ = fs::remove_file(&tmp_path);
            return Err(error);
        }
        Ok(Self {
            path,
            tmp_path,
            writer: Some(writer),
            strand,
            active: None,
            n_records: 0,
            committed: false,
        })
    }

    fn push(&mut self, pos: u64, q: u16) -> io::Result<()> {
        match (self.active, q) {
            (Some((_, active_q)), q) if q == active_q => {}
            (Some(_), 0) => self.flush_run(pos)?,
            (Some(_), q) => {
                self.flush_run(pos)?;
                self.active = Some((pos, q));
            }
            (None, 0) => {}
            (None, q) => self.active = Some((pos, q)),
        }
        Ok(())
    }

    fn flush_run(&mut self, end: u64) -> io::Result<()> {
        let Some((start, q)) = self.active.take() else {
            return Ok(());
        };
        let writer = self.writer.as_mut().expect("uncommitted run has a writer");
        writer.write_all(&(start as u32).to_le_bytes())?;
        writer.write_all(&(end as u32).to_le_bytes())?;
        writer.write_all(&q.to_le_bytes())?;
        self.n_records += 1;
        Ok(())
    }

    fn finish(&mut self, item_end: u64) -> io::Result<()> {
        self.flush_run(item_end)?;
        let mut writer = self.writer.take().expect("run is finished once");
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&self.tmp_path, &self.path)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for TempRun {
    fn drop(&mut self) {
        if !self.committed {
            self.writer.take();
            let _ = fs::remove_file(&self.tmp_path);
        }
    }
}

struct BaseOutput {
    run: TempRun,
    feature: BaseFeature,
}

pub struct TrackInput<'a> {
    pub chrom: &'a str,
    pub item_start: u64,
    pub la1: &'a [u16],
    pub utr7: &'a [u16],
    pub c: &'a [u16],
    pub feature_names: &'a [String],
    pub scale: u32,
}

/// Write both strand planes for each selected feature without buffering item runs.
pub fn write_track_runs(
    prefix: &Path,
    input: TrackInput<'_>,
) -> io::Result<BTreeMap<PathBuf, usize>> {
    let TrackInput {
        chrom,
        item_start,
        la1,
        utr7,
        c,
        feature_names,
        scale,
    } = input;
    let features = select_features(feature_names)?;
    if chrom.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "chrom must not be empty",
        ));
    }
    if chrom.len() > MAX_CHROM_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("chrom name must be at most {MAX_CHROM_BYTES} bytes"),
        ));
    }
    if scale == 0 || scale > u16::MAX as u32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("item {chrom}:{item_start}: scale must be in 1..=65535"),
        ));
    }
    if !la1.len().is_multiple_of(2) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "la1 must have shape [2, L]",
        ));
    }
    let length = la1.len() / 2;
    if length == 0 || utr7.len() != 14 * length || c.len() != 8 * length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "la1, utr7 and c must have shapes [2, L], [2, 7, L] and [2, 4, L] with L > 0",
        ));
    }
    let item_len = u64::try_from(length)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "item is too long"))?;
    let item_end = item_start.checked_add(item_len).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("item {chrom}:{item_start} end coordinate overflows u64"),
        )
    })?;
    if item_end > u32::MAX as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("item {chrom}:{item_start} ends at {item_end}, above u32::MAX"),
        ));
    }

    let need_cds = features
        .iter()
        .any(|f| matches!(f, Feature::Cds | Feature::Exon));
    let need_utr1 = features
        .iter()
        .any(|f| matches!(f, Feature::FiveUtr | Feature::Exon));
    let need_utr3 = features
        .iter()
        .any(|f| matches!(f, Feature::ThreeUtr | Feature::Exon));
    let need_donor = features.contains(&Feature::Donor);
    let need_acceptor = features.contains(&Feature::Acceptor);
    let codon_channels: Vec<_> = features
        .iter()
        .filter_map(|feature| match *feature {
            Feature::Start => Some((2, "c[:,2]")),
            Feature::Stop => Some((3, "c[:,3]")),
            _ => None,
        })
        .collect();
    let mut outputs = Vec::with_capacity(features.len() * 2);
    let mut codon_outputs = Vec::with_capacity(codon_channels.len() * 2);
    for feature in &features {
        for strand in 0..2 {
            let mut name = prefix.as_os_str().to_os_string();
            name.push(format!(
                ".{}.{}.runs",
                feature.name(),
                if strand == 0 { "plus" } else { "minus" }
            ));
            let run = TempRun::create(
                PathBuf::from(name),
                chrom,
                item_start,
                item_end,
                scale,
                *feature,
                strand,
            )?;
            match *feature {
                Feature::Cds => outputs.push(BaseOutput {
                    run,
                    feature: BaseFeature::Cds,
                }),
                Feature::Exon => outputs.push(BaseOutput {
                    run,
                    feature: BaseFeature::Exon,
                }),
                Feature::FiveUtr => outputs.push(BaseOutput {
                    run,
                    feature: BaseFeature::FiveUtr,
                }),
                Feature::ThreeUtr => outputs.push(BaseOutput {
                    run,
                    feature: BaseFeature::ThreeUtr,
                }),
                Feature::Donor => outputs.push(BaseOutput {
                    run,
                    feature: BaseFeature::Donor,
                }),
                Feature::Acceptor => outputs.push(BaseOutput {
                    run,
                    feature: BaseFeature::Acceptor,
                }),
                Feature::Start | Feature::Stop => codon_outputs.push(run),
            }
        }
    }

    let exp = exp16_table();
    let identity = identity16_table();
    for strand in 0..2 {
        for i in 0..length {
            let pos = item_start + i as u64;
            let ix = strand * length + i;
            let cds = if need_cds {
                finite_input(la1[ix], "la1", chrom, item_start, pos, strand)?;
                Some(exp[la1[ix] as usize])
            } else {
                None
            };
            let five = if need_utr1 {
                let bits = utr7[(strand * 7 + 1) * length + i];
                finite_input(bits, "utr7[:,1]", chrom, item_start, pos, strand)?;
                Some(exp[bits as usize])
            } else {
                None
            };
            let three = if need_utr3 {
                let bits = utr7[(strand * 7 + 3) * length + i];
                finite_input(bits, "utr7[:,3]", chrom, item_start, pos, strand)?;
                Some(exp[bits as usize])
            } else {
                None
            };
            let donor = if need_donor {
                let bits = c[(strand * 4) * length + i];
                finite_input(bits, "c[:,0]", chrom, item_start, pos, strand)?;
                Some(identity[bits as usize])
            } else {
                None
            };
            let acceptor = if need_acceptor {
                let bits = c[(strand * 4 + 1) * length + i];
                finite_input(bits, "c[:,1]", chrom, item_start, pos, strand)?;
                Some(identity[bits as usize])
            } else {
                None
            };
            for output in outputs.iter_mut().filter(|o| o.run.strand == strand) {
                let p = match output.feature {
                    BaseFeature::Cds => cds.expect("cds table selected"),
                    BaseFeature::Exon => {
                        cds.expect("cds table selected")
                            + five.expect("utr1 table selected")
                            + three.expect("utr3 table selected")
                    }
                    BaseFeature::FiveUtr => five.expect("utr1 table selected"),
                    BaseFeature::ThreeUtr => three.expect("utr3 table selected"),
                    BaseFeature::Donor => donor.expect("donor table selected"),
                    BaseFeature::Acceptor => acceptor.expect("acceptor table selected"),
                };
                output.run.push(pos, quantize(p, scale))?;
            }
        }
    }
    for (codon_index, &(channel, channel_name)) in codon_channels.iter().enumerate() {
        for strand in 0..2 {
            let output = &mut codon_outputs[codon_index * 2 + strand];
            for i in 0..length {
                let pos = item_start + i as u64;
                let bits = c[(strand * 4 + channel) * length + i];
                finite_input(bits, channel_name, chrom, item_start, pos, strand)?;
                output.push(pos, quantize(identity[bits as usize], scale))?;
            }
        }
    }

    let mut counts = BTreeMap::new();
    for output in &mut outputs {
        output.run.finish(item_end)?;
        counts.insert(output.run.path.clone(), output.run.n_records);
    }
    for output in &mut codon_outputs {
        output.finish(item_end)?;
        counts.insert(output.path.clone(), output.n_records);
    }
    Ok(counts)
}

/// Header metadata stored in a version-2 run part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunHeader {
    pub chrom: String,
    pub item_start: u32,
    pub item_end: u32,
    pub scale: u32,
    pub feature: String,
    pub strand: char,
}

/// Streaming reader for validated absolute half-open `(start, end, q)` records.
#[derive(Debug)]
pub struct RunReader {
    reader: BufReader<File>,
    path: PathBuf,
    header: RunHeader,
    previous: Option<(u32, u16)>,
    finished: bool,
}

fn reader_path_error(path: &Path, error: io::Error) -> io::Error {
    io::Error::new(error.kind(), format!("{}: {error}", path.display()))
}

impl RunReader {
    /// Open and validate the header, retaining records for incremental iteration.
    pub fn open(path: &Path) -> io::Result<Self> {
        let result = (|| {
            let mut reader = BufReader::new(File::open(path)?);
            let mut raw = [0u8; HEADER_LEN];
            reader.read_exact(&mut raw)?;
            if &raw[..8] != MAGIC || u16::from_le_bytes([raw[8], raw[9]]) != VERSION {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bad track run magic or version",
                ));
            }
            let scale = u32::from_le_bytes(raw[10..14].try_into().unwrap());
            let item_start = u64::from_le_bytes(raw[14..22].try_into().unwrap());
            let item_end = u64::from_le_bytes(raw[22..30].try_into().unwrap());
            let chrom_len = u32::from_le_bytes(raw[30..34].try_into().unwrap()) as usize;
            let feature_len = u16::from_le_bytes([raw[34], raw[35]]) as usize;
            let strand = raw[36];
            if chrom_len == 0 || chrom_len > MAX_CHROM_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("chrom name length must be in 1..={MAX_CHROM_BYTES} bytes"),
                ));
            }
            if strand > 1 || raw[37] != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid track run header fields",
                ));
            }
            let mut chrom = vec![0; chrom_len];
            let mut feature = vec![0; feature_len];
            reader.read_exact(&mut chrom)?;
            reader.read_exact(&mut feature)?;
            let chrom = String::from_utf8(chrom)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 chrom"))?;
            let feature = String::from_utf8(feature)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid feature name"))?;
            if Feature::parse(&feature).is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid track run feature",
                ));
            }
            if scale == 0
                || scale > u16::MAX as u32
                || item_start >= item_end
                || item_end > u32::MAX as u64
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid track run header fields for item {chrom}:{item_start}"),
                ));
            }
            Ok(Self {
                reader,
                path: path.to_path_buf(),
                header: RunHeader {
                    chrom,
                    item_start: item_start as u32,
                    item_end: item_end as u32,
                    scale,
                    feature,
                    strand: if strand == 0 { '+' } else { '-' },
                },
                previous: None,
                finished: false,
            })
        })();
        result.map_err(|error| reader_path_error(path, error))
    }

    pub fn header(&self) -> &RunHeader {
        &self.header
    }

    fn read_record(&mut self) -> io::Result<Option<(u32, u32, u16)>> {
        let mut record = [0u8; RECORD_LEN];
        match self.reader.read(&mut record[..1])? {
            0 => return Ok(None),
            1 => self.reader.read_exact(&mut record[1..])?,
            _ => unreachable!(),
        }
        let start = u32::from_le_bytes(record[0..4].try_into().unwrap());
        let end = u32::from_le_bytes(record[4..8].try_into().unwrap());
        let q = u16::from_le_bytes(record[8..10].try_into().unwrap());
        let bad_order = self
            .previous
            .is_some_and(|(prev_end, prev_q)| start < prev_end || (start == prev_end && q == prev_q));
        if start < self.header.item_start
            || start >= end
            || end > self.header.item_end
            || q == 0
            || u32::from(q) > self.header.scale
            || bad_order
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid or non-canonical track run record",
            ));
        }
        self.previous = Some((end, q));
        Ok(Some((start, end, q)))
    }
}

impl Iterator for RunReader {
    type Item = io::Result<(u32, u32, u16)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.read_record() {
            Ok(Some(record)) => Some(Ok(record)),
            Ok(None) => {
                self.finished = true;
                None
            }
            Err(error) => {
                self.finished = true;
                Some(Err(reader_path_error(&self.path, error)))
            }
        }
    }
}

