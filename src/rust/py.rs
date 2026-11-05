use crate::{trackout, tracks, twobit};
use crate::decode as dec;
use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn io_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// I/O or record-parse failures from the 2bit reader: OSError, distinct from bad names/coordinates (ValueError).
fn os_err(e: impl std::fmt::Display) -> PyErr {
    PyOSError::new_err(e.to_string())
}

#[pyclass]
pub struct Genome {
    inner: twobit::TwoBit,
}

#[pymethods]
impl Genome {
    #[new]
    fn new(py: Python<'_>, path: &str) -> PyResult<Self> {
        let p = path.to_owned();
        let tb = py.allow_threads(move || twobit::TwoBit::open(Path::new(&p))).map_err(io_err)?;
        Ok(Genome { inner: tb })
    }

    fn chrom_sizes(&self) -> PyResult<BTreeMap<String, u64>> {
        let mut m = BTreeMap::new();
        for n in self.inner.names() {
            match self.inner.len(n) {
                Ok(Some(v)) => {
                    m.insert(n.to_string(), v);
                }
                Ok(None) => return Err(os_err(format!("2bit: missing length for {n}"))),
                Err(e) => return Err(os_err(e)),
            }
        }
        Ok(m)
    }

    fn fetch<'py>(&self, py: Python<'py>, chrom: &str, start: u64, end: u64) -> PyResult<Bound<'py, PyBytes>> {
        let ch = chrom.to_owned();
        match py.allow_threads(|| self.inner.fetch(&ch, start, end)) {
            Err(e) => Err(os_err(e)),
            Ok(None) => Err(PyValueError::new_err(format!("bad fetch {chrom}:{start}-{end}"))),
            Ok(Some(buf)) => Ok(PyBytes::new_bound(py, &buf)),
        }
    }
}

fn parse_strand(s: &str) -> PyResult<dec::Strand> {
    match s {
        "+" => Ok(dec::Strand::Plus),
        "-" => Ok(dec::Strand::Minus),
        x => Err(PyValueError::new_err(format!("strand must be +|-, got {x}"))),
    }
}

// PyO3 generates a redundant PyErr conversion for these fallible wrappers.
#[allow(clippy::useless_conversion)]
mod tracks_py {
    use super::*;

    fn track_array_error(array: &Bound<'_, PyAny>, name: &str, requirement: &str) -> PyErr {
        let dtype = array
            .getattr("dtype")
            .and_then(|dtype| dtype.getattr("name"))
            .and_then(|name| name.extract::<String>())
            .unwrap_or_else(|_| "unknown".to_owned());
        PyValueError::new_err(format!("{name} must be {requirement}, got {dtype}"))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)] // Mirrors the fixed Python API.
    #[pyo3(signature = (prefix, chrom, item_start, la1, utr7, c, features, scale=1000))]
    pub(super) fn track_runs<'py>(
        py: Python<'py>,
        prefix: &str,
        chrom: &str,
        item_start: u64,
        la1: &Bound<'py, PyAny>,
        utr7: &Bound<'py, PyAny>,
        c: &Bound<'py, PyAny>,
        features: Vec<String>,
        scale: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (la1_obj, utr7_obj, c_obj) = (la1, utr7, c);
        let la1 = la1
            .extract::<PyReadonlyArray2<'py, u16>>()
            .map_err(|_| {
                track_array_error(
                    la1_obj,
                    "la1",
                    "a uint16 view of the f16 copy (a.view(np.uint16))",
                )
            })?;
        let utr7 = utr7
            .extract::<PyReadonlyArray3<'py, u16>>()
            .map_err(|_| {
                track_array_error(
                    utr7_obj,
                    "utr7",
                    "a uint16 view of the f16 copy (a.view(np.uint16))",
                )
            })?;
        let c = c
            .extract::<PyReadonlyArray3<'py, u16>>()
            .map_err(|_| {
                track_array_error(
                    c_obj,
                    "c",
                    "a uint16 view of the f16 copy (a.view(np.uint16))",
                )
            })?;
        let (la_shape, utr_shape, c_shape) = (la1.shape(), utr7.shape(), c.shape());
        if la_shape[0] != 2 {
            return Err(PyValueError::new_err(format!("la1 must have shape [2, L], got [{}, {}]", la_shape[0], la_shape[1])));
        }
        let length = la_shape[1];
        if utr_shape != [2, 7, length] {
            return Err(PyValueError::new_err(format!("utr7 must have shape [2, 7, {length}], got {utr_shape:?}")));
        }
        if c_shape != [2, 4, length] {
            return Err(PyValueError::new_err(format!("c must have shape [2, 4, {length}], got {c_shape:?}")));
        }
        if !la1.is_c_contiguous() {
            return Err(track_array_error(la1_obj, "la1", "C-contiguous uint16"));
        }
        if !utr7.is_c_contiguous() {
            return Err(track_array_error(utr7_obj, "utr7", "C-contiguous uint16"));
        }
        if !c.is_c_contiguous() {
            return Err(track_array_error(c_obj, "c", "C-contiguous uint16"));
        }
        if !(la1.data() as usize).is_multiple_of(std::mem::align_of::<u16>()) {
            return Err(track_array_error(la1_obj, "la1", "aligned uint16"));
        }
        if !(utr7.data() as usize).is_multiple_of(std::mem::align_of::<u16>()) {
            return Err(track_array_error(utr7_obj, "utr7", "aligned uint16"));
        }
        if !(c.data() as usize).is_multiple_of(std::mem::align_of::<u16>()) {
            return Err(track_array_error(c_obj, "c", "aligned uint16"));
        }
        let la_slice = la1
            .as_slice()
            .map_err(|_| track_array_error(la1_obj, "la1", "C-contiguous uint16"))?;
        let utr_slice = utr7
            .as_slice()
            .map_err(|_| track_array_error(utr7_obj, "utr7", "C-contiguous uint16"))?;
        let c_slice = c
            .as_slice()
            .map_err(|_| track_array_error(c_obj, "c", "C-contiguous uint16"))?;
        let (prefix, chrom) = (prefix.to_owned(), chrom.to_owned());
        let paths = py.allow_threads(move || {
            tracks::write_track_runs(
                Path::new(&prefix),
                tracks::TrackInput {
                    chrom: &chrom,
                    item_start,
                    la1: la_slice,
                    utr7: utr_slice,
                    c: c_slice,
                    feature_names: &features,
                    scale,
                },
            )
        }).map_err(io_err)?;
        let result = PyDict::new_bound(py);
        for (path, count) in paths {
            result.set_item(path.to_string_lossy().as_ref(), count)?;
        }
        Ok(result)
    }

    /// The run parts → the tracks. `kind` is `bigwig` (default) or
    /// `bedgraph` (`--track-type`), `threads` the tokio worker count (0: the
    /// runtime default, all cores). `item_bases` is one path per run item,
    /// without the `.<feature>.<strand>.runs` suffix: only those parts are
    /// read. Returns each output's path with its `PREFIX.prov.json` entry:
    /// records, sha256 (a bigWig has no text field and is bound to the stamp
    /// by it), feature and strand.
    #[pyfunction]
    #[pyo3(signature = (parts_dir, prefix, chrom_sizes, features, kind, stamp, threads, item_bases))]
    #[allow(clippy::too_many_arguments)] // Mirrors the fixed Python API.
    pub(super) fn finalize_tracks<'py>(
        py: Python<'py>,
        parts_dir: &str,
        prefix: &str,
        chrom_sizes: BTreeMap<String, u64>,
        features: Vec<String>,
        kind: &str,
        stamp: &str,
        threads: usize,
        item_bases: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (parts_dir, prefix, chrom_sizes, features, kind, stamp, item_bases) = (
            parts_dir.to_owned(),
            prefix.to_owned(),
            chrom_sizes,
            features,
            kind.to_owned(),
            stamp.to_owned(),
            item_bases,
        );
        let written = py
            .allow_threads(move || {
                let bases: Vec<PathBuf> = item_bases.iter().map(PathBuf::from).collect();
                trackout::finalize_tracks(
                    Path::new(&parts_dir),
                    &prefix,
                    &chrom_sizes,
                    &features,
                    &kind,
                    &stamp,
                    threads,
                    &bases,
                )
            })
            .map_err(io_err)?;
        let result = PyDict::new_bound(py);
        for (path, out) in written {
            let entry = PyDict::new_bound(py);
            entry.set_item("records", out.records)?;
            entry.set_item("sha256", out.sha256)?;
            entry.set_item("feature", out.feature)?;
            entry.set_item("strand", out.strand.to_string())?;
            result.set_item(path.to_string_lossy().as_ref(), entry)?;
        }
        Ok(result)
    }
}

#[pyfunction]
fn exp16_table<'py>(py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec_bound(py, tracks::exp16_table().to_vec())
}

// PyO3 generates a redundant PyErr conversion for these fallible wrappers.
#[allow(clippy::useless_conversion)]
mod out_py {
    use super::*;
    use crate::out;
    use pyo3::PyRef;
    use std::path::PathBuf;

    /// Sequence names, every run, before any model is loaded: the rule on
    /// the run's chroms, a duplicate anywhere in the 2bit's raw index
    /// (`TwoBit::names` — the duplicate check the
    /// HashMap-collapsed `chrom_sizes` could never see), and — with `gtf` —
    /// the names GTF cannot carry. `genome` is an `ancestra._native.Genome` or the
    /// 2bit path (a `str` or a `pathlib.Path`, which `PyOS_FSPath` handles).
    /// Returns every refused name (ascii-escaped) with its reason.
    #[pyfunction]
    pub(super) fn check_names(
        py: Python<'_>,
        genome: &Bound<'_, PyAny>,
        chroms: Vec<String>,
        gtf: bool,
    ) -> PyResult<Vec<(String, String)>> {
        if let Ok(genome) = genome.extract::<PyRef<'_, Genome>>() {
            // the raw index is already in memory: no I/O, so the GIL stays held
            Ok(out::check_names(&genome.inner, &chroms, gtf))
        } else {
            let path = genome
                .extract::<PathBuf>()
                .map_err(|_| PyValueError::new_err("genome must be a Genome or a 2bit path"))?;
            let tb = py.allow_threads(|| twobit::TwoBit::open(&path)).map_err(io_err)?;
            Ok(out::check_names(&tb, &chroms, gtf))
        }
    }

    /// The run stamp's ` genome=<2bit basename>:<sha256>` field.
    #[pyfunction]
    pub(super) fn stamp_genome(py: Python<'_>, path: &str, sha256: &str) -> PyResult<String> {
        let (p, s) = (path.to_owned(), sha256.to_owned());
        py.allow_threads(|| out::stamp_genome(Path::new(&p), &s)).map_err(io_err)
    }

    /// The stamp of a text output (its `#S` line without the `#`).
    #[pyfunction]
    pub(super) fn stamp_of(py: Python<'_>, path: &str) -> PyResult<String> {
        let p = path.to_owned();
        py.allow_threads(|| out::stamp_of(Path::new(&p))).map_err(io_err)
    }

    /// Parts (the run's item order) → BED12, and the GTF /
    /// GFF3 / bigBed when `formats` asks. `chrom_sizes` is the 2bit's
    /// (`Genome.chrom_sizes`), which only the bigBed reads. Returns each
    /// output's path and record count.
    #[pyfunction]
    #[pyo3(signature = (parts, prefix, stamp, formats, chrom_sizes))]
    pub(super) fn write_outputs(
        py: Python<'_>,
        parts: Vec<String>,
        prefix: &str,
        stamp: &str,
        formats: Vec<String>,
        chrom_sizes: BTreeMap<String, u64>,
    ) -> PyResult<Vec<(String, usize)>> {
        let (prefix, stamp, formats) = (prefix.to_owned(), stamp.to_owned(), formats);
        let written = py
            .allow_threads(move || {
                let parts: Vec<PathBuf> = parts.into_iter().map(PathBuf::from).collect();
                let formats: Vec<&str> = formats.iter().map(String::as_str).collect();
                out::write_outputs(&parts, &prefix, &stamp, &formats, &chrom_sizes)
            })
            .map_err(io_err)?;
        Ok(written
            .into_iter()
            .map(|(p, n)| (p.to_string_lossy().into_owned(), n))
            .collect())
    }

    #[pyfunction]
    #[pyo3(signature = (alt_parts, primary_parts, prefix, stamp, formats, chrom_sizes, threshold))]
    pub(super) fn write_alt_outputs(
        py: Python<'_>,
        alt_parts: Vec<String>,
        primary_parts: Vec<String>,
        prefix: &str,
        stamp: &str,
        formats: Vec<String>,
        chrom_sizes: BTreeMap<String, u64>,
        threshold: f64,
    ) -> PyResult<Vec<(String, usize)>> {
        let (prefix, stamp, formats) = (prefix.to_owned(), stamp.to_owned(), formats);
        let written = py
            .allow_threads(move || {
                let alt_parts: Vec<PathBuf> = alt_parts.into_iter().map(PathBuf::from).collect();
                let primary_parts: Vec<PathBuf> = primary_parts.into_iter().map(PathBuf::from).collect();
                let formats: Vec<&str> = formats.iter().map(String::as_str).collect();
                out::write_alt_outputs(
                    &alt_parts,
                    &primary_parts,
                    &prefix,
                    &stamp,
                    &formats,
                    &chrom_sizes,
                    threshold,
                )
            })
            .map_err(io_err)?;
        Ok(written
            .into_iter()
            .map(|(p, n)| (p.to_string_lossy().into_owned(), n))
            .collect())
    }
}

fn arr_vec(obj: &Bound<'_, PyAny>, rows: usize, what: &str) -> PyResult<(Vec<f32>, [usize; 2])> {
    // flag check first so F-order raises ValueError (any contiguous layout could be copied instead)
    let cc: bool = obj.getattr("flags").map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous f32")))?
        .get_item("C_CONTIGUOUS").map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous f32")))?
        .extract().map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous f32")))?;
    if !cc {
        return Err(PyValueError::new_err(format!("{what} must be C-contiguous f32")));
    }
    let a = obj.extract::<PyReadonlyArray2<f32>>().map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous f32")))?;
    let sh = a.shape();
    if sh[0] != rows {
        return Err(PyValueError::new_err(format!("{what} must be [{rows}, L], got [{}, {}]", sh[0], sh[1])));
    }
    let v = a.as_slice().map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous f32")))?.to_vec();
    Ok((v, [sh[0], sh[1]]))
}

fn parse_params(d: Option<Bound<'_, PyDict>>) -> PyResult<dec::Params> {
    // `Params::default` is the only source of defaults; the dict only overrides keys it has
    let mut p = dec::Params::default();
    let Some(d) = d else { return Ok(p) };
    let resolved = p.resolved();
    for key in d.keys() {
        let key = key.extract::<String>()?;
        if !resolved.iter().any(|(name, _)| name.split('.').next() == Some(key.as_str())) {
            return Err(PyValueError::new_err(format!("unknown decoder parameter: {key}")));
        }
    }
    if let Some(o) = d.get_item("c_thresh")? {
        if let Ok(v) = o.extract::<Vec<f32>>() {
            match v.len() {
                4 => p.c_thresh = [v[0], v[1], v[2], v[3]],
                1 => p.c_thresh = [v[0]; 4],
                _ => return Err(PyValueError::new_err("c_thresh must have 4 values")),
            }
        } else if let Ok(v) = o.extract::<f32>() {
            p.c_thresh = [v; 4];
        } else if let Ok(v) = o.extract::<f64>() {
            p.c_thresh = [v as f32; 4];
        } else {
            return Err(PyValueError::new_err("c_thresh must have 4 values"));
        }
    }
    if let Some(o) = d.get_item("c_bias")? {
        if let Ok(v) = o.extract::<Vec<f32>>() {
            match v.len() {
                4 => p.c_bias = [v[0], v[1], v[2], v[3]],
                1 => p.c_bias = [v[0]; 4],
                _ => return Err(PyValueError::new_err("c_bias must have 4 values")),
            }
        } else if let Ok(v) = o.extract::<f32>() {
            p.c_bias = [v; 4];
        } else if let Ok(v) = o.extract::<f64>() {
            p.c_bias = [v as f32; 4];
        } else {
            return Err(PyValueError::new_err("c_bias must have 4 values"));
        }
    }
    if let Some(o) = d.get_item("w_c")? {
        p.w_c = o.extract::<f64>().map_err(|_| PyValueError::new_err("w_c must be float"))? as f32;
    }
    if let Some(o) = d.get_item("w_b")? {
        p.w_b = o.extract::<f64>().map_err(|_| PyValueError::new_err("w_b must be float"))? as f32;
    }
    if let Some(o) = d.get_item("w_b_intron")? {
        p.w_b_intron = o.extract::<f64>().map_err(|_| PyValueError::new_err("w_b_intron must be float"))? as f32;
    }
    if let Some(o) = d.get_item("intron_min")? {
        p.intron_min = o.extract::<u64>().map_err(|_| PyValueError::new_err("intron_min must be int"))?;
    }
    if let Some(o) = d.get_item("intron_l0")? {
        p.intron_l0 = o.extract::<u64>().map_err(|_| PyValueError::new_err("intron_l0 must be int"))?;
    }
    if let Some(o) = d.get_item("intron_geom_b")? {
        p.intron_geom_b = o.extract::<f64>().map_err(|_| PyValueError::new_err("intron_geom_b must be float"))? as f32;
    }
    if let Some(o) = d.get_item("exon_max")? {
        p.exon_max = o.extract::<u64>().map_err(|_| PyValueError::new_err("exon_max must be int"))?;
    }
    if let Some(o) = d.get_item("gene_start_penalty")? {
        p.gene_start_penalty = o.extract::<f64>().map_err(|_| PyValueError::new_err("gene_start_penalty must be float"))? as f32;
    }
    if let Some(o) = d.get_item("exon_soft_min")? {
        p.exon_soft_min = o.extract::<u64>().map_err(|_| PyValueError::new_err("exon_soft_min must be int"))?;
    }
    if let Some(o) = d.get_item("exon_short_penalty")? {
        p.exon_short_penalty = o.extract::<f64>().map_err(|_| PyValueError::new_err("exon_short_penalty must be float"))? as f32;
    }
    if let Some(o) = d.get_item("cds_base_bias")? {
        p.cds_base_bias = o.extract::<f64>().map_err(|_| PyValueError::new_err("cds_base_bias must be float"))? as f32;
    }
    if let Some(o) = d.get_item("max_candidates")? {
        p.max_candidates = o.extract::<usize>().map_err(|_| PyValueError::new_err("max_candidates must be int"))?;
    }
    // the weak band defaults to empty: a copy of the (possibly overridden) strong thresholds
    p.c_thresh_weak = match d.get_item("c_thresh_weak")? {
        Some(o) => f4(&o, "c_thresh_weak")?,
        None => p.c_thresh,
    };
    if let Some(o) = d.get_item("weak_penalty")? {
        p.weak_penalty = f4(&o, "weak_penalty")?;
    }
    if let Some(o) = d.get_item("weak_budget")? {
        p.weak_budget = o.extract::<u32>().map_err(|_| PyValueError::new_err("weak_budget must be int"))?;
    }
    if let Some(o) = d.get_item("max_weak_candidates")? {
        p.max_weak_candidates = o.extract::<usize>().map_err(|_| PyValueError::new_err("max_weak_candidates must be int"))?;
    }
    if p.resolved().iter().any(|(_, v)| !v.is_finite())
        || p.c_thresh.iter().any(|v| !(0.0..=1.0).contains(v))
    {
        return Err(PyValueError::new_err("decoder parameters must be finite; c_thresh must be in [0, 1]"));
    }
    if p.c_thresh_weak.iter().any(|v| !(0.0..=1.0).contains(v)) || p.weak_penalty.iter().any(|&v| v > 0.0) {
        return Err(PyValueError::new_err("c_thresh_weak must be in [0, 1]; weak_penalty must be <= 0"));
    }
    if p.intron_geom_b > 0.0 {
        return Err(PyValueError::new_err("intron_geom_b must be <= 0 (a per-base penalty on long introns)"));
    }
    if p.weak_budget > 254 {
        return Err(PyValueError::new_err("weak_budget must be <= 254"));
    }
    Ok(p)
}

/// Compiled decoder defaults: `Params::default()` under the parameter override keys.
#[pyfunction]
fn default_params<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = dec::Params::default();
    let m = PyDict::new_bound(py);
    m.set_item("c_thresh", d.c_thresh.to_vec())?;
    m.set_item("c_bias", d.c_bias.to_vec())?;
    m.set_item("w_c", d.w_c as f64)?;
    m.set_item("w_b", d.w_b as f64)?;
    m.set_item("w_b_intron", d.w_b_intron as f64)?;
    m.set_item("intron_min", d.intron_min)?;
    m.set_item("intron_l0", d.intron_l0)?;
    m.set_item("intron_geom_b", d.intron_geom_b as f64)?;
    m.set_item("exon_max", d.exon_max)?;
    m.set_item("gene_start_penalty", d.gene_start_penalty as f64)?;
    m.set_item("exon_soft_min", d.exon_soft_min)?;
    m.set_item("exon_short_penalty", d.exon_short_penalty as f64)?;
    m.set_item("max_candidates", d.max_candidates)?;
    m.set_item("weak_budget", d.weak_budget)?;
    m.set_item("c_thresh_weak", d.c_thresh_weak.to_vec())?;
    m.set_item("weak_penalty", d.weak_penalty.to_vec())?;
    m.set_item("max_weak_candidates", d.max_weak_candidates)?;
    m.set_item("cds_base_bias", d.cds_base_bias as f64)?;
    Ok(m)
}

/// sha256 hex of the canonical defaults JSON (see `dec::Params::defaults_digest`; opaque to Python).
fn defaults_digest() -> String {
    dec::Params::defaults_digest()
}

/// Build stamp of this binding: the git sha and dirty flag of the tree it was built from (`build.rs`), the UTC build
/// time, and the sha256 of the canonical JSON of `default_params()`.
#[pyfunction]
fn build_info<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let m = PyDict::new_bound(py);
    m.set_item("git_sha", option_env!("ANCESTRA_BUILD_GIT_SHA").unwrap_or("unknown"))?;
    m.set_item("dirty", option_env!("ANCESTRA_BUILD_GIT_DIRTY").unwrap_or("1") == "1")?;
    m.set_item("built_at", option_env!("ANCESTRA_BUILD_BUILT_AT").unwrap_or("unknown"))?;
    m.set_item("defaults_digest", defaults_digest())?;
    Ok(m)
}

/// One float or four (donor, acceptor, start, stop), like `c_thresh`.
fn f4(o: &Bound<'_, PyAny>, name: &str) -> PyResult<[f32; 4]> {
    let bad = || PyValueError::new_err(format!("{name} must have 4 values"));
    if let Ok(v) = o.extract::<Vec<f32>>() {
        match v.len() {
            4 => Ok([v[0], v[1], v[2], v[3]]),
            1 => Ok([v[0]; 4]),
            _ => Err(bad()),
        }
    } else if let Ok(v) = o.extract::<f64>() {
        Ok([v as f32; 4])
    } else {
        Err(bad())
    }
}

#[pyfunction(name = "decode")]
#[pyo3(signature = (seq, strand, log_a, log_b, c, start, params=None))]
fn py_decode<'py>(
    py: Python<'py>,
    seq: Bound<'py, PyBytes>,
    strand: &str,
    log_a: Bound<'py, PyAny>,
    log_b: Bound<'py, PyAny>,
    c: Bound<'py, PyAny>,
    start: u64,
    params: Option<Bound<'py, PyDict>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let st = parse_strand(strand)?;
    let seqv = seq.as_bytes().to_vec();
    let l = seqv.len();
    let (lav, lash) = arr_vec(&log_a, 3, "log_a")?;
    let (lbv, lbsh) = arr_vec(&log_b, 3, "log_b")?;
    let (cv, csh) = arr_vec(&c, 4, "c")?;
    if lash[1] != l || lbsh[1] != l || csh[1] != l {
        return Err(PyValueError::new_err(format!(
            "array length {} vs seq len {l}",
            lash[1].max(lbsh[1]).max(csh[1])
        )));
    }
    let par = parse_params(params)?;
    let report_weak = par.weak_budget > 0;
    let genes = py.allow_threads(move || {
        let e = dec::Emissions { start, seq: &seqv, strand: st, log_a: &lav, log_b: &lbv, c: &cv };
        dec::decode(&e, &par)
    });
    let mut out = Vec::with_capacity(genes.len());
    for g in genes {
        let d = PyDict::new_bound(py);
        d.set_item("strand", if g.strand == dec::Strand::Plus { "+" } else { "-" })?;
        let cds: Vec<(u64, u64)> = g.cds.iter().map(|e| (e.start, e.end)).collect();
        d.set_item("cds", cds)?;
        d.set_item("score", g.score)?;
        if report_weak {
            d.set_item("n_weak", g.n_weak)?; // only when the budget is on: default dicts unchanged
        }
        out.push(d);
    }
    Ok(out)
}

/// Additive T=1 path-score decomposition of one CDS chain (see `dec::chain_terms`).
/// `cds` holds ascending genomic (start, end) blocks inside [start, start+len).
#[pyfunction(name = "chain_terms")]
#[pyo3(signature = (seq, strand, log_a, log_b, c, start, cds, params=None))]
#[allow(clippy::too_many_arguments)]
fn py_chain_terms<'py>(
    py: Python<'py>,
    seq: Bound<'py, PyBytes>,
    strand: &str,
    log_a: Bound<'py, PyAny>,
    log_b: Bound<'py, PyAny>,
    c: Bound<'py, PyAny>,
    start: u64,
    cds: Vec<(u64, u64)>,
    params: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (st, seqv, lav, lbv, cv) = mbr_inputs(&seq, strand, &log_a, &log_b, &c)?;
    let l = seqv.len();
    if cds.is_empty() {
        return Err(PyValueError::new_err("cds must be non-empty"));
    }
    let mut ex: Vec<dec::Exon> = Vec::with_capacity(cds.len());
    for (s, e) in &cds {
        if e <= s || *s < start || *e > start + l as u64 {
            return Err(PyValueError::new_err(format!("cds ({s},{e}) outside sequence")));
        }
        ex.push(dec::Exon { start: *s, end: *e });
    }
    ex.sort();
    let par = parse_params(params)?;
    let minus = st == dec::Strand::Minus;
    let (terms, total) = py.allow_threads(move || {
        let em = dec::Emissions { start, seq: &seqv, strand: st, log_a: &lav, log_b: &lbv, c: &cv };
        let tex = dec::to_transcript(&ex, start, l, minus);
        let t = dec::chain_terms(&em, &par, l, minus, &tex);
        let tot = t.total() as f32;
        let feas = t.sites.iter().all(|s| s.above_thresh);
        (t, if feas { tot } else { f32::NEG_INFINITY })
    });
    let d = PyDict::new_bound(py);
    d.set_item("gene_start", terms.gene_start)?;
    d.set_item("site_start", terms.site_start)?;
    d.set_item("site_donor", terms.site_donor)?;
    d.set_item("site_acceptor", terms.site_acceptor)?;
    d.set_item("site_stop", terms.site_stop)?;
    d.set_item("exon_region", terms.exon_region)?;
    d.set_item("exon_phase", terms.exon_phase)?;
    d.set_item("cds_base_bias", terms.cds_base_bias)?;
    d.set_item("intron_region", terms.intron_region)?;
    d.set_item("intron_phase", terms.intron_phase)?;
    d.set_item("intron_geom", terms.intron_geom)?;
    d.set_item("exon_short", terms.exon_short)?;
    d.set_item("total", total)?;
    d.set_item("n_weak", terms.n_weak)?;
    let mut sites = Vec::with_capacity(terms.sites.len());
    for s in &terms.sites {
        let sd = PyDict::new_bound(py);
        sd.set_item("type", s.typ)?;
        sd.set_item("pos", s.pos)?;
        sd.set_item("p", s.p)?;
        sd.set_item("motif_ok", s.motif_ok)?;
        sd.set_item("above_thresh", s.above_thresh)?;
        sites.push(sd);
    }
    d.set_item("sites", sites)?;
    Ok(d)
}

/// Emission arrays checked exactly as `decode` checks them.
fn mbr_inputs(
    seq: &Bound<'_, PyBytes>,
    strand: &str,
    log_a: &Bound<'_, PyAny>,
    log_b: &Bound<'_, PyAny>,
    c: &Bound<'_, PyAny>,
) -> PyResult<(dec::Strand, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let st = parse_strand(strand)?;
    let seqv = seq.as_bytes().to_vec();
    let l = seqv.len();
    let (lav, lash) = arr_vec(log_a, 3, "log_a")?;
    let (lbv, lbsh) = arr_vec(log_b, 3, "log_b")?;
    let (cv, csh) = arr_vec(c, 4, "c")?;
    if lash[1] != l || lbsh[1] != l || csh[1] != l {
        return Err(PyValueError::new_err(format!(
            "array length {} vs seq len {l}",
            lash[1].max(lbsh[1]).max(csh[1])
        )));
    }
    Ok((st, seqv, lav, lbv, cv))
}

/// Expected-F1 (MBR) decoding of one strand of one region.
/// Returns `(primary, lowconf)`: `primary` holds the mode chain of every locus whose mode posterior
/// is >= tau; `lowconf` holds every candidate chain of every other locus. Each is a gene dict as
/// `decode` returns (`strand`, `cds`, `score`, `n_weak`) plus `posterior`, `p_none` (the locus's
/// no-gene mass) and `locus` (index within the region); primary genes also carry `alternatives`
/// (the locus's other chains as `(cds, posterior)`). Every locus is in exactly one of the two lists.
/// Temperatures below 1e-3 are rejected (the 1/T scaling overflows f64 there).
#[pyfunction(name = "decode_mbr")]
#[pyo3(signature = (seq, strand, log_a, log_b, c, start, params=None, n_samples=200, temperature=1.0, tau=0.5, seed=0))]
#[allow(clippy::too_many_arguments)]
fn py_decode_mbr<'py>(
    py: Python<'py>,
    seq: Bound<'py, PyBytes>,
    strand: &str,
    log_a: Bound<'py, PyAny>,
    log_b: Bound<'py, PyAny>,
    c: Bound<'py, PyAny>,
    start: u64,
    params: Option<Bound<'py, PyDict>>,
    n_samples: usize,
    temperature: f64,
    tau: f64,
    seed: u64,
) -> PyResult<(Vec<Bound<'py, PyDict>>, Vec<Bound<'py, PyDict>>)> {
    let (st, seqv, lav, lbv, cv) = mbr_inputs(&seq, strand, &log_a, &log_b, &c)?;
    let par = parse_params(params)?;
    if n_samples == 0 || !(temperature >= 1e-3 && temperature.is_finite()) || !(0.0..=1.0).contains(&tau) {
        return Err(PyValueError::new_err("need n_samples > 0, finite temperature >= 1e-3, tau in [0, 1]"));
    }
    let o = dec::MbrOpts { n_samples, temperature, tau, seed };
    let m = py.allow_threads(move || {
        let e = dec::Emissions { start, seq: &seqv, strand: st, log_a: &lav, log_b: &lbv, c: &cv };
        dec::decode_mbr(&e, &par, &o)
    }).map_err(|e: String| PyValueError::new_err(e))?;
    let sgn = if st == dec::Strand::Plus { "+" } else { "-" };
    let blocks = |ch: &dec::MbrChain| -> Vec<(u64, u64)> { ch.cds.iter().map(|e| (e.start, e.end)).collect() };
    let (mut primary, mut lowconf) = (Vec::new(), Vec::new());
    for (k, loc) in m.loci.iter().enumerate() {
        let gene = |ch: &dec::MbrChain| -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new_bound(py);
            d.set_item("strand", sgn)?;
            d.set_item("cds", blocks(ch))?;
            d.set_item("score", ch.score)?;
            d.set_item("n_weak", ch.n_weak)?;
            d.set_item("posterior", ch.posterior)?;
            d.set_item("p_none", loc.p_none)?;
            d.set_item("locus", k)?;
            Ok(d)
        };
        if loc.primary {
            let d = gene(&loc.chains[0])?;
            let alt: Vec<(Vec<(u64, u64)>, f64)> = loc.chains[1..].iter().map(|ch| (blocks(ch), ch.posterior)).collect();
            d.set_item("alternatives", alt)?;
            primary.push(d);
        } else {
            for ch in &loc.chains {
                lowconf.push(gene(ch)?);
            }
        }
    }
    Ok((primary, lowconf))
}

/// The compiled extension `ancestra._native`.
#[pymodule]
#[pyo3(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Genome>()?;
    m.add_function(wrap_pyfunction!(py_decode, m)?)?;
    m.add_function(wrap_pyfunction!(default_params, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_decode_mbr, m)?)?;
    m.add_function(wrap_pyfunction!(py_chain_terms, m)?)?;
    m.add_function(wrap_pyfunction!(tracks_py::track_runs, m)?)?;
    m.add_function(wrap_pyfunction!(exp16_table, m)?)?;
    m.add_function(wrap_pyfunction!(tracks_py::finalize_tracks, m)?)?;
    m.add_function(wrap_pyfunction!(out_py::check_names, m)?)?;
    m.add_function(wrap_pyfunction!(out_py::stamp_genome, m)?)?;
    m.add_function(wrap_pyfunction!(out_py::stamp_of, m)?)?;
    m.add_function(wrap_pyfunction!(out_py::write_outputs, m)?)?;
    m.add_function(wrap_pyfunction!(out_py::write_alt_outputs, m)?)?;
    Ok(())
}
