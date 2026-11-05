// Random-access 2bit reader: the file stays on disk and every `fetch` reads (pread) only the packed-DNA bytes its
// window needs. Per genome only the index plus the parsed N-block and mask-block tables are kept in memory, so a
// worker's genome cache costs ~KB–MB per assembly instead of the whole 2bit (mean 0.65 GB). Mask/N-block semantics
// mirror the previous whole-file reader, with zero new deps (std `FileExt::read_exact_at` + `OnceLock` only).
//
// Open cost: `open` reads the 16-byte header plus the whole index region in ONE large pread and parses
// (name, record offset) pairs in memory; no sequence record is touched. Each record is parsed lazily on first use
// (dna_size+counts, then N-blocks+mask-count, then mask-blocks: 3 preads worst case, once per sequence) and cached
// in a `OnceLock`, so concurrent `fetch` under `allow_threads` parses once. `len` needs only the first 8-byte
// pread, so `chrom_sizes()` costs one tiny pread per sequence.
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const SIG: u32 = 0x1A412743;

struct Rec {
    dna_size: u32,
    n_blocks: Vec<(u32, u32)>,
    mask_blocks: Vec<(u32, u32)>,
    /// file offset of this sequence's first packed-DNA byte
    packed: u64,
}

struct Seq {
    name: String,
    /// file offset of this sequence's record
    rec_off: u64,
    dna_size: OnceLock<u32>,
    /// Full record, cached only on successful parse (a failed parse caches nothing and is retried next use)
    rec: OnceLock<Rec>,
}

pub struct TwoBit {
    path: PathBuf,
    file: File,
    le: bool,
    seqs: Vec<Seq>,
    idx: HashMap<String, usize>,
}

/// `open`'s refusal of a file that is not a 2bit names that file exactly
/// once, with `InvalidData`.
fn bad_path(path: &Path, msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("2bit: {}: {msg}", path.display()))
}

/// `open`'s open/metadata I/O failure names the file exactly once and keeps
/// its OS kind, so a missing or unreadable 2bit stays `NotFound` or
/// `PermissionDenied` rather than the `InvalidData` a malformed 2bit gets. It
/// also names a pread's non-EOF failure once, reached through `short_read` (a
/// non-empty directory gives `IsADirectory`); a short read is routed to
/// `short_read`, not here.
fn io_path(path: &Path, e: std::io::Error) -> std::io::Error {
    std::io::Error::new(e.kind(), format!("2bit: {}: {e}", path.display()))
}

/// A pread that runs off the end is a short read of a 2bit that shrank: the
/// file is named once, as a malformed-file refusal (`bad_path(path,
/// "truncated")`, `InvalidData`), both inside `open` and in `read_at`'s
/// post-open reads.
fn short_read(path: &Path, e: std::io::Error) -> std::io::Error {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        bad_path(path, "truncated")
    } else {
        io_path(path, e)
    }
}

fn get_u32(b: &[u8], le: bool) -> u32 {
    let a: [u8; 4] = b[..4].try_into().unwrap();
    if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) }
}

impl TwoBit {
    fn read_at(&self, off: u64, n: usize) -> std::io::Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.file.read_exact_at(&mut buf, off).map_err(|e| short_read(&self.path, e))?;
        Ok(buf)
    }

    pub fn open(path: &Path) -> std::io::Result<TwoBit> {
        let file = File::open(path).map_err(|e| io_path(path, e))?;
        let len = file.metadata().map_err(|e| io_path(path, e))?.len();
        if len < 16 {
            return Err(bad_path(path, "header too short"));
        }
        let mut sig = [0u8; 4];
        file.read_exact_at(&mut sig, 0).map_err(|e| short_read(path, e))?;
        let le = if u32::from_le_bytes(sig) == SIG {
            true
        } else if u32::from_be_bytes(sig) == SIG {
            false
        } else {
            return Err(bad_path(path, "bad signature"));
        };
        let mut hdr = [0u8; 16];
        file.read_exact_at(&mut hdr, 0).map_err(|e| short_read(path, e))?;
        let version = get_u32(&hdr[4..], le);
        if version != 0 {
            return Err(bad_path(path, &format!("unsupported version {version}")));
        }
        let count = get_u32(&hdr[8..], le) as usize;
        // Whole index region in one pread. Names are ≤255 B + 5 B overhead, so count*64 covers the
        // common case; grow and re-read only if an entry runs past the buffer.
        let index_end = len.saturating_sub(16) as usize;
        let mut cap = (count.saturating_mul(64)).min(index_end);
        let index: Vec<(String, u64)> = loop {
            let mut buf = vec![0u8; cap];
            if cap > 0 {
                file.read_exact_at(&mut buf, 16).map_err(|e| short_read(path, e))?;
            }
            match parse_index(&buf, count, le) {
                Ok(v) => break v,
                Err(true) => {
                    let need = buf.len().checked_mul(2).unwrap_or(usize::MAX).max(buf.len() + 256);
                    cap = need.min(index_end);
                    if cap <= buf.len() {
                        return Err(bad_path(path, "truncated index"));
                    }
                }
                Err(false) => return Err(bad_path(path, "bad index")),
            }
        };
        let mut tb = TwoBit { file, le, seqs: Vec::with_capacity(count), idx: HashMap::with_capacity(count), path: path.to_owned() };
        for (name, o) in &index {
            // A record's first 8 bytes (dna_size + nBlockCount) must be in the file: a truncation inside
            // the records refuses at open instead of reporting length 0 later. One index pread only.
            if o + 8 > len {
                return Err(bad_path(path, &format!("record offset of {name} past end of file")));
            }
        }
        for (i, (name, o)) in index.into_iter().enumerate() {
            tb.idx.insert(name.clone(), i);
            tb.seqs.push(Seq { name, rec_off: o, dna_size: OnceLock::new(), rec: OnceLock::new() });
        }
        Ok(tb)
    }

    /// dna_size with a single 8-byte pread (dna_size + nBlockCount), cached.
    fn seq_len(&self, i: usize) -> std::io::Result<u32> {
        let s = &self.seqs[i];
        if let Some(&v) = s.dna_size.get() {
            return Ok(v);
        }
        let b = self.read_at(s.rec_off, 8)?;
        let v = get_u32(&b, self.le);
        let _ = s.dna_size.set(v); // lost race => winner's identical value stands
        Ok(*s.dna_size.get().unwrap())
    }

    fn parse_record(&self, s: &Seq) -> std::io::Result<Rec> {
        let o = s.rec_off;
        let head = self.read_at(o, 8)?;
        let dna_size = get_u32(&head, self.le);
        let nb = get_u32(&head[4..], self.le) as usize;
        let mid = self.read_at(o + 8, 8 * nb + 4)?;
        let le = self.le;
        let at = |j: usize| get_u32(&mid[4 * j..], le);
        let ns: Vec<u32> = (0..nb).map(at).collect();
        let ne: Vec<u32> = (nb..2 * nb).map(at).collect();
        let mb = at(2 * nb) as usize;
        let tail = self.read_at(o + 8 + (8 * nb + 4) as u64, 8 * mb)?;
        let ms: Vec<u32> = (0..mb).map(|j| get_u32(&tail[4 * j..], le)).collect();
        let me: Vec<u32> = (mb..2 * mb).map(|j| get_u32(&tail[4 * j..], le)).collect();
        let packed = o + 8 + (8 * nb + 4) as u64 + (8 * mb) as u64 + 4; // reserved
        let mut n_blocks: Vec<(u32, u32)> = ns.into_iter().zip(ne).collect();
        let mut mask_blocks: Vec<(u32, u32)> = ms.into_iter().zip(me).collect();
        n_blocks.sort_by_key(|&(st, _)| st);
        mask_blocks.sort_by_key(|&(st, _)| st);
        if s.dna_size.get().is_none() {
            let _ = s.dna_size.set(dna_size);
        }
        Ok(Rec { dna_size, n_blocks, mask_blocks, packed })
    }

    /// Full record, parsed in 3 preads and cached: [dna_size, nBlockCount], then
    /// [N starts, N ends, maskBlockCount], then [mask starts, mask ends].
    /// Only successes are cached: a failed parse is retried once per call and caches nothing, so one
    /// transient read error does not poison the record for the life of the Genome.
    fn record(&self, i: usize) -> std::io::Result<&Rec> {
        let s = &self.seqs[i];
        if let Some(r) = s.rec.get() {
            return Ok(r);
        }
        let mut r = self.parse_record(s);
        if r.is_err() {
            r = self.parse_record(s);
        }
        match r {
            Ok(rec) => {
                let _ = s.rec.set(rec); // lost race => winner's identical value stands
                Ok(s.rec.get().unwrap())
            }
            Err(e) => Err(e),
        }
    }

    pub fn names(&self) -> Vec<&str> {
        self.seqs.iter().map(|s| s.name.as_str()).collect()
    }

    /// dna_size, or Ok(None) for an unknown name; Err on I/O failure.
    pub fn len(&self, name: &str) -> std::io::Result<Option<u64>> {
        let Some(&i) = self.idx.get(name) else { return Ok(None) };
        Ok(Some(self.seq_len(i)? as u64))
    }



    fn overwrite(blocks: &[(u32, u32)], start: u64, end: u64, out: &mut [u8], mask: bool) {
        if blocks.is_empty() || start >= end {
            return;
        }
        let hi = blocks.partition_point(|&(s, _)| (s as u64) < end);
        let mut lo = blocks.partition_point(|&(s, _)| (s as u64) < start);
        while lo > 0 && (blocks[lo - 1].0 as u64 + blocks[lo - 1].1 as u64) > start {
            lo -= 1;
        }
        for &(s, sz) in &blocks[lo..hi] {
            let a = (s as u64).max(start);
            let b = (s as u64 + sz as u64).min(end);
            if a < b {
                for p in a..b {
                    let c = &mut out[(p - start) as usize];
                    if mask {
                        if *c != b'N' {
                            c.make_ascii_lowercase();
                        }
                    } else {
                        *c = b'N';
                    }
                }
            }
        }
    }

    /// Sequence bytes over [start, end): upper case, lower case where soft-masked, N over N-blocks.
    /// Reads only the packed bytes the window touches.
    /// Ok(None) = unknown name or bad coordinates; Err = I/O or parse failure.
    pub fn fetch(&self, name: &str, start: u64, end: u64) -> std::io::Result<Option<Vec<u8>>> {
        let Some(&i) = self.idx.get(name) else { return Ok(None) };
        let r = self.record(i)?;
        if start > end || end > r.dna_size as u64 {
            return Ok(None);
        }
        if start == end {
            return Ok(Some(Vec::new()));
        }
        let (a, b) = (start / 4, end.div_ceil(4));
        let raw = self.read_at(r.packed + a, (b - a) as usize)?;
        let mut out = Vec::with_capacity((end - start) as usize);
        for pos in start..end {
            let byte = raw[(pos / 4 - a) as usize];
            out.push(match (byte >> ((3 - (pos % 4)) * 2)) & 3 {
                0 => b'T',
                1 => b'C',
                2 => b'A',
                _ => b'G',
            });
        }
        Self::overwrite(&r.n_blocks, start, end, &mut out, false);
        Self::overwrite(&r.mask_blocks, start, end, &mut out, true);
        Ok(Some(out))
    }
}

/// Parse `count` index entries from the in-memory index-region image.
/// `Ok` on success; `Err(true)` if an entry runs past the buffer (caller grows and re-reads),
/// `Err(false)` on malformed data.
fn parse_index(buf: &[u8], count: usize, le: bool) -> Result<Vec<(String, u64)>, bool> {
    let mut out = Vec::with_capacity(count);
    let mut p = 0usize;
    for _ in 0..count {
        if p + 1 > buf.len() {
            return Err(true);
        }
        let ns = buf[p] as usize;
        p += 1;
        if p + ns + 4 > buf.len() {
            return Err(true); // entry runs past the buffer: caller grows and re-reads
        }
        let name = match std::str::from_utf8(&buf[p..p + ns]) {
            Ok(s) => s.to_string(),
            Err(_) => return Err(false),
        };
        p += ns;
        let o = get_u32(&buf[p..], le) as u64;
        p += 4;
        out.push((name, o));
    }
    Ok(out)
}
