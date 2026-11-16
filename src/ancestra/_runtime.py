"""Process-wide 2bit reader cache."""
import os
import threading

from ancestra import _native

_GENOMES = {}
_GENOME_SIZES = {}
_GENOME_LOCK = threading.Lock()


def genome(path):
    """One 2bit reader per absolute path and file metadata, shared by the whole process."""
    absolute = os.path.abspath(path)
    try:
        stat = os.stat(absolute)
    except OSError:
        return _native.Genome(path)  # Preserve the native reader's path-specific error.
    key = (absolute, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    with _GENOME_LOCK:
        if key not in _GENOMES:
            _GENOMES.clear()
            _GENOME_SIZES.clear()
            _GENOMES[key] = _native.Genome(path)
        return _GENOMES[key]


def chrom_sizes(genome):
    with _GENOME_LOCK:
        for key, reader in _GENOMES.items():
            if reader is genome:
                if key not in _GENOME_SIZES:
                    _GENOME_SIZES[key] = genome.chrom_sizes()
                return _GENOME_SIZES[key]
    return genome.chrom_sizes()
