# syntax=docker/dockerfile:1
#
# Ancestra 1.0.0 release image: the CLI, its pinned GPU runtime and the sealed weights. No ENTRYPOINT (Nextflow runs
# its own scripts); any UID; read-only image; GPUs through the host driver.
#
# Build (context = the repository root; CI passes the commit, whether the checkout is clean, and the commit time):
#   docker build -f assets/container/Dockerfile -t "$IMAGE" \
#     --build-arg ANCESTRA_GIT_SHA="$(git rev-parse HEAD)" \
#     --build-arg ANCESTRA_GIT_DIRTY="$(test -z "$(git status --porcelain)" && echo 0 || echo 1)" \
#     --build-arg SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" .
# Inputs:
#   ANCESTRA_GIT_SHA      required, the 40-hex commit; the binding's stamp (`ancestra -V`, every output) carries it.
#   ANCESTRA_GIT_DIRTY    0 when the context is exactly that commit; default 1, so the stamp says ",dirty".
#   SOURCE_DATE_EPOCH     the commit time; with it the wheel is byte-reproducible.
#   ANCESTRA_WEIGHTS_URL  the weights, fetched at build time and refused unless their sha256 is RELEASE.json's.
#                         Default: the Hugging Face repo alejandrogzi/ancestra, revision v1.0.0.
# Run (IMAGE = the tag CI pushes, e.g. ghcr.io/hillerlab/ancestra:1.0.0):
#   docker run --rm --gpus all -u "$(id -u):$(id -g)" -v "$PWD:$PWD" -w "$PWD" "$IMAGE" \
#     ancestra -t genome.2bit -o out/genome --devices 0
#   apptainer exec --nv "docker://$IMAGE" ancestra -t genome.2bit -o out/genome --devices 0
#   Nextflow: container 'ghcr.io/hillerlab/ancestra:1.0.0' (the IMAGE) in the process, and in nextflow.config
#             docker.runOptions = '--gpus all' or apptainer.runOptions = '--nv'.

# --- the wheel: the tested 1.0.0 recipe (pypa manylinux_2_28, Rust 1.92.0, maturin 1.15.0, remapped paths) ----------
FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:4131ddd7acb9c6c966af02fe7e71edc51e078c2d41f475966697676430bfa2ae AS build
ENV CARGO_HOME=/cargo/cargo-home RUSTUP_HOME=/cargo/rustup-home PATH=/cargo/cargo-home/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf -o /tmp/rustup-init \
      https://static.rust-lang.org/rustup/archive/1.29.2/x86_64-unknown-linux-gnu/rustup-init \
 && echo "5be191e8c70a5007074254cb74d78dbfdd25cd5a61c27de7b8e5410122ec24f4  /tmp/rustup-init" > /tmp/rustup-init.sha256 \
 && sha256sum -c /tmp/rustup-init.sha256 \
 && chmod +x /tmp/rustup-init \
 && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.92.0 \
 && rm /tmp/rustup-init /tmp/rustup-init.sha256 \
 && /opt/python/cp311-cp311/bin/python -m venv /opt/maturin \
 && /opt/maturin/bin/pip install --no-cache-dir maturin==1.15.0
WORKDIR /src
COPY Cargo.toml Cargo.lock pyproject.toml LICENSE ./
COPY src ./src
ARG ANCESTRA_GIT_SHA
ARG ANCESTRA_GIT_DIRTY=1
ARG SOURCE_DATE_EPOCH
RUN case "$ANCESTRA_GIT_SHA" in *[!0-9a-f]* | "") false ;; esac && [ "${#ANCESTRA_GIT_SHA}" -eq 40 ] \
      || { echo "ANCESTRA_GIT_SHA must be the 40-hex commit, got '$ANCESTRA_GIT_SHA'" >&2; exit 1; } \
 && RUSTFLAGS="--remap-path-prefix=/src=. --remap-path-prefix=$CARGO_HOME/registry=/registry --remap-path-prefix=$CARGO_HOME=/cargo" \
    /opt/maturin/bin/maturin build --release --locked --interpreter /opt/python/cp311-cp311/bin/python --out /dist

# --- the image ------------------------------------------------------------------------------------------------------
# python:3.11.11-slim-bookworm, its linux/amd64 manifest
# (index sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5)
FROM python:3.11.11-slim-bookworm@sha256:a8e0a3090316aed0b11037aac613aef32fb1747dcc1dcb5c0f6c727a0113a07f AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ps, for Nextflow's task metrics (bash and coreutils are in the base).
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps \
 && rm -rf /var/lib/apt/lists/*

# Triton compiles its CUDA launchers with the system C compiler at run time (triton/runtime/build.py _build).
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

# The tested runtime, pinned exactly: torch's own metadata pins triton, cuda-bindings and the nvidia-* wheels; the
# rest are pinned here. triton comes from the PyTorch index (its build differs from PyPI's). Then delete files that
# nothing in the image names (no ELF NEEDED entry, no dlopen string, no import): NVRTC's .alt variants, cuSOLVER's
# multi-GPU library, and Triton's CUPTI copy and instrumentation plugins, which only its proton profiler and
# TRITON_INSTRUMENTATION_MODE load. Re-check them on any torch, triton or nvidia version change.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
      torch==2.10.0+cu128 \
      "triton @ https://download-r2.pytorch.org/whl/triton-3.6.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=e021e87e8f266c6f87bf379b669c43c2800aac6247bdf5d518cb553e3c403c00" \
      numpy==2.4.6 einops==0.8.2 packaging==26.3 cuda-pathfinder==1.6.0 filelock==3.32.3 fsspec==2026.7.0 \
      jinja2==3.1.6 markupsafe==3.0.3 mpmath==1.3.0 networkx==3.6.1 sympy==1.14.0 typing-extensions==4.16.0 \
 && sp=/usr/local/lib/python3.11/site-packages \
 && rm "$sp/nvidia/cuda_nvrtc/lib/libnvrtc.alt.so.12" "$sp/nvidia/cuda_nvrtc/lib/libnvrtc-builtins.alt.so.12.8" \
       "$sp/nvidia/cusolver/lib/libcusolverMg.so.11" \
 && rm -r "$sp/triton/backends/nvidia/lib/cupti" "$sp/triton/instrumentation"

RUN --mount=type=bind,from=build,source=/dist,target=/dist \
    pip install --no-cache-dir --no-deps /dist/ancestra-1.0.0-cp311-abi3-manylinux_2_28_x86_64.whl

# The weights, baked in (compute nodes are often offline); a fetch error or a sha256 other than RELEASE.json's fails.
ARG ANCESTRA_WEIGHTS_URL=https://huggingface.co/alejandrogzi/ancestra/resolve/v1.0.0/ancestra-1.0.0.weights.pt
RUN python - "$ANCESTRA_WEIGHTS_URL" /opt/ancestra/ancestra-1.0.0.weights.pt <<'PY'
import hashlib, json, os, sys, urllib.request
from ancestra import DATA_DIR
url, dst = sys.argv[1:]
want = json.load(open(os.path.join(DATA_DIR, "RELEASE.json")))["weights_sha256"]
os.makedirs(os.path.dirname(dst))
h = hashlib.sha256()
with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
    for block in iter(lambda: r.read(1 << 20), b""):
        h.update(block)
        f.write(block)
if h.hexdigest() != want:
    sys.exit(f"{url}: sha256 {h.hexdigest()} != RELEASE.json weights_sha256 {want}")
PY
ENV ANCESTRA_WEIGHTS=/opt/ancestra/ancestra-1.0.0.weights.pt

# Caches under /tmp, one private directory per UID: the image is read-only, HOME may be unwritable or unmounted, and
# users who share the host's /tmp (Apptainer binds it) must not share a cache, since Triton loads compiled launchers
# from it. Values the caller sets win.
COPY <<'PY' /usr/local/lib/python3.11/site-packages/sitecustomize.py
import os

_caches = (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("XDG_CACHE_HOME", "cache"))
if any(k not in os.environ for k, _ in _caches):
    _root = "/tmp/ancestra-%d" % os.getuid()
    try:
        os.mkdir(_root, 0o700)
    except FileExistsError:
        if os.lstat(_root).st_uid != os.getuid():
            raise SystemExit(f"{_root} belongs to another user: set TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR "
                             "and XDG_CACHE_HOME")
    for k, sub in _caches:
        os.environ.setdefault(k, os.path.join(_root, sub))
PY

# Build-time smoke, no GPU: the stamp names this commit and the sealed weights, the runtime imports, the caches resolve.
ARG ANCESTRA_GIT_SHA
LABEL org.opencontainers.image.source=https://github.com/hillerlab/ancestra \
      org.opencontainers.image.version=1.0.0 \
      org.opencontainers.image.revision=$ANCESTRA_GIT_SHA
RUN stamp="$(ancestra -V)" && echo "$stamp" \
 && case "$stamp" in "ancestra 1.0.0 git=$ANCESTRA_GIT_SHA "*) ;; \
      *) echo "stamp does not name $ANCESTRA_GIT_SHA" >&2; exit 1 ;; esac \
 && python -c "import os, torch, triton; \
assert os.environ['TRITON_CACHE_DIR'] == '/tmp/ancestra-%d/triton' % os.getuid()" \
 && rm -r "/tmp/ancestra-$(id -u)"
