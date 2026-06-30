FROM python:3.11-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ffmpeg = audio encoding/m4b assembly; espeak-ng = phonemizer-fork's G2P
# backend; libsndfile1 = soundfile backend. ocl-icd-libopencl1/libze1 = the
# generic OpenCL/Level-Zero loaders the Intel GPU plugin below talks through.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg espeak-ng libsndfile1 git curl ca-certificates wget \
        ocl-icd-libopencl1 libze1 \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU compute userspace (compute-runtime + IGC) for OpenVINO's GPU
# plugin. Intel's apt repo (repositories.intel.com/gpu) 403s (confirmed
# 2026-07-01), so pull pinned .deb packages straight from the GitHub
# releases instead — the same versions immich-machine-learning's official
# -openvino image uses, which is proven stable on this exact Battlemage
# (B580) box. Installs both the current and a legacy compute-runtime build
# side by side (matching upstream) for broader hardware coverage.
RUN set -eux; \
    tmp="$(mktemp -d)"; cd "$tmp"; \
    wget -nv "https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-core-2_2.36.3+21719_amd64.deb"; \
    wget -nv "https://github.com/intel/intel-graphics-compiler/releases/download/v2.36.3/intel-igc-opencl-2_2.36.3+21719_amd64.deb"; \
    wget -nv "https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/intel-opencl-icd_26.22.38646.4-0_amd64.deb"; \
    wget -nv "https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-core_1.0.17537.24_amd64.deb"; \
    wget -nv "https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.24/intel-igc-opencl_1.0.17537.24_amd64.deb"; \
    wget -nv "https://github.com/intel/compute-runtime/releases/download/24.35.30872.36/intel-opencl-icd-legacy1_24.35.30872.36_amd64.deb"; \
    wget -nv "https://github.com/intel/compute-runtime/releases/download/26.22.38646.4/libigdgmm12_22.10.0_amd64.deb"; \
    dpkg -i *.deb; \
    cd / && rm -rf "$tmp"

WORKDIR /app

# Kokoro via ONNX Runtime + the OpenVINO execution provider (GPU plugin) --
# NOT PyTorch's native XPU backend, which segfaults the B580's blitter
# engine on this box (confirmed 2026-07-01, see git history). OpenVINO is
# the same GPU access path already proven stable here (Immich's ML
# container). onnxruntime-openvino bundles its own onnxruntime build, so
# kokoro-onnx's own onnxruntime dependency is skipped (--no-deps) to avoid
# pip installing a second, conflicting onnxruntime package.
RUN pip install onnxruntime-openvino espeakng-loader "phonemizer-fork>=3.3.2" \
    && pip install --no-deps kokoro-onnx

COPY requirements-arc.txt requirements.txt
RUN pip install -r requirements.txt

COPY app/ /app/

EXPOSE 7777
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:7777/healthz || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777"]
