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

# Kokoro via a pre-converted OpenVINO IR model (Echo9Zulu/Kokoro-82M-FP16-
# OpenVINO, the same one the actively-maintained OpenArc project uses) run
# through the native `openvino` package -- NOT PyTorch's native XPU backend
# (segfaults the B580's blitter engine on every inference, confirmed
# 2026-07-01) and NOT ONNX Runtime's OpenVINOExecutionProvider (rejects this
# model's ONNX export outright: Interpolate/STFT op support gaps in the GPU
# plugin, also confirmed 2026-07-01). The native IR path actually works, but
# needs INFERENCE_PRECISION_HINT=f32 set at runtime (see app/tts.py) --
# without it, GPU output is all-NaN.
#
# `kokoro` (PyTorch, CPU-only wheel) + misaki are kept ONLY for their
# tokenizer/G2P (KPipeline(model=False) skips building the actual model) --
# actual audio synthesis runs through the OpenVINO IR model, not PyTorch.
# en_core_web_sm is misaki's English G2P model; installing it at build time
# avoids it auto-downloading over the network on first request.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "kokoro>=0.9.4" "misaki[en]>=0.9.4" openvino huggingface_hub \
    && pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

COPY requirements-arc.txt requirements.txt
RUN pip install -r requirements.txt

COPY app/ /app/

EXPOSE 7777
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:7777/healthz || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777"]
