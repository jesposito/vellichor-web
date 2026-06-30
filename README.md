# Vellichor 🎧

A self-hosted web app that turns your written stories (and ebooks) into
narrated audiobooks using the **Kokoro-82M** TTS model. GPU-accelerated and
light enough to run on a modest card (originally built on a GTX 1080); falls
back to CPU. Built for writing & converting stories. Open source (MIT).

## Requirements
- **Docker** and **Docker Compose**.
- **(Optional) NVIDIA GPU** for acceleration — requires the NVIDIA Container
  Toolkit on the host (on Unraid, the **Nvidia Driver** plugin). **(Optional)
  Intel Arc GPU** (Alchemist/Battlemage) is also supported via a separate
  build — see *Intel Arc build (OpenVINO)* below. With no GPU it runs on CPU
  instead — see the no-GPU note in *Getting started*.
- ~6 GB of disk for the image plus models (Kokoro + the Ollama LLM, downloaded
  on first use).

## Getting started
```bash
# 1. Clone
git clone https://github.com/woodscode/vellichor-web.git
cd vellichor-web

# 2. Create your .env: set a login password and a cookie-signing key
cp .env.example .env
sed -i "s|^SECRET_KEY=.*|SECRET_KEY=$(openssl rand -hex 32)|" .env
$EDITOR .env                       # set VELLICHOR_PASSWORD

# 3. Edit docker-compose.yml for your box:
#    - Audiobookshelf export mount (…:/library) — repoint to your library, or
#      remove the volume if you don't use Audiobookshelf
#    - host port (default 7777:7777)
#    - NO GPU? remove the `runtime: nvidia` and `NVIDIA_*` lines from BOTH
#      services (it then runs on CPU — slower, but works)

# 4. Build & start
docker compose up -d --build

# 5. (Optional) enable AI Smart cast — pull the local LLM once
docker exec vellichor-ollama ollama pull llama3.2:3b
```
Then open **http://<server-ip>:7777** and log in with your `VELLICHOR_PASSWORD`.
Store that password in your password manager.

## Features
- Built-in **story editor** (type/paste, `#` lines become chapters) + upload
  `.txt`, `.md`, `.epub`, `.pdf`, `.docx`.
- **Voice picker** with 35 voices, grouped/filterable, each with a ▶ sample.
  Story-friendly voices are starred (★). `af_heart` is the default.
- **Live preview** — hear your chosen voice read the current text before
  committing to a full conversion.
- Reading-speed slider, optional cover art, author label.
- **Live progress** (stage, segment count, ETA) + per-job log.
- **🎭 Multi-voice cast** — give each character their own voice. Three ways:
  - **🪄 Smart cast (AI)** — a local Ollama model (Llama 3.2 3B) reads the story,
    attributes each line to a speaker, and auto-inserts `[Name]` tags for you to
    review. Best for messy/untagged dialogue. Falls back to Quick detect if the
    model isn't ready.
  - **🔎 Quick detect** — fast rule-based: quotes + dialogue tags, with gender
    inference (honorifics + pronouns) to pick matching-gender voices.
  - **`[Name]` markup** — tag speakers yourself for exact control, e.g.
    `[Pip] "I can do it!"`.
  The cast panel lets you assign/preview a voice per character before converting.
  Works for **uploaded files too**: multi-voice auto-assigns distinct gender-matched
  voices to detected characters with zero setup, and 🔎/🪄 read the file's text so
  you can review/override the cast first.
- **🎵 Background ambience** — mix a bed under the narration: built-in
  license-free beds (Soft Rain, Gentle Night, Warm Hum, Dreamy Pad), or upload
  your own / drop files in `data/ambience/`. Volume slider + auto-ducking
  (music dips under speech).
- Output: **chaptered M4B** + **per-chapter MP3** (zip). Both downloadable.
- **Auto-export** to an Audiobookshelf library (path + owner UID/GID
  configurable; see `docker-compose.yml`).
- **🎨 Themes** — Dark, Light, Sepia, and Midnight, switchable from the header
  and remembered per browser.
- Password login; job history survives restarts.

## Managing it
```bash
cd vellichor-web                # your project directory
docker compose up -d            # start / apply changes
docker compose logs -f          # watch logs
docker compose down             # stop
docker compose up -d --build    # rebuild after editing app/ code
```

## AI Smart cast (Ollama)
Smart cast is **optional** — without it, multi-voice casting uses the
rule-based **Quick detect** instead. To enable it, the `ollama` service (in
docker-compose) runs a local LLM. After the first `docker compose up -d`, pull
the model once:
```bash
docker exec vellichor-ollama ollama pull llama3.2:3b
```
Both models share the GPU; `OLLAMA_KEEP_ALIVE=2m` unloads the LLM from VRAM
after use so Kokoro has room (on an 8 GB card they can't both stay resident).
To try a more accurate (heavier) model, pull it and set `SMARTCAST_MODEL` in
`.env` (e.g. `qwen2.5:7b`), then `up -d`.

## Intel Arc build (OpenVINO)
This fork adds a second build path for Intel Arc GPUs (Alchemist/Battlemage,
e.g. the A380/B580). It runs a pre-converted OpenVINO IR build of Kokoro-82M
(`Echo9Zulu/Kokoro-82M-FP16-OpenVINO`, the same model the actively-maintained
[OpenArc](https://github.com/SearchSavior/OpenArc) project uses) through the
native `openvino` Python package. `kokoro` + `misaki` (CPU-only PyTorch
build) are still installed, but only for their tokenizer/G2P — actual audio
synthesis runs through the OpenVINO model, not PyTorch.

**Why this specific path** — two other approaches were tried and rejected
first, confirmed on a real B580 (2026-07-01):
- **PyTorch's native XPU backend**: `torch.xpu.is_available()` detects the
  GPU fine, but the moment Kokoro actually runs inference on it, it
  segfaults and resets the GPU's blitter engine
  (`xe ...: Engine reset: engine_class=bcs`), every time.
- **ONNX Runtime + OpenVINOExecutionProvider**, using `kokoro-onnx`'s
  published ONNX exports: rejected outright at graph-compile time — an
  `Interpolate` op with an unsupported tensor rank, and (with `HETERO`/`AUTO`
  partial-offload) a dynamic-rank `STFT` node neither the GPU nor CPU
  OpenVINO plugin would accept. A *third* independent attempt at converting
  Kokoro for OpenVINO (`Echo9Zulu/Kokoro-82M-FP16-OpenVINO`'s own model card)
  documents yet another distinct unsupported op (`ScatterNDUpdate`/INT64) and
  ships as CPU-only for that reason.
- **What actually worked**: the same `Echo9Zulu/Kokoro-82M-FP16-OpenVINO` IR
  model *does* compile and run on GPU via the native `openvino` package
  (not `onnxruntime`) — but raw GPU output was 100% NaN until setting
  `INFERENCE_PRECISION_HINT: f32` on `core.compile_model(...)` (forces FP32
  compute even though the IR is stored FP16 — a known fix for this failure
  class). After a one-time ~5s JIT warmup (paid once at startup), GPU
  inference is ~20-25% faster than CPU for the same input, and a real
  CPU-vs-GPU audio comparison confirmed by ear that GPU output is correct
  (not garbled) — RMS energy was within ~4% of the CPU reference despite a
  large sample-level numerical divergence (expected: LSTM-based duration
  predictor, floating-point execution-order differences compound over many
  timesteps without affecting perceptual quality).

```bash
docker compose -f docker-compose.arc.yml up -d --build
```

- Uses `Dockerfile.arc` — no PyTorch CUDA/XPU wheel; just the CPU PyTorch
  wheel (tokenizer only) + native `openvino`.
- Passes `/dev/dri` through instead of `runtime: nvidia` — no Nvidia Driver
  plugin needed, just the host having Arc's kernel driver (i915/xe) loaded.
- Downloads the OpenVINO IR model + vocab + per-voice style vectors on first
  use, cached under `./data/ov-cache/`.
- Smart cast points at an **external** Ollama-compatible endpoint
  (`OLLAMA_URL`, default `http://ipex-ollama:11434`) reachable on the
  `ansiblenet`-style network you configure — there's no bundled Ollama
  service in this compose file. Ollama itself has no mainline Intel Arc
  backend, so use a real GPU-backed endpoint (e.g. `intelanalytics/ipex-llm`)
  if you want Smart cast to actually use the GPU; otherwise point it at any
  CPU-only Ollama.
- The `⚡ GPU` chip in the header reads `OPENVINO` when the GPU compiles and
  passes a finite-output warmup check; it falls back to CPU (`device: cpu`)
  cleanly if either fails, rather than crashing.
- The Intel driver `.deb` pins in `Dockerfile.arc` are the same versions
  used by Immich's official `-openvino` image — proven stable on a real B580
  as of 2026-07-01, but check
  github.com/intel/compute-runtime/releases for newer Battlemage fixes if
  you hit GPU issues.

## Configuration (`.env`)
- `VELLICHOR_PASSWORD` — login password (change anytime, then `up -d`).
- `SECRET_KEY` — session-cookie signing key (don't change or logins reset).
- `NOTIFY_URL` — optional. Set to an ntfy/gotify URL to get a push when a
  conversion finishes, e.g. `http://<server-ip>:8087/vellichor`.

## Data
- `./data/` — uploads, job workdirs, job history (`jobs.json`), cached voice
  samples (`samples/`), and the Hugging Face model cache (`hf-cache/`).
- Models download on first use and are cached in `./data/hf-cache`.

## Notes
- GPU is used automatically (`⚡ GPU` chip in the header). Falls back to CPU if
  the NVIDIA runtime is unavailable.
- Conversions run one at a time (single worker) so the GPU isn't oversubscribed.

## Security & deployment
- **Set a password.** Auth is a single shared password (`VELLICHOR_PASSWORD`).
  If it's left **blank, authentication is disabled entirely** — only do that on
  a trusted private network.
- **Don't expose it directly to the internet.** This is a self-hosted personal
  tool with a single-password gate, not a hardened multi-user service. If you
  need remote access, put it behind a reverse proxy (Nginx Proxy Manager,
  Traefik, Caddy) with HTTPS and ideally an extra auth layer (e.g. Authelia).
- **Keep `.env` private** (`chmod 600`). It holds your password and
  `SECRET_KEY` and is gitignored — never commit it.
- **`SECRET_KEY`** signs the session cookie. Generate one with
  `openssl rand -hex 32`. Changing it invalidates existing logins.
- **Uploaded files** (epub/pdf/docx) are parsed server-side; only allow
  uploads from people you trust.

## License
[MIT](LICENSE) — free to use, modify, and redistribute. TTS by
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0).
