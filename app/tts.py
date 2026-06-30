"""Kokoro TTS engine wrapper: pipeline caching, text chunking, synthesis,
and on-demand voice sample generation.

Two interchangeable backends live behind the same Engine API, picked by
whichever package the image installed (only one is ever present):
  - `kokoro` (PyTorch) -- the Nvidia/CPU build (Dockerfile)
  - `kokoro_onnx` (ONNX Runtime + OpenVINO GPU plugin) -- the Intel Arc build
    (Dockerfile.arc). PyTorch's native XPU backend segfaults the B580's
    blitter engine on this box (confirmed 2026-07-01); OpenVINO is the GPU
    access path already proven stable here (Immich's ML container)."""
import os
import re
import threading
import urllib.request
import numpy as np
import soundfile as sf

import voices as voicecat

SAMPLE_RATE = 24000
SAMPLES_DIR = "/data/samples"
ONNX_CACHE_DIR = "/data/onnx-cache"

try:
    from kokoro_onnx import Kokoro as _KokoroONNX
    _BACKEND = "onnx"
except ImportError:
    _BACKEND = "torch"

# kokoro-onnx's `lang` expects espeak-style locale codes; Kokoro's voice-id
# prefix scheme (voicecat.lang_code) uses single letters. Only languages
# actually in the voice catalog are mapped.
_ONNX_LANG = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr",
              "h": "hi", "i": "it", "p": "pt-br"}

_ONNX_MODEL_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/"
                    "download/model-files-v1.0/kokoro-v1.0.fp16-gpu.onnx")
_ONNX_VOICES_URL = ("https://github.com/thewh1teagle/kokoro-onnx/releases/"
                     "download/model-files-v1.0/voices-v1.0.bin")


class Engine:
    def __init__(self):
        self._pipelines = {}
        self._lock = threading.Lock()
        self.device = "cpu"
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        if _BACKEND == "onnx":
            self._init_onnx()
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                pass

    @staticmethod
    def _fetch(url, dest):
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return
        tmp = dest + ".part"
        print(f"[tts] downloading {url} -> {dest}", flush=True)
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, dest)

    def _init_onnx(self):
        """Fetch the ONNX weights once (cached on the /data volume) and build
        an onnxruntime session pinned to the OpenVINO GPU execution provider,
        falling back to CPU if the GPU plugin fails to initialize."""
        import onnxruntime as rt
        os.makedirs(ONNX_CACHE_DIR, exist_ok=True)
        model_path = os.path.join(ONNX_CACHE_DIR, "kokoro-v1.0.fp16-gpu.onnx")
        voices_path = os.path.join(ONNX_CACHE_DIR, "voices-v1.0.bin")
        self._fetch(_ONNX_MODEL_URL, model_path)
        self._fetch(_ONNX_VOICES_URL, voices_path)
        try:
            session = rt.InferenceSession(
                model_path,
                providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
                provider_options=[{"device_type": "GPU"}, {}],
            )
            if session.get_providers()[0] == "OpenVINOExecutionProvider":
                self.device = "openvino"
        except Exception as e:  # noqa: BLE001 — fall back to CPU EP
            print(f"[tts] OpenVINO GPU session failed, falling back to CPU: {e}", flush=True)
            session = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._kokoro = _KokoroONNX.from_session(session, voices_path)

    def pipeline(self, lang_code: str):
        """PyTorch backend only — kokoro-onnx uses a single session, no
        per-language pipeline cache."""
        with self._lock:
            if lang_code not in self._pipelines:
                from kokoro import KPipeline
                self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
            return self._pipelines[lang_code]

    def unload(self):
        """Drop cached pipelines and release GPU memory, so another model
        (Ollama, for Smart cast) can claim the VRAM. Pipelines reload lazily
        on the next synth. No-op on the OpenVINO backend: an 82M-param FP16
        session is small, and Smart cast there talks to an externally-managed
        Ollama instance with its own GPU memory lifecycle, not this process."""
        if _BACKEND == "onnx":
            return
        with self._lock:
            self._pipelines.clear()
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def clean_speech_text(text: str) -> str:
        """Strip Markdown markup so the TTS never vocalizes it (e.g. reading a
        heading '#' as 'hashtag', or '*' as 'asterisk'). Conservative: only
        removes formatting characters, never words or sentence punctuation."""
        if not text:
            return text
        # ATX headings: leading #'s at the start of a line (with or w/o a space)
        text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*", "", text)
        # setext heading underlines (=== / --- on their own line)
        text = re.sub(r"(?m)^[ \t]{0,3}[=\-]{3,}[ \t]*$", "", text)
        # emphasis / code markers and blockquote arrows
        text = text.replace("`", "")
        text = re.sub(r"\*{1,3}", "", text)
        text = re.sub(r"(?m)^[ \t]{0,3}>[ \t]?", "", text)
        text = re.sub(r"(?<![A-Za-z0-9])_(?![A-Za-z0-9])", "", text)  # _emphasis_, not in_words
        return text

    @staticmethod
    def chunk_text(text: str, max_chars: int = 500):
        """Split text into synthesis-sized chunks at sentence boundaries."""
        text = text.strip()
        if not text:
            return []
        # split into sentences while keeping terminal punctuation
        sentences = re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
        chunks, cur = [], ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) > max_chars:
                # hard-wrap an over-long sentence on commas/spaces
                for piece in re.findall(r".{1," + str(max_chars) + r"}(?:\s|$)", s):
                    piece = piece.strip()
                    if piece:
                        chunks.append(piece)
                continue
            if len(cur) + len(s) + 1 <= max_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        return chunks

    def synth_chunk(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize one chunk, returning a float32 mono waveform at 24kHz."""
        text = self.clean_speech_text(text)
        if _BACKEND == "onnx":
            lang = _ONNX_LANG.get(voicecat.lang_code(voice), "en-us")
            samples, _sr = self._kokoro.create(text, voice=voice, speed=speed, lang=lang)
            return np.asarray(samples, dtype="float32")
        pipe = self.pipeline(voicecat.lang_code(voice))
        audio_parts = []
        for _, _, audio in pipe(text, voice=voice, speed=speed):
            if audio is None:
                continue
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            audio_parts.append(arr.astype("float32"))
        if not audio_parts:
            return np.zeros(0, dtype="float32")
        return np.concatenate(audio_parts)

    def sample_path(self, voice: str) -> str:
        return os.path.join(SAMPLES_DIR, f"{voice}.mp3")

    def ensure_sample(self, voice: str) -> str:
        """Generate (and cache) a short preview clip for a voice. Returns mp3 path."""
        out = self.sample_path(voice)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
        import gpu
        with gpu.LOCK:
            gpu.release_ollama()        # reclaim VRAM from Smart cast before TTS
            wav = self.synth_chunk(voicecat.SAMPLE_TEXT, voice, speed=1.0)
        tmp_wav = out.replace(".mp3", ".wav")
        sf.write(tmp_wav, wav, SAMPLE_RATE)
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav, "-c:a", "libmp3lame", "-q:a", "5", out],
            check=True, capture_output=True,
        )
        try:
            os.remove(tmp_wav)
        except OSError:
            pass
        return out

    def prewarm(self, voice_ids):
        """Generate samples for a list of voices in the background."""
        for vid in voice_ids:
            try:
                self.ensure_sample(vid)
            except Exception as e:  # noqa: BLE001
                print(f"[prewarm] {vid} failed: {e}", flush=True)


ENGINE = Engine()
