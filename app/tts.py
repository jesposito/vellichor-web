"""Kokoro TTS engine wrapper: pipeline caching, text chunking, synthesis,
and on-demand voice sample generation.

Two interchangeable backends live behind the same Engine API, picked by
whichever package the image installed (only one is ever present):
  - `kokoro` (PyTorch) -- the Nvidia/CPU build (Dockerfile)
  - native OpenVINO IR -- the Intel Arc build (Dockerfile.arc), using a
    pre-converted Kokoro-82M IR model (Echo9Zulu/Kokoro-82M-FP16-OpenVINO, as
    used by the OpenArc project). Two other approaches were tried and
    rejected first (confirmed 2026-07-01): PyTorch's native XPU backend
    segfaults the B580's blitter engine on every inference, and ONNX
    Runtime's OpenVINOExecutionProvider rejects this model's ONNX export
    outright (Interpolate/STFT op support gaps in the GPU plugin). The
    native OpenVINO IR path actually runs, but needs
    INFERENCE_PRECISION_HINT=f32 on GPU -- without it, output is all-NaN
    (FP16 GPU compute overflows somewhere in this graph). Verified by ear:
    GPU output sounds correct, matches CPU in RMS energy."""
import os
import re
import threading
import numpy as np
import soundfile as sf

import voices as voicecat

SAMPLE_RATE = 24000
SAMPLES_DIR = "/data/samples"
OV_CACHE_DIR = "/data/ov-cache"

try:
    import openvino as ov
    _BACKEND = "openvino"
except ImportError:
    _BACKEND = "torch"

_OV_REPO = "Echo9Zulu/Kokoro-82M-FP16-OpenVINO"
_VOCAB_REPO = "hexgrad/Kokoro-82M"


class Engine:
    def __init__(self):
        self._pipelines = {}        # torch backend: KPipeline per lang_code
        self._quiet_pipelines = {}  # openvino backend: tokenize-only KPipeline per lang_code
        self._voice_cache = {}      # openvino backend: voice pack tensors
        self._lock = threading.Lock()
        self.device = "cpu"
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        if _BACKEND == "openvino":
            self._init_openvino()
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                pass

    def _init_openvino(self):
        """Download the pre-converted IR model + vocab once (cached on the
        /data volume) and compile it for GPU, falling back to CPU if the GPU
        plugin fails to initialize or produces non-finite output."""
        import json
        from huggingface_hub import hf_hub_download
        os.makedirs(OV_CACHE_DIR, exist_ok=True)
        xml_path = hf_hub_download(_OV_REPO, "openvino_model.xml", cache_dir=OV_CACHE_DIR)
        hf_hub_download(_OV_REPO, "openvino_model.bin", cache_dir=OV_CACHE_DIR)
        cfg_path = hf_hub_download(_VOCAB_REPO, "config.json", cache_dir=OV_CACHE_DIR)
        self._vocab = json.load(open(cfg_path))["vocab"]

        core = ov.Core()
        model = core.read_model(xml_path)
        try:
            compiled = core.compile_model(model, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})
            warmup = compiled([np.array([[0, 1, 0]], dtype=np.int64),
                                np.zeros((1, 256), dtype=np.float32),
                                np.array(1.0, dtype=np.float32)])[0]
            if not np.isfinite(warmup).all():
                raise RuntimeError("GPU produced non-finite output on warmup")
            self._compiled = compiled
            self.device = "openvino"
        except Exception as e:  # noqa: BLE001 — fall back to CPU
            print(f"[tts] OpenVINO GPU compile/warmup failed, falling back to CPU: {e}", flush=True)
            self._compiled = core.compile_model(model, "CPU")

    def _quiet_pipeline(self, lang_code: str):
        """OpenVINO backend only -- G2P/tokenization without a PyTorch model
        attached (kokoro's KPipeline(model=False) skips building KModel)."""
        with self._lock:
            if lang_code not in self._quiet_pipelines:
                from kokoro import KPipeline
                self._quiet_pipelines[lang_code] = KPipeline(lang_code=lang_code, model=False)
            return self._quiet_pipelines[lang_code]

    def _voice_pack(self, voice: str):
        """OpenVINO backend only -- per-voice style vectors, shape [510,1,256]
        (max-510-phoneme buckets), downloaded once and cached on /data."""
        with self._lock:
            if voice not in self._voice_cache:
                import torch
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(_OV_REPO, f"voices/{voice}.pt", cache_dir=OV_CACHE_DIR)
                self._voice_cache[voice] = torch.load(path, weights_only=True).numpy()
            return self._voice_cache[voice]

    def pipeline(self, lang_code: str):
        """PyTorch backend only."""
        with self._lock:
            if lang_code not in self._pipelines:
                from kokoro import KPipeline
                self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
            return self._pipelines[lang_code]

    def unload(self):
        """Drop cached pipelines and release GPU memory, so another model
        (Ollama, for Smart cast) can claim the VRAM. Pipelines reload lazily
        on the next synth. No-op on the OpenVINO backend: the compiled model
        stays resident permanently, and Smart cast there talks to an
        externally-managed Ollama instance with its own GPU memory
        lifecycle, not this process."""
        if _BACKEND == "openvino":
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
        if _BACKEND == "openvino":
            pipe = self._quiet_pipeline(voicecat.lang_code(voice))
            ps = None
            for _, phonemes, _ in pipe(text, voice=voice, speed=speed):
                ps = phonemes
                break
            if not ps:
                return np.zeros(0, dtype="float32")
            ids = [i for i in (self._vocab.get(p) for p in ps) if i is not None]
            input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
            ref_s = self._voice_pack(voice)[len(ps) - 1].astype(np.float32)
            speed_arr = np.array(speed, dtype=np.float32)
            audio = self._compiled([input_ids, ref_s, speed_arr])[0]
            return np.asarray(audio, dtype="float32")
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
