"""Serialize the two GPU consumers — Kokoro TTS and the Ollama Smart-cast model
— so they never fight over VRAM at the same time, and so it's handed cleanly
from one to the other.

On a single card, both can't hold the GPU at once: whichever sits resident
leaves too little free VRAM for the other, which then silently falls back to
running entirely on the CPU. Whichever consumer is about to run takes LOCK and
evicts the other from VRAM first, so it gets a full GPU offload instead of CPU
fallback. (Note: on the Intel XPU build, Ollama itself isn't GPU-accelerated —
see docker-compose.xpu.yml — so this mainly matters on the Nvidia build.)
"""
import os
import threading

import requests

# Reentrant so a single thread can nest acquisitions safely; different threads
# (the conversion worker vs. a Smart-cast request) still serialize.
LOCK = threading.RLock()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
MODEL = os.environ.get("SMARTCAST_MODEL", "llama3.2:3b")


def release_kokoro():
    """Drop Kokoro's cached pipelines and free its VRAM (so Ollama can offload
    to the GPU instead of falling back to CPU)."""
    from tts import ENGINE
    ENGINE.unload()


def release_ollama():
    """Ask Ollama to evict the model from VRAM immediately (keep_alive=0) so
    Kokoro can reclaim the GPU. No-op/quick if the model isn't loaded."""
    try:
        requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": MODEL, "keep_alive": 0}, timeout=10)
    except requests.RequestException:
        pass
