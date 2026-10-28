"""Local speech-to-text with faster-whisper (replaces the Google Web Speech API of the first version)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str
    language_probability: float


class Transcriber:
    """Whisper on CPU, int8. `model_size`: tiny, base, small... (multilingual checkpoints)."""

    def __init__(self, model_size: str = "base", compute_type: str = "int8", cpu_threads: int = 0):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """`audio`: mono 16 kHz float32. `language=None` lets Whisper detect it."""
        segments, info = self.model.transcribe(
            audio.astype(np.float32, copy=False),
            language=language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(text, info.language, float(info.language_probability))
