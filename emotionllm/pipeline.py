"""One conversational turn: audio -> emotion -> transcript -> tone-adapted reply.

The turn is a generator of snapshots so that an interface can show each stage as soon as it is ready
(emotion in milliseconds, transcript in seconds, reply token by token).
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .asr import Transcript
from .audio import resample, to_mono_float
from .features import SAMPLE_RATE
from .inference import EmotionClassifier, EmotionResult
from .llm import ChatBackend
from .prompt import build_messages, tone_for


@dataclass
class Turn:
    stage: str  # "emotion", "transcript", "reply", "done", "no_speech" or "no_transcript"
    emotion: EmotionResult | None = None
    transcript: Transcript | None = None
    tone: str = ""
    reply: str = ""


class VoiceAssistant:
    def __init__(self, classifier: EmotionClassifier, transcriber, backend: ChatBackend, max_new_tokens: int = 160):
        self.classifier = classifier
        self.transcriber = transcriber
        self.backend = backend
        self.max_new_tokens = max_new_tokens

    def respond(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE, threshold: float | None = None, language: str | None = None
    ) -> Iterator[Turn]:
        emotion = self.classifier.classify(audio, sr, threshold=threshold)
        if emotion.reason == "no speech":
            yield Turn("no_speech", emotion)
            return
        tone = tone_for(emotion.label)
        yield Turn("emotion", emotion, tone=tone)

        transcript = self.transcriber.transcribe(self._to_16k(audio, sr), language=language)
        if not transcript.text:
            yield Turn("no_transcript", emotion, transcript, tone)
            return
        yield Turn("transcript", emotion, transcript, tone)

        messages = build_messages(transcript.text, emotion.label)
        reply = ""
        for reply in self.backend.stream(messages, max_new_tokens=self.max_new_tokens):
            yield Turn("reply", emotion, transcript, tone, reply)
        yield Turn("done", emotion, transcript, tone, reply)

    @staticmethod
    def _to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
        return resample(to_mono_float(audio), sr, SAMPLE_RATE)
