import pytest

from emotionllm.asr import Transcript
from emotionllm.inference import EmotionResult
from emotionllm.llm import ChatBackend, create_backend
from emotionllm.model import LABELS
from emotionllm.pipeline import VoiceAssistant
from emotionllm.prompt import BASE_PROMPT, TONES, build_messages, system_prompt, tone_for

from .conftest import silence, speech_like


def test_every_label_has_a_tone():
    assert set(TONES) == set(LABELS)
    assert tone_for("Angry") == "calm" and tone_for("Sad") == "empathetic"
    assert tone_for("Frustrated") == "reassuring" and tone_for("Excited") == "enthusiastic"
    assert tone_for("Happy") == "positive"


def test_neutral_prompt_adds_no_tone_instruction():
    assert system_prompt("Neutral") == BASE_PROMPT
    assert system_prompt("Unknown label") == BASE_PROMPT
    assert "sad" in system_prompt("Sad") and "empathetic" in system_prompt("Sad")


def test_build_messages():
    msgs = build_messages("  I lost my keys.  ", "Frustrated")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "I lost my keys."


class FakeClassifier:
    def __init__(self, label="Sad", reason=""):
        self.label, self.reason = label, reason

    def classify(self, audio, sr, threshold=None):
        return EmotionResult(
            label=self.label,
            top_label=self.label,
            confidence=0.8,
            probs={n: 0.0 for n in LABELS},
            uncertain=False,
            reason=self.reason,
            n_windows=1,
            speech_seconds=1.0,
        )


class FakeASR:
    def __init__(self, text="hello there"):
        self.text, self.seen = text, None

    def transcribe(self, audio, language=None):
        self.seen = audio
        return Transcript(self.text, "en", 0.99)


class FakeLLM(ChatBackend):
    def __init__(self):
        self.messages = None

    def generate(self, messages, max_new_tokens=200, temperature=0.6):
        self.messages = messages
        return "I am sorry to hear that."

    def stream(self, messages, max_new_tokens=200, temperature=0.6):
        self.messages = messages
        yield "I am sorry"
        yield "I am sorry to hear that."


def test_turn_stages_in_order_and_prompt_uses_emotion():
    llm = FakeLLM()
    assistant = VoiceAssistant(FakeClassifier("Sad"), FakeASR(), llm)
    turns = list(assistant.respond(speech_like(2.0)))
    assert [t.stage for t in turns] == ["emotion", "transcript", "reply", "reply", "done"]
    assert turns[0].tone == "empathetic" and turns[-1].reply == "I am sorry to hear that."
    assert "empathetic" in llm.messages[0]["content"] and llm.messages[1]["content"] == "hello there"


def test_asr_gets_16k_audio_whatever_the_input_rate():
    asr = FakeASR()
    two_seconds_at_8k = speech_like(2.0)[::2]
    list(VoiceAssistant(FakeClassifier(), asr, FakeLLM()).respond(two_seconds_at_8k, sr=8000))
    assert len(asr.seen) == 2 * 16_000


def test_no_speech_stops_before_asr_and_llm():
    asr, llm = FakeASR(), FakeLLM()
    turns = list(VoiceAssistant(FakeClassifier("Neutral", "no speech"), asr, llm).respond(silence(1.0)))
    assert [t.stage for t in turns] == ["no_speech"] and asr.seen is None and llm.messages is None


def test_empty_transcript_stops_before_llm():
    llm = FakeLLM()
    turns = list(VoiceAssistant(FakeClassifier(), FakeASR(text=""), llm).respond(speech_like(1.0)))
    assert turns[-1].stage == "no_transcript" and llm.messages is None


def test_default_stream_yields_the_whole_reply():
    class Plain(ChatBackend):
        def generate(self, messages, max_new_tokens=200, temperature=0.6):
            return "ok"

    assert list(Plain().stream([])) == ["ok"]


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        create_backend("gpt-17")
