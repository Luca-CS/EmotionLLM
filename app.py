"""Gradio interface (used by the Hugging Face Space): record or upload speech, get emotion, transcript, reply."""

import os

import gradio as gr

from emotionllm.asr import Transcriber
from emotionllm.audio import load_audio
from emotionllm.features import SAMPLE_RATE
from emotionllm.inference import EmotionClassifier
from emotionllm.llm import create_backend
from emotionllm.model import LABELS
from emotionllm.pipeline import Turn, VoiceAssistant

MAX_SECONDS = 30  # CPU-only hosting: longer recordings are cut
LANGUAGES = {"Auto-detect": None, "English": "en", "French": "fr"}

INTRO = """# EmotionLLM
Speak (or upload a clip): a CNN on MFCCs guesses the **emotion of your voice**, Whisper transcribes it locally, and a small
language model answers in a tone adapted to that emotion (angry → calm, sad → empathetic, frustrated → reassuring...).

*CentraleSupélec ST7 project, trained on IEMOCAP (acted English speech). On speakers it has never heard it gets
about 37% of the six classes right (chance: 17%), so the label is a hint, not a diagnosis: the model only commits to an
emotion when it is confident enough, otherwise the reply stays neutral. Measurements and limits are in the
[README](https://github.com/Luca-CS/EmotionLLM#readme). Recordings are processed in memory and are not stored by the app.*"""

STATUS = {
    "no_speech": "No speech detected in the recording.",
    "no_transcript": "Speech detected but nothing could be transcribed.",
}


def decision_text(turn: Turn) -> str:
    emotion = turn.emotion
    if emotion is None:
        return ""
    if emotion.reason == "no speech":
        return STATUS["no_speech"]
    text = f"**{emotion.label}**"
    if emotion.uncertain:
        text += (
            f" (fallback: the model hesitates, its top guess is {emotion.top_label.lower()} at "
            f"{emotion.confidence:.0%}). Tone of the reply: {turn.tone}."
        )
    else:
        text += f" ({emotion.confidence:.0%}). Tone of the reply: {turn.tone}."
    return text + f" Speech analysed: {emotion.speech_seconds:.1f} s in {emotion.n_windows} window(s)."


def render(turn: Turn):
    """Turn -> values of the output components (probabilities, decision, transcript, reply)."""
    probs = turn.emotion.probs if turn.emotion and turn.emotion.reason != "no speech" else None
    decision = decision_text(turn)
    transcript = turn.transcript.text if turn.transcript else STATUS.get(turn.stage, "")
    if turn.transcript and turn.transcript.language:
        transcript = f"[{turn.transcript.language}] {transcript}" if transcript else transcript
    return probs, decision, transcript, turn.reply


def make_handler(assistant: VoiceAssistant):
    def handler(audio_path, threshold, language):
        if audio_path is None:
            yield None, "Record or upload an audio clip first.", "", ""
            return
        audio = load_audio(audio_path)
        audio = audio[: MAX_SECONDS * SAMPLE_RATE]
        for turn in assistant.respond(audio, SAMPLE_RATE, threshold=threshold, language=LANGUAGES[language]):
            yield render(turn)

    return handler


def build_demo(assistant: VoiceAssistant) -> gr.Blocks:
    with gr.Blocks(title="EmotionLLM", delete_cache=(3600, 3600)) as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    sources=["microphone", "upload"], type="filepath", label=f"Speech (up to {MAX_SECONDS} s)"
                )
                threshold = gr.Slider(
                    0.0,
                    0.9,
                    value=assistant.classifier.config.threshold,
                    step=0.05,
                    label="Confidence threshold",
                    info="Below it, the emotion falls back to Neutral.",
                )
                language = gr.Dropdown(list(LANGUAGES), value="Auto-detect", label="Speech language")
                go = gr.Button("Analyse and answer", variant="primary")
            with gr.Column():
                probs = gr.Label(num_top_classes=len(LABELS), label="Emotion probabilities")
                decision = gr.Markdown()
                transcript = gr.Textbox(label="Transcript (Whisper, local)", lines=2)
                reply = gr.Textbox(label="Reply", lines=5)
        go.click(make_handler(assistant), [audio, threshold, language], [probs, decision, transcript, reply])
    demo.queue(default_concurrency_limit=1, max_size=4)
    return demo


def build_assistant() -> VoiceAssistant:
    return VoiceAssistant(
        EmotionClassifier.from_weights(),
        Transcriber(os.environ.get("EMOTIONLLM_WHISPER", "base")),
        create_backend(),
    )


if __name__ == "__main__":
    build_demo(build_assistant()).launch()
