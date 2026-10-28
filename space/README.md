---
title: EmotionLLM
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Voice emotion (CNN on MFCCs) steering a small LLM's reply
preload_from_hub:
  - Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf
  - Systran/faster-whisper-base
---

Space of [Luca-CS/EmotionLLM](https://github.com/Luca-CS/EmotionLLM): record or upload speech, a CNN on MFCCs guesses the
emotion of the voice, Whisper (local, CPU) transcribes it and a quantised Qwen2.5-1.5B answers in a tone adapted to the
emotion. See the GitHub README for the method, the measured accuracy and the limits.
