"""Original Tkinter interface, now a thin layer over the `emotionllm` package.

    pip install pyaudio            # microphone access (not needed for the Gradio app)
    python desktop_app.py          # EMOTIONLLM_BACKEND=llama3-8b for the original Llama 3 8B (GPU, gated model)
"""

import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from app import build_assistant
from emotionllm.features import SAMPLE_RATE

CHUNK = 1024


class DesktopApp:
    def __init__(self, root: tk.Tk, assistant):
        self.root, self.assistant = root, assistant
        self.recording = False
        self.frames: list[bytes] = []
        root.title("Emotion Enhanced LLM")

        frame = ttk.Frame(root, padding=16)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.button = ttk.Button(frame, text="Record", command=self.toggle)
        self.button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.emotion = tk.StringVar()
        self.transcript = tk.StringVar()
        ttk.Label(frame, text="Emotion:").grid(row=1, column=0, sticky="nw")
        ttk.Label(frame, textvariable=self.emotion, wraplength=420).grid(row=1, column=1, sticky="nw")
        ttk.Label(frame, text="Transcript:").grid(row=2, column=0, sticky="nw")
        ttk.Label(frame, textvariable=self.transcript, wraplength=420).grid(row=2, column=1, sticky="nw")
        ttk.Label(frame, text="Reply:").grid(row=3, column=0, sticky="nw")
        self.reply = tk.Text(frame, height=8, width=56, wrap="word")
        self.reply.grid(row=3, column=1, sticky="nsew")

    def toggle(self):
        if self.recording:
            self.recording = False
            self.button.config(text="Record")
            return
        self.recording, self.frames = True, []
        self.button.config(text="Stop")
        threading.Thread(target=self._record, daemon=True).start()

    def _record(self):
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)
        while self.recording:
            self.frames.append(stream.read(CHUNK, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
        pa.terminate()
        audio = np.frombuffer(b"".join(self.frames), dtype=np.int16)
        self._run(audio)

    def _run(self, audio: np.ndarray):
        for turn in self.assistant.respond(audio, SAMPLE_RATE):
            self.root.after(0, self._show, turn)

    def _show(self, turn):
        if turn.emotion:
            e = turn.emotion
            self.emotion.set(f"{e.label} ({e.confidence:.0%})" if not e.reason else f"{e.label} ({e.reason})")
        if turn.transcript:
            self.transcript.set(turn.transcript.text)
        self.reply.delete("1.0", tk.END)
        self.reply.insert(tk.END, turn.reply)


if __name__ == "__main__":
    window = tk.Tk()
    DesktopApp(window, build_assistant())
    window.mainloop()
