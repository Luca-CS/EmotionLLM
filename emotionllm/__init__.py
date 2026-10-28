"""EmotionLLM: speech emotion recognition (CNN on MFCCs) feeding a tone-adapted chat reply.

The package holds the logic only (features, model, preprocessing, prompts, backends).
The interfaces live next to it: `app.py` (Gradio, used by the Hugging Face Space) and
`desktop_app.py` (the original Tkinter app).
"""

__version__ = "0.2.0"
