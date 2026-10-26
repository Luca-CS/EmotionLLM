# EmotionLLM

Speak into the microphone and get an answer from Llama 3 8B whose tone adapts to the emotion detected in your
voice. Built for a CentraleSupélec course project (ST7, March 2025).

## Pipeline

1. **Recording**: a small Tkinter app records from the microphone (PyAudio).
2. **Transcription**: Google Web Speech API through `SpeechRecognition` (this step needs an internet
   connection), then punctuation restored with `deepmultilingualpunctuation`.
3. **Emotion recognition**: a CNN on MFCCs (13 coefficients × 500 frames) classifies the recording into six
   classes: angry, excited, frustrated, happy, neutral, sad. It was trained on
   [IEMOCAP](https://sail.usc.edu/iemocap/) (acted speech) and runs on the CPU to leave the GPU to the LLM.
4. **Response**: `meta-llama/Meta-Llama-3-8B-Instruct`, run locally with `transformers` (bfloat16,
   `device_map="auto"`). The detected emotion becomes a tone instruction in the system prompt: angry → calm,
   frustrated → reassuring, sad → empathetic, excited → enthusiastic, happy → positive.

## Files

| File | Role |
|---|---|
| `main.py` | Tkinter app: recording, transcription, emotion, generation |
| `model.py` | Feature extraction (MFCC, pitch, energy) and the CNN |
| `emotion_classification_model_weight_3.pth` | Trained CNN weights (5 MB) |
| `rapport_LF.ipynb` | Training and evaluation of the CNN on Google Colab (confusion matrices, metrics) |

## Running it

Requirements: Python 3.11, a microphone, and a GPU with enough memory for Llama 3 8B (or enough RAM for
`device_map="auto"` to offload part of it).

```bash
pip install -r requirements.txt
huggingface-cli login     # or set HF_TOKEN; Llama 3 is a gated model, request access on its Hugging Face page first
python main.py
```

Click **Record Audio**, speak, click again to stop, then **Generate**.

## Limitations

- The CNN was trained for 5 epochs on a 5,120-utterance subset of IEMOCAP. Acted emotions transfer imperfectly to
  spontaneous speech, so treat the detected label as a hint.
- IEMOCAP is not included: it is distributed under a license agreement with USC.
- Transcription is not local (Google Web Speech API).

## License

[MIT](LICENSE)
