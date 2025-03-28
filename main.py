import tkinter as tk
from tkinter import ttk
import threading
import pyaudio
import wave
import speech_recognition as sr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from deepmultilingualpunctuation import PunctuationModel

# Import the emotion functions from model.py
from model import audio_to_emotion, encodage_label

# Global GUI variables
record_button = None
transcript_var = None
label_var = None
output_text = None
generate_button = None

# Globals for the Llama model (initialized as None)
tokenizer = None
model = None

# Global punctuation restoration pipeline
punctuation_pipeline = None

# Global variable for storing the encoded emotion token for generation
encoded_label_token = None

# Global variables for recording
is_recording = False
frames = []
recording_thread = None
p = pyaudio.PyAudio()


def load_model():
    global tokenizer, model, punctuation_pipeline
    access_token = "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"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print("Loading model...")
    tokenizer_local = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=access_token)
    model_local = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=access_token
    )
    tokenizer = tokenizer_local
    model = model_local
    print("Loading punctuation restoration model...")
    try:
        punctuation_pipeline = PunctuationModel()
    except Exception as e:
        print("Punctuation pipeline load failed:", e)
        punctuation_pipeline = None
    print("Models loaded!")
    generate_button.config(state="normal")


def record_loop():
    global is_recording, frames, encoded_label_token
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    while is_recording:
        try:
            data = stream.read(1024)
        except Exception as e:
            print("Error reading stream:", e)
            break
        frames.append(data)
    stream.stop_stream()
    stream.close()
    
    filename = "recorded_audio.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Audio recorded and saved as", filename)
    
    # Generate transcript from the recorded file
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio_data)
    except Exception as e:
        transcript = "Audio could not be understood"
    
    # Restore punctuation using the punctuation pipeline
    if punctuation_pipeline is not None:
        punctuated = punctuation_pipeline.restore_punctuation(transcript)
        transcript = punctuated
    
    # Update the transcript in the GUI (via the main thread)
    root.after(0, lambda: transcript_var.set(transcript))
    
    # Update GUI: show progress while extracting emotion
    root.after(0, lambda: label_var.set("Extracting emotion label from audio"))
    
    # Process the audio file using the imported functions:
    emotion = audio_to_emotion(filename)  # e.g., "Angry"
    encoded_token = encodage_label(emotion)  # e.g., "<|calm|>" for generation
    
    # Update the GUI with the raw emotion label (e.g., "Angry")
    root.after(0, lambda: label_var.set(emotion))
    # Store the encoded token for generation
    encoded_label_token = encoded_token

def toggle_recording():
    global is_recording, frames, recording_thread
    if not is_recording:
        is_recording = True
        frames = []
        record_button.config(text="Press to stop recording", bg="#e74c3c")
        recording_thread = threading.Thread(target=record_loop)
        recording_thread.start()
    else:
        is_recording = False
        record_button.config(text="Record Audio", bg="#3498db")
        root.after(100, check_thread)

def check_thread():
    if recording_thread.is_alive():
        root.after(100, check_thread)
    else:
        print("Recording thread finished.")

def generate_text():
    generate_button.config(state="disabled")
    threading.Thread(target=generate_text_worker, daemon=True).start()

def generate_text_worker():
    transcript = transcript_var.get()
    token_to_use = encoded_label_token if encoded_label_token is not None else label_var.get()
    messages = [
        {"role": "system", "content": token_to_use},
        {"role": "user", "content": transcript},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    if isinstance(inputs, dict):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    else:
        input_ids = inputs
        attention_mask = torch.ones_like(input_ids)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,  # Longer response
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(response, skip_special_tokens=True)
    root.after(0, update_output, generated_text)

def update_output(text):
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, text)
    generate_button.config(state="normal")

# Initialize the tkinter GUI
root = tk.Tk()
root.title("Emotion Enhanced LLM")

# Apply dark theme and configure resizing
root.configure(bg="#2C2F33")
style = ttk.Style(root)
style.theme_use("clam")
style.configure("TFrame", background="#2C2F33")
style.configure("TLabel", background="#2C2F33", foreground="white", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12))
style.configure("TEntry", font=("Helvetica", 12))

# Allow window resizing
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

transcript_var = tk.StringVar()
label_var = tk.StringVar(value="")  # Initially empty; will be updated after emotion extraction

frame = ttk.Frame(root, padding=20)
frame.grid(row=0, column=0, sticky="nsew")
frame.grid_rowconfigure(2, weight=1)
frame.grid_columnconfigure(1, weight=1)

record_button = tk.Button(frame, text="Record Audio", command=toggle_recording,
                          font=("Helvetica", 12), bg="#3498db", fg="white", bd=0, padx=10, pady=5)
record_button.grid(row=0, column=0, pady=10, padx=10, sticky="ew")

generate_button = tk.Button(frame, text="Generate Response", command=generate_text, state="disabled",
                            font=("Helvetica", 12), bg="#2ecc71", fg="white", bd=0, padx=10, pady=5)
generate_button.grid(row=0, column=1, pady=10, padx=10, sticky="ew")

label_entry = tk.Entry(frame, textvariable=label_var, font=("Helvetica", 12), width=50, bd=2, relief="groove")
label_entry.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

transcript_label = tk.Label(frame, text="Transcript:", font=("Helvetica", 12), bg="#2C2F33", fg="white")
transcript_label.grid(row=2, column=0, sticky="nw", padx=10, pady=(10, 0))

transcript_display = tk.Label(frame, textvariable=transcript_var, font=("Helvetica", 12),
                              wraplength=400, justify="left", bg="#2C2F33", fg="white")
transcript_display.grid(row=2, column=1, sticky="nw", padx=10, pady=(10, 0))

output_label = tk.Label(frame, text="Generated Response:", font=("Helvetica", 12), bg="#2C2F33", fg="white")
output_label.grid(row=3, column=0, sticky="nw", padx=10, pady=(10, 0))

output_text = tk.Text(frame, height=8, width=50, font=("Helvetica", 12), bd=2, relief="groove", bg="#23272A", fg="white")
output_text.grid(row=3, column=1, pady=10, padx=10, sticky="nsew")

# Start model loading in a background thread
threading.Thread(target=load_model, daemon=True).start()

print("Launching tkinter app...")
root.mainloop()


