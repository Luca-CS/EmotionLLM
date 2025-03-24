import tkinter as tk
from tkinter import ttk
import threading
import pyaudio
import wave
import speech_recognition as sr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Placeholder global definitions to appease the IDE.
record_button = None
transcript_var = None
label_var = None
output_text = None

# Load Meta-Llama-3-8B-Instruct model and tokenizer ONCE at startup
access_token="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"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=access_token
)

# Global variables for recording
is_recording = False
frames = []
recording_thread = None
p = pyaudio.PyAudio()

def record_loop():
    global is_recording, frames
    # Open the audio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
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
    
    # Save the recorded audio as a WAV file in the app's directory
    filename = "recorded_audio.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Audio recorded and saved as", filename)
    
    # Generate transcript from the recorded file using speech_recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio_data)
    except Exception as e:
        transcript = "Audio could not be understood"
    transcript_var.set(transcript)

def toggle_recording():
    global is_recording, frames, recording_thread
    if not is_recording:
        # Start recording
        is_recording = True
        frames = []  # Reset frames
        record_button.config(text="Press to stop recording", bg="red")
        recording_thread = threading.Thread(target=record_loop)
        recording_thread.start()
    else:
        # Stop recording
        is_recording = False
        record_button.config(text="Record Audio", bg="SystemButtonFace")
        # Schedule a check for thread termination without blocking the GUI.
        root.after(100, check_thread)

def check_thread():
    if recording_thread.is_alive():
        root.after(100, check_thread)
    else:
        print("Recording thread finished.")

def generate_text():
    """Generate an answer using Meta-Llama-3-8B-Instruct."""
    transcript = transcript_var.get()
    label_token = label_var.get()

    # Format the input
    messages = [
        {"role": "system", "content": label_token},
        {"role": "user", "content": transcript},
    ]

    # Tokenize input
    # Use the chat template function to build input_ids.
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Set terminators as in the readme.
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=50,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Extract only the generated response.
    response = outputs[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(response, skip_special_tokens=True)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, generated_text)

# Setting up the tkinter GUI.
root = tk.T
