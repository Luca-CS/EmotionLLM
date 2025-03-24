import tkinter as tk
from tkinter import ttk
import threading
import pyaudio
import wave
import speech_recognition as sr
from transformers import pipeline

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
    # The hidden token is provided by the dropdown selection.
    label_token = label_var.get()  
    transcript = transcript_var.get()
    # Construct the prompt with the transcript and the hidden token.
    prompt = transcript + ". \n " + label_token
    # Load the best available pipeline for instruction-following.
    generator = pipeline('text-generation', model='meta-llama/Meta-Llama-3-8B-Instruct')
    # Structured input for better results
    prompt = f"Question: {transcript}\nInstruction: {label_token}\nAnswer:"
    result = generator(prompt, max_length=50, num_return_sequences=1)
    # Extract and display the answer
    generated_text = result[0]['generated_text']
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, generated_text)

# Setting up the tkinter GUI.
root = tk.Tk()
root.title("Emotion Enhanced GPT-2")

# Variables to store transcript and dropdown selection.
transcript_var = tk.StringVar()
label_var = tk.StringVar(value="Emotion label")

# Button to record audio.
record_button = tk.Button(root, text="Record Audio", command=toggle_recording)
record_button.pack(pady=10)

# Display for transcript.
transcript_label = tk.Label(root, textvariable=transcript_var, wraplength=400)
transcript_label.pack(pady=5)

# Dropdown menu for the hidden token.
options = ["Answer in one word.", "Answer in one sentence.", "Explain why."]
dropdown = ttk.Combobox(root, values=options, textvariable=label_var, state="readonly")
dropdown.pack(pady=5)

# Button to generate text.
generate_button = tk.Button(root, text="Generate Answer", command=generate_text)
generate_button.pack(pady=10)

# Text widget to display the generated answer.
output_text = tk.Text(root, height=10, width=60)
output_text.pack(pady=5)

root.mainloop()



