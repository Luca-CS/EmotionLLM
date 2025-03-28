import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Use local weight file (ensure it’s in the same directory as main.py)
weight_save_path = "emotion_classification_model_weight_3.pth"

def extract_features(waveform, sample_rate, n_mfcc=13):
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=waveform.numpy(), sr=sample_rate, n_mfcc=n_mfcc)
    
    # Compute pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=waveform.numpy(), sr=sample_rate)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

    # Compute energy
    energy = np.sum(waveform.numpy() ** 2) / len(waveform)
    
    return mfccs, pitch, energy

def preprocess_data(data_set):
    """
    data_set: list of tuples (waveform, sample_rate)
    """
    X = []
    for waveform, sample_rate in data_set:
        mfccs, pitch, energy = extract_features(waveform, sample_rate)
        # Convert MFCCs to tensor; expected shape: (n_mfcc, time)
        mfcc_tensor = torch.tensor(mfccs, dtype=torch.float32)
        X.append(mfcc_tensor)
    
    # Define maximum time length (number of frames)
    max_length = 500
    print(f"Maximum sequence length: {max_length}")

    processed_sequences = []
    for seq in X:
        # seq.shape is (n_mfcc, time); we pad or truncate along the time dimension (dim=1)
        pad_size = max_length - seq.shape[1]
        if pad_size > 0:
            padding = torch.zeros((seq.shape[0], pad_size), dtype=torch.float32)
            seq_padded = torch.cat((seq, padding), dim=1)
        else:
            seq_padded = seq[:, :max_length]
        processed_sequences.append(seq_padded)
    
    # Stack into a single tensor of shape (batch, n_mfcc, time)
    X = torch.stack(processed_sequences)
    return X

# Define the list of emotion labels
labels = ['Angry', 'Excited', 'Frustrated', 'Happy', 'Neutral', 'Sad']

class EmotionClassificationCNN(nn.Module):
    def __init__(self):
        super(EmotionClassificationCNN, self).__init__()
        # Convolutional layers: input has 1 channel (MFCCs treated as an image)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.3)
        # Adjust fc1 input size based on expected shape after conv+pooling.
        # Here we assume the processed MFCC image is resized to (1, 62) after pooling.
        self.fc1 = nn.Linear(64 * 1 * 62, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 6)  # 6 emotion classes

    def forward(self, x):
        # x should have shape (batch, 1, n_mfcc, time)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def encode_emotion(emotion):
    # Map detected emotion to a token string
    emotion_to_token = {
        'Angry': '<|calm|>',
        'Excited': '<|enthusiastic|>',
        'Frustrated': '<|reassuring|>',
        'Happy': '<|positive|>',
        'Neutral': '<||>',
        'Sad': '<|empathetic|>'
    }
    return emotion_to_token.get(emotion, '<||>')

def audio_to_emotion(input_data):
    # If input_data is a string, assume it's a filename and load the audio
    if isinstance(input_data, str):
        waveform, sr_value = librosa.load(input_data, sr=16000)
        # Convert waveform to a tensor and wrap in a list of tuples
        input_data = [(torch.tensor(waveform), sr_value)]
    
    model_instance = EmotionClassificationCNN()
    state = torch.load(weight_save_path, map_location=torch.device('cpu'))
    model_instance.load_state_dict(state, strict=False)
    model_instance.eval()
    
    X = preprocess_data(input_data)
    # Expecting X to have shape (batch, n_mfcc, time); add channel dimension
    X = X.unsqueeze(1)
    
    with torch.no_grad():
        logits = model_instance(X)
    predicted_index = torch.argmax(logits, dim=1)[0].item()
    emotion = labels[predicted_index]
    return emotion

def encodage_label(emotion):
    # Convert the detected emotion into the label token for Llama-8B
    return encode_emotion(emotion)

# For testing purposes (remove or comment out in production):
if __name__ == "__main__":
    # Example: load a test set (assuming test_set.pkl exists)
    import pickle
    with open("test_set.pkl", "rb") as f:
        test_set = pickle.load(f)
    # Assume test_set is a list of tuples (waveform, sample_rate)
    input_data = [test_set[27][:2]]
    detected_emotion = audio_to_emotion(input_data)
    token = encodage_label(detected_emotion)
    print("Detected Emotion:", detected_emotion)
    print("Encoded Label:", token)
