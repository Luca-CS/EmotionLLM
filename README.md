# EmotionLLM

## Emotion Recognition and Response Generation System

### Abstract

Welcome to **EmotionLLM**, an innovative system designed to recognize emotions from audio inputs and generate contextually appropriate responses. This project seamlessly integrates advanced machine learning models, including a Convolutional Neural Network (CNN) for emotion recognition, a transcription pipeline for converting audio to text with proper punctuation, and a locally cached Llama 8B model for response generation. The application provides an interactive experience that combines speech emotion recognition with natural language processing capabilities, all running efficiently on your local machine.

### Getting Started

#### Prerequisites

Before diving into the project, ensure you have the following installed:

- **Python 3.11.1**: The project is compatible with Python version 3.11.1. You can download it from the [official Python website](https://www.python.org/).

- **pip**: Python's package installer. It is included by default in Python 3. Ensure it's up to date:

  ```bash
  python -m pip install --upgrade pip
  ```

#### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/emotion-recognition-response-system.git
   cd emotion-recognition-response-system
   ```

2. **Install Dependencies**:

   Install the required Python packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   *Note*: The `requirements.txt` file includes all necessary libraries for the project. Ensure that all dependencies are installed correctly to avoid runtime issues.

### Hugging Face Access Token

The project utilizes the Llama 8B model, which is cached locally between your desktop's VRAM and RAM for efficient access. To access this model, you need to create a Hugging Face access token.

#### What is Hugging Face?

Hugging Face is a platform that provides access to a wide range of pre-trained machine learning models, datasets, and tools for natural language processing and other AI applications. It facilitates collaboration and sharing within the AI community.

#### Creating a Hugging Face Access Token

1. **Sign Up or Log In**: Visit [Hugging Face](https://huggingface.co/) and create an account or log in to your existing account.

2. **Generate Access Token**:
   - Navigate to your account settings.
   - Click on the "Access Tokens" tab.
   - Click on the "New token" button.
   - Provide a name for your token and select the appropriate role.
   - Click "Generate" to create the token.

   *Note*: Keep this token secure, as it grants access to your Hugging Face account resources.

3. **Configure the Token in Your Environment**:

   Set the `HF_HOME` environment variable to the directory where Hugging Face will store the model files. Then, use the `huggingface-cli` to log in:

   ```bash
   export HF_HOME=~/.cache/huggingface
   huggingface-cli login
   ```

   When prompted, enter your access token.

### Running the Application

1. **Launch the Application**:

   Execute the main script to start the application:

   ```bash
   python main.py
   ```

2. **Using the Application**:

   - **Input**: Click on the "Record Audio" button, then click again on "Stop recording" when done. The system will process this file to recognize the emotion conveyed.

   - **Processing**:
     - The audio is transcribed into text.
     - In the meantime, emotion is extracted from the audio file.
     - Based on the recognized emotion, the system generates a contextually relevant response.

   - **Output**: The generated response is displayed to the user.

### Models Involved

The system integrates several models to achieve its functionality:

1. **CNN Audio Model**:
   - **Purpose**: Emotion recognition from audio inputs.
   - **Training Data**: Trained on the IEMOCAP dataset, which contains acted emotional speech.
   - **Implementation**: Utilizes convolutional neural networks to analyze audio features and classify emotions. Loaded on the CPU to save space for the Llama 8B model.

2. **Transcription Pipeline**:
   - **Purpose**: Convert audio input into text with proper punctuation.
   - **Implementation**: Employs a speech-to-text model followed by a punctuation restoration model to ensure the transcribed text is accurately punctuated.

3. **Llama 8B Model**:
   - **Purpose**: Generate responses based on the transcribed and emotion-analyzed text.
   - **Implementation**: A large language model capable of producing coherent and contextually appropriate responses. Cached locally for efficient access and execution.

### Model Loading and Execution

- **Local Execution**:
  - The CNN audio model and transcription pipeline run locally on the user's machine. The CNN model is loaded on the CPU to save space for the Llama 8B model.
  - These models are loaded into memory during the application's initialization and process the audio input directly.

- **Local Caching**:
  - The Llama 8B model is cached locally between the desktop's VRAM and RAM.
  - After local processing, the application leverages the cached Llama 8B model to generate responses, ensuring fast and efficient performance.

This hybrid approach leverages local computation for initial processing and utilizes powerful locally cached models for complex language generation tasks, balancing performance and resource utilization.

### Notes

- Ensure your system has sufficient VRAM and RAM to cache the Llama 8B model efficiently.
- The application is designed to run on a local machine, providing a seamless and responsive user experience.

---
