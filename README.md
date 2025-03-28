# EmotionLLM
Emotion Enhanced LLM Prompting (Audio RNN + Llama-8B API)

# Emotion Recognition and Response Generation System

## Abstract

This project presents an integrated system capable of recognizing emotions from audio inputs and generating contextually appropriate responses. The system leverages a combination of machine learning models, including an RNN trained on the IEMOCAP dataset for emotion recognition, a transcription pipeline for converting audio to text with proper punctuation, and the Llama 8B model for response generation. The application is designed to run locally, with certain models operating remotely via APIs, providing users with an interactive experience that combines speech emotion recognition with natural language processing capabilities.

## Getting Started

### Prerequisites

Before running the project, ensure that you have the following installed:

- **Python 3.11.1**: The project is compatible with Python version 3.11.1. You can download it from the official Python website.

- **pip**: Python's package installer. It is included by default in Python 3. Ensure it's up to date:

  ```bash
  python -m ensurepip --upgrade
  python -m pip install --upgrade pip
  ```


### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/emotion-recognition-response-system.git
   cd emotion-recognition-response-system
   ```


2. **Install Dependencies**:

   Install the required Python packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```


   *Note*: The `requirements.txt` file includes all necessary libraries for the project. Ensure that all dependencies are installed correctly to avoid runtime issues.

### Hugging Face Access Token

The project utilizes the Llama 8B model hosted on Hugging Face. To access this model, you need to create a Hugging Face access token.

#### What is Hugging Face?

Hugging Face is a platform that provides access to a wide range of pre-trained machine learning models, datasets, and tools for natural language processing and other AI applications. It facilitates collaboration and sharing within the AI community.

#### Creating a Hugging Face Access Token

1. **Sign Up or Log In**: Visit [Hugging Face](https://huggingface.co/) and create an account or log in to your existing account.

2. **Generate Access Token**:
   - Navigate to your account settings.
   - Click on the "Access Tokens" tab.
   - Click on the "New token" button.
   - Provide a name for your token and select the appropriate role.
   - Click "Generate" to create the token.

   *Note*: Keep this token secure, as it grants access to your Hugging Face account resources.

3. **Configure the Token in Your Environment**:

   Set the `HF_HOME` environment variable to the directory where Hugging Face will store the model files. Then, use the `huggingface-cli` to log in:

   ```bash
   export HF_HOME=~/.cache/huggingface
   huggingface-cli login
   ```


   When prompted, enter your access token.

## Running the Application

1. **Launch the Application**:

   Execute the main script to start the application:

   ```bash
   python main.py
   ```


2. **Using the Application**:

   - **Input**: Provide an audio file as input. The system will process this file to recognize the emotion conveyed.

   - **Processing**:
     - The audio is transcribed into text with appropriate punctuation.
     - The transcribed text is analyzed to detect the underlying emotion.
     - Based on the recognized emotion, the system generates a contextually relevant response.

   - **Output**: The generated response is displayed to the user.

## Models Involved

The system integrates several models to achieve its functionality:

1. **RNN Audio Model**:
   - **Purpose**: Emotion recognition from audio inputs.
   - **Training Data**: Trained on the IEMOCAP dataset, which contains acted emotional speech.
   - **Implementation**: Utilizes recurrent neural networks to analyze audio features and classify emotions.

2. **Transcription Pipeline**:
   - **Purpose**: Convert audio input into text with proper punctuation.
   - **Implementation**: Employs a speech-to-text model followed by a punctuation restoration model to ensure the transcribed text is accurately punctuated.

3. **Llama 8B Model**:
   - **Purpose**: Generate responses based on the transcribed and emotion-analyzed text.
   - **Implementation**: A large language model capable of producing coherent and contextually appropriate responses. Hosted on Hugging Face and accessed via their API.

## Model Loading and Execution

- **Local Execution**:
  - The RNN audio model and transcription pipeline run locally on the user's machine.
  - These models are loaded into memory during the application's initialization and process the audio input directly.

- **Remote Execution**:
  - The Llama 8B model is hosted remotely on Hugging Face's servers.
  - After local processing, the application sends the processed text to the Llama 8B model via the Hugging Face API to generate the response.

This hybrid approach leverages local computation for initial processing and utilizes powerful remote models for complex language generation tasks, balancing performance and resource utilization.

## Notes

- 