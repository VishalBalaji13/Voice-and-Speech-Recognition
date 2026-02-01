import os
import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import jiwer
import speech_recognition as sr



# Specify the path to the local dataset folder
data_dir = os.path.join(
    "/Users", "vishal", "Downloads", "final_project_AI", "cv-corpus-21.0-delta-2025-03-14"
)
dataset = load_dataset(
    "mozilla-foundation/common_voice_11_0", 
    "en", 
    data_dir=data_dir,
    trust_remote_code=True
)

# Sample data: Load a specific example from the dataset
audio_example = dataset['train'][0]  # Take the first example from the training split

# Extract audio and transcript
audio_path = audio_example['path']
transcript = audio_example['sentence']

# Load the audio file using librosa
audio, sample_rate = librosa.load(audio_path, sr=16000)
sf.write('temp_audio.wav', audio, sample_rate)

# Load the pre-trained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", sampling_rate=16000)

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load the audio file into the correct format
input_audio, _ = librosa.load('temp_audio.wav', sr=16000)

# Tokenize the audio
input_values = processor(input_audio, return_tensors="pt").input_values

# Perform transcription
with torch.no_grad():
    logits = model(input_values).logits

# Convert logits to predicted IDs
predicted_ids = torch.argmax(logits, dim=-1)

# Decode the IDs to text
transcription = processor.decode(predicted_ids[0])
print("Transcription:", transcription)

# Compare transcription with the ground truth
wer = jiwer.wer(transcript, transcription)
print("Word Error Rate (WER):", wer)

# Microphone-based transcription
recognizer = sr.Recognizer()

try:
    # Use the microphone to capture audio
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)

    # Save the recorded audio to a temporary file
    with open("temp_mic_audio.wav", "wb") as f:
        f.write(audio.get_wav_data())

    # Load the audio file
    audio_input, _ = librosa.load("temp_mic_audio.wav", sr=16000)

    # Tokenize the recorded audio
    input_values = processor(audio_input, return_tensors="pt").input_values

    # Perform transcription
    with torch.no_grad():
        logits = model(input_values).logits

    # Convert logits to predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the IDs to text
    transcription = processor.decode(predicted_ids[0])
    print("Microphone Transcription:", transcription)

except Exception as e:
    print("Error capturing audio from microphone:", e)
