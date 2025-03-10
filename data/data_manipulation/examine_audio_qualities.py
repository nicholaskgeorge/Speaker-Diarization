import librosa
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_path)

audio ,sr = librosa.load("data/audio_files/for_embedding_training/paired_dataset/pair_1_0/p240_355.wav")

print(type(audio[0]))