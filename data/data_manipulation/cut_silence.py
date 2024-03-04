import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os

THRESHOLD = 0.054
BATCH_LENGTH = 1500


def cut_silence(audio):
    length_aduio = len(audio)
    new_audio = np.array([])
    for i in range(BATCH_LENGTH,length_aduio,BATCH_LENGTH):
        audio_slice = audio[i-BATCH_LENGTH:i]

        if(np.max(np.abs(audio_slice))>THRESHOLD):
            new_audio = np.append(new_audio, audio_slice)

    return new_audio

if __name__ == "__main__":
    audio, sr = librosa.load(r"general_testing\test_file.wav")
    new_audio = cut_silence(audio)
    sf.write(os.path.join(r"general_testing", "cleaned_audio.wav"), new_audio, sr)


