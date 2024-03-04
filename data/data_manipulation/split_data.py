import librosa
from get_file_names import get_file_names
from random import shuffle
import os
import soundfile as sf

speaker_data_path = r"C:\Users\nicok\Speaker-Diarization\Training_Data\Augmented Data"
testing_set_path = r"C:\Users\nicok\Speaker-Diarization\Training_Data\test_set"
training_set_path = r"C:\Users\nicok\Speaker-Diarization\Training_Data\training_set"

files = get_file_names(speaker_data_path)
shuffle(files)

test_set_frac = 0.1

num_files = len(files)
middle = int(num_files*(1-test_set_frac))
train_set = files[:middle]
test_set = files[middle:]

for file in train_set:
    audio, sr = librosa.load(os.path.join(speaker_data_path, file))
    sf.write(os.path.join(training_set_path, file), audio, sr)

for file in test_set:
    audio, sr = librosa.load(os.path.join(speaker_data_path, file))
    sf.write(os.path.join(testing_set_path, file), audio, sr)
