import librosa
import soundfile as sf
from cut_silence import cut_silence
import os
from get_files_in_folder import get_files_in_folder
from delete_files_in_folder import delete_files_in_folder

RAW_DATA_PATH = r"data\training_data\training_set"
DESTINATION = r"data\training_data\silence_removed_training_set"

files = get_files_in_folder(RAW_DATA_PATH)

for file in files:
    audio_path= os.path.join(RAW_DATA_PATH, file)
    audio, sr = librosa.load(audio_path)
    clean_aduio = cut_silence(audio)
    clean_file_name = f"{file}_cleaned.wav"
    sf.write(os.path.join(DESTINATION, clean_file_name), clean_aduio, sr)
