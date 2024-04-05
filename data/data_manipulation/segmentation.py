import librosa
import soundfile as sf
from cut_silence import cut_silence
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(r"file_movement_scripts"))
from delete_files_in_folder import delete_files_in_folder
from get_files_in_folder import get_files_in_folder
from math import ceil
from segmentation_parameters import SEGMENTATION_LENGTH

RAW_DATA_PATH = r"data\training_data\silence_removed_training_set"
DESTINATION = r"data\training_data\segmentation\audio_files"
#length in seconds for each segmentation


delete_files_in_folder(DESTINATION)
files = get_files_in_folder(RAW_DATA_PATH)

for file in files:
    file_name = file[:file.find(".")]
    audio_path= os.path.join(RAW_DATA_PATH, file)
    audio, sr = librosa.load(audio_path)
    segment_time_increment = int(SEGMENTATION_LENGTH*sr)
    audio_len = len(audio)
    num_iterations = int(ceil(audio_len/segment_time_increment))
    segment = 0
    segment_start = 0
    segment_end = segment_time_increment
    for segment in range(num_iterations):
        if (segment_start < len(audio)):
            segment_file_name = f"{file_name}_segment_{segment}.wav"
            segment_audio = audio[segment_start:segment_end]
            if len(segment_audio)<segment_time_increment:
                zeros_missing = [0]*(segment_time_increment-len(segment_audio))
                segment_audio = np.concatenate((segment_audio, zeros_missing))
            sf.write(os.path.join(DESTINATION, segment_file_name), segment_audio, sr)
            segment_start += segment_time_increment
            segment_end += segment_time_increment

    





