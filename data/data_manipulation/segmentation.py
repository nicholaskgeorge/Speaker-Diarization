import librosa
import soundfile as sf
from cut_silence import cut_silence
import os
from get_files_in_folder import get_files_in_folder
from delete_files_in_folder import delete_files_in_folder

RAW_DATA_PATH = r"data\training_data\training_set"
DESTINATION = r"data\training_data\segmentation\audio_files"
#length in seconds for each segmentation
SEGMENTATION_LENGTH = 0.25
DATA_POINT_LENGTH = 4
NUM_ITERATIONS = int(DATA_POINT_LENGTH/SEGMENTATION_LENGTH)

delete_files_in_folder(DESTINATION)
files = get_files_in_folder(RAW_DATA_PATH)

for file in files:
    file_name = file[:file.find(".")]
    audio_path= os.path.join(RAW_DATA_PATH, file)
    audio, sr = librosa.load(audio_path)
    clean_aduio = cut_silence(audio)
    segment = 0
    segment_time_increment = int(sr*SEGMENTATION_LENGTH)
    segment_start = 0
    segment_end = segment_time_increment
    for segment in range(NUM_ITERATIONS):
        if (segment_start < len(clean_aduio)):
            segment_file_name = f"{file_name}_segment_{segment}.wav"
            segment_audio = clean_aduio[segment_start:segment_end]
            sf.write(os.path.join(DESTINATION, segment_file_name), segment_audio, sr)
            segment_start += segment_time_increment
            segment_end += segment_time_increment

    





