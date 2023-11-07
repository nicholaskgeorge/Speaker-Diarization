import os
import numpy as np
from secrets import randbelow
import librosa
import soundfile as sf

NUM_DATA_POINTS = 400
MAX_LEN_DATA_SEC= 12
SAMEPLE_RATE = 44100
MAX_NUM_SPEAKERS = 2

speaker_data = r"C:\Users\nicok\Speaker-Diarization\Speaker_seperated_data\data"
destination = r"C:\Users\nicok\Speaker-Diarization\Training_Data\Augmented Data"

def get_file_names(file_path):
    files = []
    for file_name in os.listdir(file_path):
        path = os.path.join(file_path, file_name)
        if os.path.isfile(path) or os.path.isdir(path):
            files.append(file_name)
    return files

def make_12_secs(audio, sr):
    new_aduio = 0
    # Calculate the duration of the spliced audio in seconds
    duration = len(audio) / sr

    new_audio = "none"
    # If the duration is greater than 12 seconds, shorten the audio to 12 seconds
    if duration > MAX_LEN_DATA_SEC:
        new_audio = audio[:int(MAX_LEN_DATA_SEC * sr)]
    elif duration < MAX_LEN_DATA_SEC:

        # Calculate the number of samples required to achieve a 12-second duration
        target_duration = MAX_LEN_DATA_SEC  # Target duration in seconds
        target_samples = int(target_duration * sr)

        padding_samples = target_samples - len(audio)

        new_audio = np.pad(audio, (0, padding_samples), mode='constant')

    return new_audio
    

speakers = get_file_names(speaker_data)

total_num_speakers = max([int(i[1:]) for i in speakers])
num_speakers = 0
s1 = 0
s2 = 0
s1_file = 0
s2_file = 0

audio_signal = 0

for i in range(NUM_DATA_POINTS):
    #choose numbers of speakers
    num_speakers = randbelow(MAX_NUM_SPEAKERS+1)

    s1 = randbelow(total_num_speakers)
    s2 = randbelow(total_num_speakers)
    #choose speakers
    while s1 == s2:
        s1 = randbelow(total_num_speakers)
        s2 = randbelow(total_num_speakers)
    
    s1+=1
    s2+=1

    data_point_name = f"point{i}_{num_speakers}_speakers.wav"

    #get random file from the speaker
    s1_path = os.path.join(speaker_data, f"s{s1}")
    s1_files = get_file_names(s1_path)
    choose_file = randbelow(len(s1_files))
    audio_s1, sr_s1 = librosa.load(os.path.join(s1_path, s1_files[choose_file]))

    s2_path = os.path.join(speaker_data, f"s{s2}")
    s2_files = get_file_names(s2_path)
    choose_file = randbelow(len(s2_files))
    audio_s2, sr_s2 = librosa.load(os.path.join(s2_path, s2_files[choose_file]))

    if sr_s1 != sr_s2:
        print("sampling issue")
    
    audio_signal = 0
    #concatinate clips. Make it 12 seconds long
    # if num_speakers == 0:
    #     audio_signal = np.zeros(int(SAMEPLE_RATE*MAX_LEN_DATA_SEC/2))
    if num_speakers in [1,0]:
        audio_signal = make_12_secs(audio_s1, sr_s1)
    else:
        audio_signal = np.concatenate((audio_s1, audio_s2))
        audio_signal = make_12_secs(audio_signal,sr_s2)

    print(f"length of file {data_point_name} is: {(audio_signal.shape)}")

    sf.write(os.path.join(destination, data_point_name), audio_signal, sr_s1)
