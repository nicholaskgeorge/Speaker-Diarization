import os
import sys
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os


# Define source and destination directories
source_dir = "data/audio_files/raw_dataset/VCTK/wav48"
dest_dir = "data/audio_files/silence_removed_vctk"

# Create the destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Parameters
target_duration_ms = 2000  # Target duration for concatenated clips
min_silence_len = 500  # Minimum length of silence to be considered as a split point (in ms)
silence_thresh = -40  # Silence threshold in dBFS

# Iterate over each speaker's folder
for speaker_folder in os.listdir(source_dir):
    speaker_path = os.path.join(source_dir, speaker_folder)
    if os.path.isdir(speaker_path):
        # Create corresponding folder in the destination directory
        dest_speaker_path = os.path.join(dest_dir, speaker_folder)
        os.makedirs(dest_speaker_path, exist_ok=True)
        
        # Initialize variables
        concatenated_audio = AudioSegment.silent(duration=0)
        clip_count = 0
        
        # Iterate over each audio file in the speaker's folder
        for audio_file in os.listdir(speaker_path):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(speaker_path, audio_file)
                audio = AudioSegment.from_wav(audio_path)
                
                # Detect non-silent chunks in the audio file
                nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
                
                # Extract and concatenate non-silent chunks
                for start_i, end_i in nonsilent_ranges:
                    chunk = audio[start_i:end_i]
                    concatenated_audio += chunk
                    
                    # If the concatenated audio reaches the target duration, save it
                    if len(concatenated_audio) >= target_duration_ms:
                        output_path = os.path.join(dest_speaker_path, f"{clip_count}.wav")
                        concatenated_audio[:target_duration_ms].export(output_path, format="wav")
                        clip_count += 1
                        concatenated_audio = concatenated_audio[target_duration_ms:]
        
        # Save any remaining audio that didn't reach the full target duration
        if len(concatenated_audio) > 0:
            output_path = os.path.join(dest_speaker_path, f"{clip_count}.wav")
            concatenated_audio.export(output_path, format="wav")