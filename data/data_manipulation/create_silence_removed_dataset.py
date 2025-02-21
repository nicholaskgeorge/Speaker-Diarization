

import os
import numpy as np
from scipy.io import wavfile
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_path)

from silence_detection.zero_cross_classifier import zcr_classify  # Ensure this function is available

def create_speaker_clips(dev_clean_path, output_path, clip_duration=2, chunk_duration=0.5):
    """
    Processes the dev-clean dataset to create 2-second speaker clips with non-silent audio segments.
    
    Parameters:
    - dev_clean_path: Path to the 'dev-clean' dataset.
    - output_path: Path to save the 'silence_removed_data_set'.
    - clip_duration: Duration of each clip in seconds (default is 2 seconds).
    - chunk_duration: Duration of each chunk to analyze in seconds (default is 0.5 seconds).
    """
    # Create the output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over each speaker directory in dev-clean
    for speaker_id, speaker_folder in enumerate(os.listdir(dev_clean_path), start=1):
        speaker_path = os.path.join(dev_clean_path, speaker_folder)
        if not os.path.isdir(speaker_path):
            continue

        # Create a subfolder for the speaker in the output directory
        speaker_output_path = os.path.join(output_path, f"speaker_{speaker_id}")
        os.makedirs(speaker_output_path, exist_ok=True)

        clip_count = 0  # Counter for naming clips

        # Iterate over each Level 2 folder (text reading) for the speaker
        for text_folder in os.listdir(speaker_path):
            text_path = os.path.join(speaker_path, text_folder)
            if not os.path.isdir(text_path):
                continue

            # Collect all .wav files in the current text folder
            wav_files = [f for f in os.listdir(text_path) if f.endswith('.wav')]

            # Process each .wav file
            for wav_file in wav_files:
                wav_file_path = os.path.join(text_path, wav_file)
                sample_rate, audio_data = wavfile.read(wav_file_path)

                # Calculate the number of samples per chunk and per clip
                chunk_samples = int(chunk_duration * sample_rate)
                clip_samples = int(clip_duration * sample_rate)

                # Process the audio data in chunks
                non_silent_chunks = []
                for start in range(0, len(audio_data) - chunk_samples + 1, chunk_samples):
                    chunk = audio_data[start:start + chunk_samples]
                    if zcr_classify(chunk) == 1:  # Non-silent chunk
                        non_silent_chunks.append(chunk)

                # Create clips from non-silent chunks
                while len(non_silent_chunks) >= 4:
                    clip = np.concatenate(non_silent_chunks[:4])
                    non_silent_chunks = non_silent_chunks[4:]

                    # Ensure the clip is exactly the desired duration
                    if len(clip) > clip_samples:
                        clip = clip[:clip_samples]
                    elif len(clip) < clip_samples:
                        # If the clip is too short, pad with silence
                        padding = np.zeros(clip_samples - len(clip))
                        clip = np.concatenate((clip, padding))

                    # Save the clip as a new .wav file
                    clip_count += 1
                    clip_filename = f"speaker_{speaker_id}_clip_{clip_count}.wav"
                    clip_output_path = os.path.join(speaker_output_path, clip_filename)
                    wavfile.write(clip_output_path, sample_rate, clip.astype(np.int16))

if __name__ == "__main__":
    dev_clean_path = "data/raw_dataset/LibriSpeech/dev-clean"  # Replace with the actual path
    output_path = "data/silence_removed_data_set"
    create_speaker_clips(dev_clean_path, output_path)