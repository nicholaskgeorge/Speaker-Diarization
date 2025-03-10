import os
import random
from pydub import AudioSegment

def extract_random_segment(file_path, segment_duration_ms):
    audio = AudioSegment.from_wav(file_path)
    duration_ms = len(audio)
    if duration_ms <= segment_duration_ms:
        return audio
    start_ms = random.randint(0, duration_ms - segment_duration_ms)
    end_ms = start_ms + segment_duration_ms
    return audio[start_ms:end_ms]

def main(source_dir, dest_dir, num_files):
    if num_files % 2 != 0:
        raise ValueError("The number of files must be even.")

    os.makedirs(dest_dir, exist_ok=True)

    # Gather all .wav files from the source directory
    wav_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    num_speech_files = num_files // 2
    num_silent_files = num_files // 2

    # Randomly select files for speech segments
    selected_files = random.sample(wav_files, num_speech_files)

    # Determine the sample rate from the first file
    sample_audio = AudioSegment.from_wav(selected_files[0])
    sample_rate = sample_audio.frame_rate

    # Create and save silent audio clips
    silent_duration_ms = 1000  # 1 second
    silent_audio = AudioSegment.silent(duration=silent_duration_ms, frame_rate=sample_rate)
    for i in range(num_silent_files):
        silent_filename = os.path.join(dest_dir, f'silent_{i}_0.wav')
        silent_audio.export(silent_filename, format='wav')

    # Create and save speech audio clips
    for i, file_path in enumerate(selected_files):
        segment = extract_random_segment(file_path, silent_duration_ms)
        speech_filename = os.path.join(dest_dir, f'speech_{i}_1.wav')
        segment.export(speech_filename, format='wav')

if __name__ == "__main__":
    source_directory = "data/audio_files/silence_removed_data_set_librispeech"
    destination_directory = "data/audio_files/silence_accuracy_testing"
    number_of_files_to_generate = 300  # Example: 50 speech and 50 silent clips

    main(source_directory, destination_directory, number_of_files_to_generate)