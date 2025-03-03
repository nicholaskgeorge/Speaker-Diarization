import os
import shutil
import random

# Configuration
DATASET_PATH = 'data/audio_files/silence_removed_vctk'  # Path to the dataset
OUTPUT_PATH = 'data/audio_files/for_embedding_training/paired_dataset'           # Path to save the new dataset
DIFF_SPEAKER_RATIO = 0.7                                 # Ratio of pairs from different speakers
MAX_PAIRS = 3000                                          # Maximum number of pairs to generate

def create_paired_dataset(dataset_path, output_path, diff_speaker_ratio=0.5, max_pairs=1000):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Get list of speakers
    speakers = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    pair_index = 0

    # Create pairs
    while pair_index < max_pairs:
        if random.random() < diff_speaker_ratio:
            # Pair from different speakers
            speaker1, speaker2 = random.sample(speakers, 2)
            label = 0
        else:
            # Pair from the same speaker
            speaker1 = random.choice(speakers)
            speaker2 = speaker1
            label = 1

        speaker1_path = os.path.join(dataset_path, speaker1)
        speaker2_path = os.path.join(dataset_path, speaker2)
        audio_files1 = [f for f in os.listdir(speaker1_path) if f.endswith('.wav')]
        audio_files2 = [f for f in os.listdir(speaker2_path) if f.endswith('.wav')]

        if not audio_files1 or not audio_files2:
            continue

        audio_file1 = random.choice(audio_files1)
        audio_file2 = random.choice(audio_files2)

        # Ensure the selected files are not the same
        if speaker1 == speaker2 and audio_file1 == audio_file2:
            continue

        pair_index += 1
        pair_folder = os.path.join(output_path, f'pair_{pair_index}_{label}')
        os.makedirs(pair_folder, exist_ok=True)

        # Copy and rename files to include speaker identifiers
        new_name1 = f'{speaker1}_{audio_file1}'
        new_name2 = f'{speaker2}_{audio_file2}'
        shutil.copy(os.path.join(speaker1_path, audio_file1), os.path.join(pair_folder, new_name1))
        shutil.copy(os.path.join(speaker2_path, audio_file2), os.path.join(pair_folder, new_name2))

if __name__ == "__main__":
    create_paired_dataset(DATASET_PATH, OUTPUT_PATH, DIFF_SPEAKER_RATIO, MAX_PAIRS)