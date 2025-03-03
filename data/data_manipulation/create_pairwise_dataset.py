import os
import numpy as np
import librosa

# Configuration
PAIRED_DATASET_PATH = 'data/audio_files/for_embedding_training/paired_dataset'  # Path to the existing paired dataset
OUTPUT_FILE_PATH = 'data/audio_files/for_embedding_training/pairwise_numpy/pair_dataset'   # Path to save the new MFCC dataset
SAMPLE_RATE = 44100                                     # Sample rate of the audio files
DURATION = 1.0                                          # Duration to analyze (in seconds)
N_MFCC = 48                                             # Number of MFCC coefficients
HOP_LENGTH = int(SAMPLE_RATE * DURATION / 10)           # Hop length to get 10 frames per second

def extract_mfcc(file_path, sample_rate, duration, n_mfcc, hop_length):
    """Extract MFCC features from the first `duration` seconds of the audio file."""
    y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc.T[:10, :]  # Ensure the shape is (10, n_mfcc)

def process_paired_dataset(paired_dataset_path):
    mfcc_1_list = []
    mfcc_2_list = []
    labels = []
    pair_folders = [d for d in os.listdir(paired_dataset_path) if os.path.isdir(os.path.join(paired_dataset_path, d))]

    for pair_folder in pair_folders:
        pair_path = os.path.join(paired_dataset_path, pair_folder)
        audio_files = [f for f in os.listdir(pair_path) if f.endswith('.wav')]

        if len(audio_files) != 2:
            continue  # Skip if the pair folder doesn't contain exactly two audio files

        mfcc_features = []
        for audio_file in audio_files:
            file_path = os.path.join(pair_path, audio_file)
            mfcc = extract_mfcc(file_path, SAMPLE_RATE, DURATION, N_MFCC, HOP_LENGTH)
            if mfcc.shape != (10, N_MFCC):
                continue  # Skip if MFCC extraction doesn't result in the expected shape
            mfcc_features.append(mfcc)

        if len(mfcc_features) != 2:
            continue  # Skip if we don't have two valid MFCC matrices

        # Determine label from folder name (assuming format: pair_{index}_{label})
        label = int(pair_folder.split('_')[-1])

        mfcc_1_list.append(mfcc_features[0])
        mfcc_2_list.append(mfcc_features[1])
        labels.append(label)
    
    mfcc_1_list = np.array(mfcc_1_list)
    mfcc_2_list = np.array(mfcc_2_list)

    # normalize across each frame
    mfcc_1_list_mean = np.mean(mfcc_1_list, axis=-1, keepdims=True)
    mfcc_2_list_mean = np.mean(mfcc_2_list, axis=-1, keepdims=True)
    mfcc_1_list_std = np.std(mfcc_1_list, axis=-1, keepdims=True)
    mfcc_2_list_std = np.std(mfcc_2_list, axis=-1, keepdims=True)

    mfcc_1_list = (mfcc_1_list - mfcc_1_list_mean) / mfcc_1_list_std
    mfcc_2_list = (mfcc_2_list - mfcc_2_list_mean) / mfcc_2_list_std

    return mfcc_1_list, mfcc_2_list, np.array(labels)

if __name__ == "__main__":
    mfcc_1, mfcc_2, labels = process_paired_dataset(PAIRED_DATASET_PATH)
    # Save the dataset as a compressed .npz file
    np.savez_compressed(OUTPUT_FILE_PATH, mfcc_1=mfcc_1, mfcc_2=mfcc_2, labels=labels)