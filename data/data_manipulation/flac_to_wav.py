import os
from pydub import AudioSegment

def convert_flac_to_wav(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.flac'):
                flac_path = os.path.join(root, file)
                wav_path = os.path.splitext(flac_path)[0] + '.wav'
                
                # Load the FLAC file
                audio = AudioSegment.from_file(flac_path, format='flac')
                
                # Export as WAV, overwriting the original FLAC file
                audio.export(wav_path, format='wav')
                
                # Optionally, remove the original FLAC file
                os.remove(flac_path)
                print(f"Converted and replaced: {flac_path} -> {wav_path}")

# Replace '/path/to/your/folder' with the path to your target directory
convert_flac_to_wav('data/raw_dataset/LibriSpeech/dev-clean')