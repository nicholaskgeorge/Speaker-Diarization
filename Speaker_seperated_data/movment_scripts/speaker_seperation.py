""""
This file is used to go into the datasets and move them to the speaker seperated folder.

"""

import os
import shutil


# Source folder containing folders with .wav files
source_folder = r"C:\Users\nicok\Speaker-Diarization\Raw_Datasets\archive\VCTK-Corpus\VCTK-Corpus\wav48"

# Destination folder where .wav files will be moved
destination_folder = r'C:\Users\nicok\Speaker-Diarization\Speaker_seperated_data\data'

# Enumerate through folders in the source folder
for idx, folder_name in enumerate(os.listdir(source_folder)):
    if os.path.isdir(os.path.join(source_folder, folder_name)):
        # Create new folder name with 's' prefix and enumeration
        new_folder_name = f's{idx + 1}'
        os.makedirs(os.path.join(destination_folder, new_folder_name), exist_ok=True)
        
        # Enumerate through .wav files in the current folder
        for file_idx, filename in enumerate(os.listdir(os.path.join(source_folder, folder_name))):
            if filename.endswith('.wav'):
                # Generate new filename with folder prefix and enumeration
                new_filename = f'{new_folder_name}_{file_idx + 1}.wav'
                # Move the file to the destination folder with the new name
                shutil.move(os.path.join(source_folder, folder_name, filename),
                            os.path.join(destination_folder, new_folder_name, new_filename))                                                                                                                    
                
print('Files moved successfully.')

