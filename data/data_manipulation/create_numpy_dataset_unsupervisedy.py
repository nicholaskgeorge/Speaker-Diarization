import librosa
import numpy as np
import os 
import re
from segmentation_parameters import SEGMENTATION_LENGTH
import sys
sys.path.append(os.path.abspath(r"file_movement_scripts"))
from delete_files_in_folder import delete_files_in_folder
from get_files_in_folder import get_files_in_folder

sys.path.append(os.path.abspath(r"feature_extraction"))
from get_mfcc import get_mffcc_stuff

segmented_data_path = r"data\training_data\segmentation\audio_files"
dest_file_path = r"data\training_data\segmentation\numpy_files\cluster_data"
label_file_path = r"data\training_data\segmentation\numpy_files\cluster_labels"
use_mean = False

delete_files_in_folder(dest_file_path)
delete_files_in_folder(label_file_path)
def extract_numbers(string):
    pattern = r'point(\d+)_(\d+)_speakers_segment_(\d+)\.wav'
    match = re.search(pattern, string)
    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        c = int(match.group(3))
        return a, b, c
    else:
        return None, None, None


def load_into_matrix(path):
    #loop through folder with files

    #get names of all files in data folder
    files = []
    for file_name in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_name)):
            files.append(file_name)

    #get the number of features in the data set
    full_data_path = os.path.join(path, files[0])
    feature_num = len(get_mffcc_stuff(full_data_path))


    #initiate matrix
    data_matrix = np.empty((0, feature_num))
    label_vector = np.array([])
    run_count = 0
    old_data_point = ""
    #load each one
    for file in files:
        #get all mfcc data
        full_data_path = os.path.join(path, file)
        data_point = get_mffcc_stuff(full_data_path)
        # print(f"the shape of {file} data is {data_point.shape}")
        #add the vector to matrix
        data_matrix = np.vstack((data_matrix, data_point))

        #add to label vector
        #get the label of the data point of form "point0_23_speakers.wav"
        data_point_num, num_speakers, segment = extract_numbers(file)
        print(f"current is {data_point_num}")
        print(f"old is {old_data_point}")
        if old_data_point != data_point_num:
            print(f"running label at {data_point_num}")
            run_count += 1
            print(run_count)
            label_vector = np.append(label_vector, num_speakers)
            old_data_point = data_point_num

    
    print(f"len of labels is {len(label_vector)}")

    # Do normalization
    data_mean = data_matrix.mean(axis=0)

    return data_matrix, label_vector, data_mean

files = get_files_in_folder(segmented_data_path)
full_data_path = os.path.join(segmented_data_path, files[0])
feature_num = len(get_mffcc_stuff(full_data_path))
run_count = 0
data_matrix = np.empty((0, feature_num))
label_vector = np.array([])
#initiate matrix
old_data_point = [0,0,0]

for file in files:
    point, num_speakers, segment = extract_numbers(file)

    if old_data_point[0] != point:
        run_count = 0
        label_vector = np.append(label_vector, old_data_point[1])
        data_point_np_path = os.path.join(dest_file_path, f"point{old_data_point[0]}_{old_data_point[1]}_speakers.npy")
        np.save(data_point_np_path, data_matrix)

    if run_count == 0:
        data_matrix = np.empty((0, feature_num))


    full_data_path = os.path.join(segmented_data_path, file)
    data_point = get_mffcc_stuff(full_data_path)
    data_matrix = np.vstack((data_matrix, data_point))

    run_count += 1
    old_data_point = [point,num_speakers,segment]

label_np_path = os.path.join(label_file_path, f"all_labels.npy")
np.save(label_np_path, label_vector)
    



