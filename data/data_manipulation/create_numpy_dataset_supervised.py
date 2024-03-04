import librosa
import numpy as np
import os 




testing_audio_data_path = r"C:\Users\nicok\Speaker-Diarization\Training_Data\test_set"
training_audio_data_path = r"C:\Users\nicok\Speaker-Diarization\Training_Data\training_set"
dest_file_path = r"C:\Users\nicok\Speaker-Diarization\Training_Data\Numpy_data"
use_mean = False
def get_mffcc_stuff(data_path):
    # load audio files with librosa
    signal, sr = librosa.load(data_path)

    #normalize the audio
    signal  = librosa.util.normalize(signal)
    #get mfccs and all deltas
    # print(data_path)
    # print(f"signal length is {len(signal)}")
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    # delta_mfccs = librosa.feature.delta(mfccs)
    # delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    #make into one data point
    mfccs_features = mfccs #np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

    #flatten data set
    data_point = mfccs_features.flatten()

    return data_point


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

    #load each one
    for file in files:
        #get all mfcc data
        full_data_path = os.path.join(path, file)
        data_point = get_mffcc_stuff(full_data_path)
        print(f"the shape of {file} data is {data_point.shape}")
        #add the vector to matrix
        data_matrix = np.vstack((data_matrix, data_point))

        #add to label vector
        #get the label of the data point of form "point0_23_speakers.wav"
        underscore_locs = file.find("_")
        label = int(file[underscore_locs+1:underscore_locs+2])
        label_vector = np.append(label_vector, label)

    # Do normalization
    data_mean = data_matrix.mean(axis=0)

    return data_matrix, label_vector, data_mean

# # shuffle the data
# np.random.seed(49)


# # Shuffle the indices
# shuffled_indices = np.random.permutation(len(data_matrix))

train_matrix, train_labels, train_data_mean = load_into_matrix(training_audio_data_path)
test_matrix, test_labels, test_data_mean = load_into_matrix(testing_audio_data_path)

#normalize data
if use_mean:
    train_matrix = train_matrix-train_data_mean
    test_matrix = test_matrix-train_data_mean

#save the matrix
training_data_path = os.path.join(dest_file_path, "speaker_training_data.npy")
training_label_path = os.path.join(dest_file_path, "speaker_training_labels.npy")
testing_data_path = os.path.join(dest_file_path, "speaker_testing_data.npy")
testing_label_path = os.path.join(dest_file_path, "speaker_testing_labels.npy")
data_mean_path = os.path.join(dest_file_path, "speaker_data_feature_mean.npy")

np.save(training_data_path, train_matrix)
np.save(training_label_path, train_labels)
np.save(testing_data_path, test_matrix)
np.save(testing_label_path, test_labels)
np.save(data_mean_path, train_data_mean)



