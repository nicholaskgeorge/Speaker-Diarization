import librosa

def get_mffcc_stuff(data_path):
    # load audio files with librosa
    signal, sr = librosa.load(data_path)

    #normalize the audio
    signal  = librosa.util.normalize(signal)
    #get mfccs and all deltas
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    #make into one data point
    mfccs_features = mfccs #np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

    #flatten data set
    data_point = mfccs_features.flatten()

    return data_point