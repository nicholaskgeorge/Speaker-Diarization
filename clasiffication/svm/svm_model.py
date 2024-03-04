import librosa
import numpy as np
import pickle
import os
from sklearn import svm
from sklearn.metrics import accuracy_score

training_data_set = r"C:\Users\nicok\Speaker-Diarization\Training_Data\Numpy_data\speaker_training_data.npy"
training_label_path = r"C:\Users\nicok\Speaker-Diarization\Training_Data\Numpy_data\speaker_training_labels.npy"
testing_data_set = r"C:\Users\nicok\Speaker-Diarization\Training_Data\Numpy_data\speaker_testing_data.npy"
testing_label_path = r"C:\Users\nicok\Speaker-Diarization\Training_Data\Numpy_data\speaker_testing_labels.npy"
mean_path = r"C:\Users\nicok\Speaker-Diarization\Training_Data\Numpy_data\speaker_data_feature_mean.npy"

#Load all of the data
X_train = np.load(training_data_set)
y_train = np.load(training_label_path)
X_test = np.load(testing_data_set)
y_test = np.load(testing_label_path)
mean = np.load(mean_path)

print(X_train.shape)

# Create an SVM classifier
clf = svm.SVC(kernel='rbf')

# Train the SVM classifier
clf.fit(X_train, y_train)

# Predict labels for the test set
y_pred = clf.predict(X_test)


# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print("Overall Accuracy:", accuracy)


filename = r'C:\Users\nicok\Speaker-Diarization\Models\svm_model.sav'
pickle.dump(clf, open(filename, 'wb'))