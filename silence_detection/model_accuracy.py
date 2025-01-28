import os
import wave
import numpy as np
from pydub import AudioSegment
from signal_energy_classifier import sig_classify
from zero_cross_classifier import zcr_classify

def calculate_accuracy(output_folder, decision_function):
    """
    Calculates the accuracy of the decision function on the audio clips.

    Args:
        output_folder (str): Path to the folder containing generated audio clips.
        decision_function (function): Your custom decision function that takes an audio signal as input.

    Returns:
        float: The accuracy of the decision function.
    """
    correct_predictions = 0
    total_files = 0
    
    # Loop through the output folder and check each file
    for filename in os.listdir(output_folder):
        if filename.endswith(".wav"):
            # Get the path to the audio file
            audio_file_path = os.path.join(output_folder, filename)
            
            # Extract the ground truth label from the filename
            label = int(filename.split('_')[-1].split('.')[0])  # Extract the label from the filename
            
            # Load the audio file (use pydub or wave module)
            audio = AudioSegment.from_wav(audio_file_path)
            
            # Apply the input decision function to the audio signal
            signal = np.array(audio.get_array_of_samples())  # Convert audio to numpy array
            prediction = decision_function(signal)
            
            # Compare the prediction with the ground truth
            if prediction == label:
                correct_predictions += 1
            total_files += 1
    
    # Calculate and print accuracy
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0
    print(f"Total Files: {total_files}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy

# Example usage
output_folder = r"C:\Users\nicok\Speaker-Diarization\data\silence_detection_data\Data\edited"  # Replace with your output folder path

# # Call the function to calculate accuracy, passing the decision function as input
# print("Print Sig Classify")
# calculate_accuracy(output_folder, sig_classify)

print("ZCR classify")
calculate_accuracy(output_folder, zcr_classify)