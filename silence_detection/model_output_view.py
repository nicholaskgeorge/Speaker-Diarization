import numpy as np
from pydub import AudioSegment
from signal_energy_calculation import signal_energy
from zero_cross_rate import zcr_calc

def apply_decision_function(decision_function, audio_file_path):
    """
    Applies a custom decision function to an audio file and returns the raw output value.

    Args:
        decision_function (function): The custom decision function to be applied to the audio file.
        audio_file_path (str): Path to the audio file to be processed.

    Returns:
        float: The raw output value from the decision function.
    """
    # Load the audio file (use pydub or wave module)
    audio = AudioSegment.from_wav(audio_file_path)
    
    # Convert the audio to a numpy array (signal)
    signal = np.array(audio.get_array_of_samples())
    
    # Apply the custom decision function to the audio signal and return the result
    result = decision_function(signal)
    
    return result


# Test the function with a sample file and decision function
audio_file = r"C:\Users\nicok\Speaker-Diarization\data\silence_detection_data\Data\edited\silence_detect_clip_6_1.wav"  # Replace with your audio file path
output = apply_decision_function(zcr_calc, audio_file)

print(f"Decision function output: {output}")
