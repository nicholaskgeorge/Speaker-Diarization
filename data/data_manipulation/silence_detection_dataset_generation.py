import os
import re
from pydub import AudioSegment

def parse_textgrid(file_path):
    """
    Parses a TextGrid file and extracts silence/non-silence labels.
    
    Args:
        file_path (str): Path to the TextGrid file.
    
    Returns:
        list of dict: A list of dictionaries, each containing xmin, xmax, and text.
    """
    intervals = []
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Find intervals in the file using regex
        interval_pattern = re.compile(
            r"intervals \[\d+\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = \"(\d+)\""
        )
        matches = interval_pattern.findall(content)
        
        # Process each match
        for match in matches:
            xmin, xmax, text = match
            intervals.append({
                "xmin": float(xmin),
                "xmax": float(xmax),
                "text": text
            })
    
    return intervals

def extract_audio_segments(audio_file_path, intervals, output_dir, clip_num, min_clip_length, max_clip_length):
    """
    Extracts audio segments based on the TextGrid intervals and saves them to the output directory.
    
    Args:
        audio_file_path (str): Path to the audio file (wav).
        intervals (list of dict): The list of intervals from the TextGrid file.
        output_dir (str): The directory where the audio segments will be saved.
        clip_num (int): The current clip number.
        min_clip_length (float): The minimum length of a clip in milliseconds.
        max_clip_length (float): The maximum length of a clip in milliseconds.
    """
    # Load the audio file
    audio = AudioSegment.from_wav(audio_file_path)
    
    # Initialize variables for merging intervals
    merged_intervals = []
    current_start = intervals[0]["xmin"]
    current_end = intervals[0]["xmax"]
    current_label = intervals[0]["text"]

    # Iterate through the intervals and merge consecutive ones with the same label
    for interval in intervals[1:]:
        if interval["text"] == current_label:
            # Extend the current interval if the label is the same
            current_end = interval["xmax"]
        else:
            # Save the previous merged interval
            merged_intervals.append({
                "xmin": current_start,
                "xmax": current_end,
                "text": current_label
            })
            # Start a new interval
            current_start = interval["xmin"]
            current_end = interval["xmax"]
            current_label = interval["text"]

    # Add the last merged interval
    merged_intervals.append({
        "xmin": current_start,
        "xmax": current_end,
        "text": current_label
    })
    
    # Extract and save audio for each merged interval
    for merged_interval in merged_intervals:
        xmin = merged_interval["xmin"] * 1000  # Convert to milliseconds
        xmax = merged_interval["xmax"] * 1000  # Convert to milliseconds
        label = merged_interval["text"]
        
        # Ensure we're not extracting a segment that's out of bounds
        if xmin < 0:
            xmin = 0
        if xmax > len(audio):
            xmax = len(audio)
        
        # Extract the segment
        segment = audio[xmin:xmax]
        
        # Check if the segment length is valid (longer than the minimum clip length)
        if len(segment) >= min_clip_length:  # Check against the min clip length parameter
            # Split the segment if it exceeds max_clip_length
            while len(segment) > max_clip_length:
                split_point = max_clip_length
                segment_chunk = segment[:split_point]
                segment = segment[split_point:]
                
                # Create the output filename without the original file name
                output_filename = os.path.join(output_dir, f"silence_detect_clip_{clip_num}_{label}.wav")
                segment_chunk.export(output_filename, format="wav")
                print(f"Saved: {output_filename}")  # Log the saved clip
                
                clip_num += 1

            # If the remaining segment fits the max clip length, save it
            if len(segment) <= max_clip_length:
                output_filename = os.path.join(output_dir, f"silence_detect_clip_{clip_num}_{label}.wav")
                segment.export(output_filename, format="wav")
                print(f"Saved: {output_filename}")  # Log the saved clip
                
                clip_num += 1
        else:
            print(f"Warning: Segment for {audio_file_path} from {xmin} to {xmax} is too short (less than {min_clip_length / 1000}s) and was skipped")
    
    return clip_num

def process_folders(audio_folder, annotation_folder, output_folder, starting_clip_num=0, min_clip_length=250, max_clip_length=1500):
    """
    Processes the files in the audio folder (wav) and annotation folder (TextGrid), extracts the silence/voice clips, 
    and saves them to the output folder.
    
    Args:
        audio_folder (str): Path to the folder containing the audio files (wav).
        annotation_folder (str): Path to the folder containing the TextGrid files.
        output_folder (str): Path to the folder where the output clips will be saved.
        starting_clip_num (int): The starting number for the clip sequence (default is 0).
        min_clip_length (float): The minimum length of a clip in milliseconds (default is 250ms).
        max_clip_length (float): The maximum length of a clip in milliseconds (default is 1500ms).
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    clip_num = starting_clip_num  # Initialize the global clip number

    # Loop through each file in the audio folder
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav"):
            # Get the corresponding TextGrid file by replacing .wav with .TextGrid
            textgrid_filename = filename.replace(".wav", ".TextGrid")
            textgrid_file_path = os.path.join(annotation_folder, textgrid_filename)
            
            # Check if the TextGrid file exists
            if os.path.exists(textgrid_file_path):
                print(f"Processing: {filename}")
                
                # Parse the TextGrid file to get silence/non-silence intervals
                intervals = parse_textgrid(textgrid_file_path)
                
                # Get the path for the current audio file
                audio_file_path = os.path.join(audio_folder, filename)
                
                # Extract and save the audio segments based on the intervals
                clip_num = extract_audio_segments(audio_file_path, intervals, output_folder, clip_num, min_clip_length, max_clip_length)
            else:
                print(f"Warning: No matching TextGrid file for {filename}")

# Example usage
audio_folder = r"C:\Users\nicok\Speaker-Diarization\data\silence_detection_data\Data\original\Audio\Female\PTDB-TUG"  # Replace with your audio folder path
annotation_folder = r"C:\Users\nicok\Speaker-Diarization\data\silence_detection_data\Data\original\Annotation\Female\PTDB-TUG"  # Replace with your annotation folder path
output_folder = r"C:\Users\nicok\Speaker-Diarization\data\silence_detection_data\Data\edited"  # Replace with your output folder path
starting_clip_num = 0  # Specify the starting number for the clip sequence
min_clip_length = 500  # Minimum length for clips in milliseconds (default: 500ms)
max_clip_length = 1000  # Maximum length for clips in milliseconds (default: 1500ms)

process_folders(audio_folder, annotation_folder, output_folder, starting_clip_num, min_clip_length, max_clip_length)
