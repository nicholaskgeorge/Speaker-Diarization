import os

def delete_files_in_folder(folder_path):
    try:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):  # Check if the path is a file
                os.remove(file_path)  # Delete the file
                # print(f"Deleted file: {file_path}")
        print("All files deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

