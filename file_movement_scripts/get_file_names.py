import os

def get_file_names(file_path):
    files = []
    for file_name in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, file_name)):
            files.append(file_name)
    return files