import numpy as np
import os

def get_files_in_folder(path):
    #get names of all files in data folder
    abs_path = os.path.abspath(path)
    files = []
    for file_name in os.listdir(abs_path):
        if os.path.isfile(os.path.join(abs_path, file_name)):
            files.append(file_name)
    return files

