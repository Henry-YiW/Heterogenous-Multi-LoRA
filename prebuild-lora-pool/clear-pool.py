# Run this to remove all files from all three folders for the lora pool

import os
import shutil

def clear_directories():
    # Define the subdirectories
    directories = ["lora-pool", "lora-pool-meta", "lora-pool-img"]
    
    for directory in directories:
        # Check if the directory exists
        if os.path.exists(directory):
            # Remove all files and subdirectories
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete the file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete the subdirectory
            print(f"Cleared all files in '{directory}'.")
        else:
            print(f"Directory '{directory}' does not exist.")

# Run the function
clear_directories()
