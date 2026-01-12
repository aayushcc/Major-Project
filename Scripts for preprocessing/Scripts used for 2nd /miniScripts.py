from pathlib import Path
import os
import shutil
# Removes all txt files in a specific folder structure
def removetxt():
    root = Path("/home/aayush/Documents/Dataset Manipulation/Edited/E2 copy XML only/")

    for folder in root.iterdir():

        for file in folder.iterdir():
            if file.suffix.lower() == ".txt":
                file.unlink()



def backup(input_dir: str):
    # to backup the folder before manipulating it, increaste the number below before running script again
    dst = os.path.join(os.path.dirname(input_dir), "script_backup_0")
    shutil.copytree(input_dir, dst) 

backup("/home/aayush/Documents/Dataset Manipulation/Edited/E2 copy XML only")                                                                                                                                                                       
# removetxt()