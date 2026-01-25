"""
Docstring for script

We assume the folders are in the same directory
In each folder we:
    List name in alphabeticla or ID order
    Label or signify the percentages
        Train: 0% → 75%
        Gap (discard): 75% → 80%
        Val: 80% → 90%
        Gap (discard): 90% → 95%
        Test: 95% → 100%
    Make Train directory and put the 0-75% of sorted images in the directory
    Make Val directory and do the same as defined above
    Similar for Test directory

After that In each folder they are splitted according to ratio
Algo/Function 2:
Make the three directory in the parent folder
Inside each folder containing the images
    Go inside train, copy/cut all images to 2 step outside and inside train
    Similar for validate and test
Repeat for other foldrs containing the images

"""
from os import mkdir
import shutil
from pathlib import Path

def pairs_creator(dir_path):
    #This is the first function defined above in the algo
    root = Path(dir_path)
    train, val, test = [], [], []


    for folder in root.iterdir():
        
        if not folder.is_dir():
            continue

        images = {}
        labels = {}

        for file in folder.iterdir():
            if file.is_file():
                if file.suffix==".jpg":
                    images[file.stem] = file
                elif file.suffix==".txt":
                    labels[file.stem]=file

        stems = sorted(images.keys())

        pairs = []
        for stem in stems:
            if stem in labels:
                pairs.append((images[stem], labels[stem]))

        total = len(pairs)
        split_75 = int(0.75*total)
        split_80 = int(0.80*total)
        split_90 = int(0.90*total)
        split_95 = int(0.95*total)

        train.extend(pairs[:split_75])
        val.extend(pairs[split_80:split_90])
        test.extend(pairs[split_95:])
    return(train, val, test)

def move_pairs(pairs, image_dst, label_dst):
    image_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)

    for img, lbl in pairs:
        shutil.copy(img, image_dst / img.name)
        shutil.copy(lbl, label_dst / lbl.name)



# MAIN PROGRAM

train, val, test = pairs_creator("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Both/")

mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)")
mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/train")
mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/test")
mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/val")
mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/train/images")
mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/train/labels")
mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/test/images")
mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/test/labels")
mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/val/images")  
mkdir("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/val/labels")

train_img_dest = Path("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/train/images")
train_label_dest = Path("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/train/labels")

test_img_dest = Path("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/test/images")
test_label_dest = Path("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/test/labels")
val_img_dest = Path("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/val/images")
val_label_dest = Path("/home/aayush/Documents/Dataset Manipulation/Edited/A1 Post Merge(Fixed + Removed Classes + Both lanes)/val/labels")



move_pairs(train, train_img_dest, train_label_dest)
move_pairs(test, test_img_dest, test_label_dest)
move_pairs(val, val_img_dest, val_label_dest)
 
