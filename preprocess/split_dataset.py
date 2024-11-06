import os
import shutil
import random

def split_trajectories(source_dir, train_dir, test_dir, train_ratio=0.8):
    # Create train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all trajectory folders
    trajectory_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    # Randomly shuffle the folders
    random.shuffle(trajectory_folders)

    # Calculate the split index
    split_index = int(len(trajectory_folders) * train_ratio)

    # Split the folders
    train_folders = trajectory_folders[:split_index]
    test_folders = trajectory_folders[split_index:]

    # Copy train folders
    for folder in train_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(train_dir, folder)
        shutil.copytree(src, dst)
        print(f"Copied {folder} to train set")

    # Copy test folders
    for folder in test_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(test_dir, folder)
        shutil.copytree(src, dst)
        print(f"Copied {folder} to test set")

    print(f"Split complete. {len(train_folders)} folders in train, {len(test_folders)} folders in test.")

# Usage
source_dir = "/home/sye40/TCFM/data/march_npz"
train_dir = "/home/sye40/TCFM/data/train"
test_dir = "/home/sye40/TCFM/data/test"
train_ratio = 0.8  # 80% train, 20% test

split_trajectories(source_dir, train_dir, test_dir, train_ratio)