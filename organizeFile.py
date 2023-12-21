import os
import shutil
import random

# Paths
archive_dir = ''
train_dir = ''
val_dir = ''
test_dir = ''

# Split proportions
train_split = 0.7  
val_split = 0.2  

# Collect all filenames from the archive directory
all_files = os.listdir(archive_dir)
random.shuffle(all_files)  # Shuffle the list randomly

# Calculate split indices
total_images = len(all_files)
train_end = int(train_split * total_images)
val_end = train_end + int(val_split * total_images)

# Split files
train_files = all_files[:train_end]
val_files = all_files[train_end:val_end]
test_files = all_files[val_end:]

# Function to copy files
def copy_files(files, destination):
    for file in files:
        shutil.copy(os.path.join(archive_dir, file), destination)

# Copy files to respective directories
copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)

print("Files successfully copied into train, val, and test directories.")
