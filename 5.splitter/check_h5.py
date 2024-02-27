import h5py
import os
from collections import Counter

def read_hdf5_data(file_path, min_instances=None, top_k_classes=None, banned_classes=set()):
    """Read data from an HDF5 file."""
    classes = []
    videoName = []
    data = []

    with h5py.File(file_path, "r") as f:
        for index in f.keys():
            classes.append(f[index]['label'][...].item().decode('utf-8'))
            videoName.append(f[index]['video_name'][...].item().decode('utf-8'))
            data.append(f[index]['data'][...])
    print(len(classes))
    print(len(videoName))
    print(len(data))
    # Count the occurrences of each class
    class_counts = Counter(classes)
    print(class_counts)
    # Select classes based on the given criteria
    if min_instances is not None:
        valid_classes = [cls for cls, count in class_counts.items() if count >= min_instances and cls not in banned_classes]
    elif top_k_classes is not None:
        valid_classes = [cls for cls, count in class_counts.most_common(top_k_classes) if cls not in banned_classes]
    else:
        valid_classes = [cls for cls, count in class_counts.items() if cls not in banned_classes]
    
    valid_classes_total = [cls for cls, count in class_counts.items() if cls not in banned_classes]

    return valid_classes, valid_classes_total, videoName, data

# Define the file path
k = 6
file_path = "../../Datasets/LSA64/Data/H5/Downsampled"
dataset = f"LSA64_downsampled_by_{k}"
file_path_val = os.path.join(file_path, f"{dataset}--mediapipe--Val.hdf5")
file_path_train = os.path.join(file_path, f"{dataset}--mediapipe--Train.hdf5")
# Define banned classes, if any
banned_classes = []  # Example: set(['banned_class_1', 'banned_class_2'])

# Read data from the HDF5 file
valid_classes, valid_classes_total, videoName, data = read_hdf5_data(file_path_val, banned_classes=banned_classes)
valid_classes_train, valid_classes_total_train, videoName_train, data_train = read_hdf5_data(file_path_train, banned_classes=banned_classes)

# Optionally, print out some of the data to check
print("Valid Classes:", valid_classes)
print("Video Names:", videoName)
print("Total videos:", len(videoName))
print("Total Valid Classes Val:", len(valid_classes_total))
print("Total Valid Classes Train:", len(valid_classes_total_train))
# Print out the shape of the first item in the data list to check
print("Data Shape Val:",len(data))
print("Data Shape Train:", len(data_train))