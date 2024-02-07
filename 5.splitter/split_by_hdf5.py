# Standard library imports
import argparse
import warnings
import os
import sys
import datetime

# Third party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import h5py
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
# Local imports
sys.path.append('../')
from utils.reduction_functions import read_h5_indexes, filter_same_landmarks, filter_data, saving_reduced_hdf5, plot_length_distribution, get_args


# Title
parser = argparse.ArgumentParser(description='Split datasets using hdf5 files')

#########################################################################################################
# File paths
parser.add_argument('--folder', type=str, help='relative path of hdf5 file')
parser.add_argument('--dataset', type=str, help='Dataset name')
parser.add_argument('--csv_filename', default='label_mapping.csv', type=str, help='relative path of csv file')
parser.add_argument('--reduced', action='store_true', help='If true, the hdf5 file is the reduced subset')
parser.add_argument('--baseline', action='store_true', help='If true, the hdf5 file the baseline subset')
parser.add_argument('--downsampled', action='store_true', help='If true, the hdf5 file the downsampled subset')
parser.add_argument('--crossval', action='store_true', help='If true, the split is done for cross validation')

args = parser.parse_args()

csv_filename = args.csv_filename
DATASET = args.dataset
KPMODEL = "mediapipe"

if args.reduced:
    SUBSET = "Reduced"
elif args.baseline:
    SUBSET = "Baseline"
elif args.downsampled:
    SUBSET = "Downsampled"
else:
    SUBSET = "Original"


if SUBSET == "Reduced":
    h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}_reduced--mediapipe.hdf5')
elif SUBSET == "Downsampled":
    h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}_downsampled--mediapipe.hdf5')
else:
    h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}--mediapipe.hdf5')
print(f"Reading HDF5 file from {h5_filepath}...")

original_classes, original_videoNames, original_dataArrs = read_h5_indexes(h5_filepath)

if args.crossval and DATASET == "LSA64":
    n_splits = 5
    # Create an instance of StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    csv_path = os.path.join(args.folder, f'{DATASET}/Data/CSV/{csv_filename}')
    csv_labels_path = os.path.join(args.folder, f'{DATASET}/Data/CSV/LSA64_monitoring_reduced.csv')
    
    label_mapping_df = pd.read_csv(csv_path, index_col=0)
    reduced_labels_df = pd.read_csv(csv_labels_path, index_col=0)

    label_mapping_df = label_mapping_df.fillna("None")
    label_mapping = label_mapping_df.to_dict()['meaning']

    with h5py.File(h5_filepath, "r") as ori_h5_file:

        # Get list of video names and corresponding keys
        video_names = []
        labels = []
        key_ids = []
        for key in tqdm(ori_h5_file.keys(), desc="Getting video names"):
            video_names.append(ori_h5_file[key]['video_name'][...].item().decode('utf-8'))
            labels.append(ori_h5_file[key]['label'][...].item().decode('utf-8'))
            key_ids.append(key)

        # Use StratifiedKFold to split the data into train and validation sets
        for i, (train_index, val_index) in enumerate(skf.split(video_names, labels)):
            train_video_names = [video_names[idx] for idx in train_index]
            val_video_names = [video_names[idx] for idx in val_index]
            train_key_ids = [key_ids[idx] for idx in train_index]
            val_key_ids = [key_ids[idx] for idx in val_index]

            # Create new HDF5 files for train and validation sets
            train_h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}--mediapipe--Train_{i}.hdf5')
            val_h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}--mediapipe--Val_{i}.hdf5')
            
            train_h5_file = h5py.File(train_h5_filepath, "w")
            val_h5_file = h5py.File(val_h5_filepath, "w")

            print("Length of train:", len(train_video_names))
            print("Length of val:", len(val_video_names))


            # Copy data to train and validation sets
            for name in tqdm(train_key_ids, desc="Copying train videos"):
                ori_h5_file.copy(name, train_h5_file)
                label_num = ori_h5_file[name]['label'][...]
                label_str = label_num.item().decode('utf-8')  # Decode bytes to string
                label_int = int(label_str)  # Convert label to integer
                train_h5_file[name]['label'][...] = label_mapping[label_int].encode('utf-8')

            for name in tqdm(val_key_ids, desc="Copying validation videos"):
                ori_h5_file.copy(name, val_h5_file)
                label_num = ori_h5_file[name]['label'][...]
                label_str = label_num.item().decode('utf-8')
                label_int = int(label_str)
                val_h5_file[name]['label'][...] = label_mapping[label_int].encode('utf-8')
            # Close HDF5 files
            train_h5_file.close()
            val_h5_file.close()

elif DATASET == "AUTSL":
    # Read train and test labels from CSV files

    csv_path = os.path.join(args.folder, f'{args.dataset}/Data/Videos')
    print(f"Reading CSV files from {csv_path}...")

    train_labels_df = pd.read_csv(os.path.join(csv_path, 'train_labels.csv'), header=None, names=['video_name', 'label'])
    test_labels_df = pd.read_csv(os.path.join(csv_path, 'test_labels.csv'), header=None, names=['video_name', 'label'])

    print('Creating train and validation sets...')
    print('Train path:', os.path.join(csv_path, 'test_labels.csv'))
    print('Validation path:', os.path.join(csv_path, 'test_labels.csv'))

    with h5py.File(h5_filepath, "r") as ori_h5_file:
        # Create new HDF5 files for train and validation sets
        train_h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}--mediapipe--Train.hdf5')
        val_h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}--mediapipe--Val.hdf5')

        train_h5_file = h5py.File(train_h5_filepath, "w")
        val_h5_file = h5py.File(val_h5_filepath, "w")

        # Split video names into train and test sets based on the video_name
        for name, group in ori_h5_file.items():
            video_name = group["video_name"][()].decode('utf-8')
            split_name = video_name.split("/")
            if split_name[0] == "train":
                ori_h5_file.copy(name, train_h5_file)
            elif split_name[0] == "test":
                ori_h5_file.copy(name, val_h5_file)
        
        # Close HDF5 files
        train_h5_file.close()
        val_h5_file.close()

        # Show the number of items for train and test
        with h5py.File(train_h5_filepath, "r") as train_h5_file:
            print("Number of items in train:", len(train_h5_file.keys()))
            
        with h5py.File(val_h5_filepath, "r") as val_h5_file:
            print("Number of items in test:", len(val_h5_file.keys()))

elif DATASET == "LSA64":
    csv_path = os.path.join(args.folder, f'{DATASET}/Data/CSV/{csv_filename}')
    csv_labels_path = os.path.join(args.folder, f'{DATASET}/Data/CSV/LSA64_monitoring_reduced.csv')
    
    label_mapping_df = pd.read_csv(csv_path, index_col=0)
    reduced_labels_df = pd.read_csv(csv_labels_path, index_col=0)

    label_mapping_df = label_mapping_df.fillna("None")
    label_mapping = label_mapping_df.to_dict()['meaning']
    
    with h5py.File(h5_filepath, "r") as ori_h5_file:
        # Create new HDF5 files for train and validation sets
        train_h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}--mediapipe--Train.hdf5')
        val_h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}--mediapipe--Val.hdf5')
        
        train_h5_file = h5py.File(train_h5_filepath, "w")
        val_h5_file = h5py.File(val_h5_filepath, "w")

        # Get list of video names and corresponding keys
        video_names = []
        labels = []
        key_ids = []
        for key in tqdm(ori_h5_file.keys(), desc="Getting video names"):
            video_names.append(ori_h5_file[key]['video_name'][...].item().decode('utf-8'))
            labels.append(ori_h5_file[key]['label'][...].item().decode('utf-8'))
            key_ids.append(key)
        
        if SUBSET == "Reduced" or SUBSET == "Baseline":
            print(f"Working with {SUBSET} dataset")
            # Read the CSV to find out which classes have less than 50 instances after reduction
            reduced_class_info = reduced_labels_df['Reduced'].to_dict()
            reduced_class_val = reduced_labels_df['Has_005'].to_dict()
            

            print(reduced_class_info)

            # Split video names and keys into train and validation sets
            train_video_names = []
            val_video_names = []
            train_key_ids = []
            val_key_ids = []
            for video_name, label, key_id in zip(video_names, labels, key_ids):
                # print("Current video name:", video_name, "Current label:", label, "Current key:", key_id)

                class_instance_count = reduced_class_info.get(int(label), 0)
                # print("Class instance count {} for class {}".format(class_instance_count, label))

                if class_instance_count >=50:
                    if video_name.split('_')[-1][:3] != "005":
                        train_video_names.append(video_name)
                        train_key_ids.append(key_id)
                        # print(f"Class {label} has more than 50 instances and current video is no '005' video. Taking {video_name} for training.")
                    elif video_name.split('_')[-1][:3] == "005":
                        val_video_names.append(video_name)
                        val_key_ids.append(key_id)
                        # print(f"Class {label} has more than 50 instances and current video is '005' video. Taking {video_name} for validation.")
                elif class_instance_count > 1 and class_instance_count < 50:
                    class_instance_has_005 = reduced_class_val.get(int(label), False)
                    if class_instance_has_005:
                        print(video_name)
                        if video_name.split('_')[-1][:3] != "005":
                            train_video_names.append(video_name)
                            train_key_ids.append(key_id)
                            print(f"Class {label} has less than 50 instances and has '005' video. Taking {video_name} for training.")
                        elif video_name.split('_')[-1][:3] == "005":
                            val_video_names.append(video_name)
                            val_key_ids.append(key_id)
                            print(f"Class {label} has less than 50 instances and current video is '005' video. Taking {video_name} for validation.")
            
            print("Length of train:", len(train_video_names))
            print("Length of val:", len(val_video_names))


        elif SUBSET == "Original" or SUBSET == "Downsampled":
            print("Working with original dataset")
            # Split video names and keys into train and validation sets
            train_video_names = [name for name in video_names if name.split('_')[-1][:3] != "005"]
            train_key_ids = [key for key, name in zip(key_ids, video_names) if name.split('_')[-1][:3] != "005"]
            val_video_names = [name for name in video_names if name.split('_')[-1][:3] == "005"]
            val_key_ids = [key for key, name in zip(key_ids, video_names) if name.split('_')[-1][:3] == "005"]

            print("Length of train:", len(train_video_names))
            print("Length of val:", len(val_video_names))

        # Copy data to train and validation sets
        for name in tqdm(train_key_ids, desc="Copying train videos"):
            ori_h5_file.copy(name, train_h5_file)
            label_num = ori_h5_file[name]['label'][...]
            label_str = label_num.item().decode('utf-8')  # Decode bytes to string
            label_int = int(label_str)  # Convert label to integer
            train_h5_file[name]['label'][...] = label_mapping[label_int].encode('utf-8')

        for name in tqdm(val_key_ids, desc="Copying validation videos"):
            ori_h5_file.copy(name, val_h5_file)
            label_num = ori_h5_file[name]['label'][...]
            label_str = label_num.item().decode('utf-8')
            label_int = int(label_str)
            val_h5_file[name]['label'][...] = label_mapping[label_int].encode('utf-8')

        # Close HDF5 files
        train_h5_file.close()
        val_h5_file.close()
else:
    # TODO: Implement random train/val split for other datasets
    # Randomly split data into train and validation sets
    with h5py.File(h5_filepath, "r") as ori_h5_file:
        video_names = list(ori_h5_file.keys())
        np.random.shuffle(video_names)
        split_idx = int(len(video_names) * 0.8)
        train_video_names = video_names[:split_idx]
        val_video_names = video_names[split_idx:]

        # Create new HDF5 files for train and validation sets
        train_h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}--mediapipe--Train.hdf5')
        val_h5_filepath = os.path.join(args.folder, f'{args.dataset}/Data/H5/{SUBSET}/{args.dataset}--mediapipe--Val.hdf5')

        train_h5_file = h5py.File(train_h5_filepath, "w")
        val_h5_file = h5py.File(val_h5_filepath, "w")
        print('Writing train and validation sets to HDF5 files...')
        print('Train path:', train_h5_filepath)
        print('Validation path:', val_h5_filepath)
        # Copy data to train and validation sets
        for name in train_video_names:
            ori_h5_file.copy(name, train_h5_file)
            train_h5_file[name]['label'][...] = ori_h5_file[name]['label'][...]
            train_h5_file[name]['video_name'][...] = ori_h5_file[name]['video_name'][...]

        for name in val_video_names:
            ori_h5_file.copy(name, val_h5_file)
            val_h5_file[name]['label'][...] = ori_h5_file[name]['label'][...]
            val_h5_file[name]['video_name'][...] = ori_h5_file[name]['video_name'][...]

        # Close HDF5 files
        ori_h5_file.close()
        train_h5_file.close()
        val_h5_file.close()