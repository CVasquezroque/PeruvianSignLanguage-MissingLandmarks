# Standard library imports
import argparse
import warnings
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as stats

# Local imports
sys.path.append('../')
from utils.reduction_functions import read_h5_indexes, filter_same_landmarks, getting_filtered_data, get_args

# Title
parser = argparse.ArgumentParser(description='Analyze distribution of datasets')

#########################################################################################################
# File paths
parser.add_argument('--h5_path', type=str, help='relative path of hdf5 file')

parser.add_argument('--dataset', type=str, help='Dataset name')
parser.add_argument('--train', type=bool, default=False, help='Train Flag')
parser.add_argument('--val', type=bool, default=False, help='Validation Flag')
args = parser.parse_args()


#########################################################################################################

DATASET = args.dataset
args.h5_path = os.path.normpath(args.h5_path)
KPMODEL = 'mediapipe'
VAL = args.val
TRAIN = args.train
print(DATASET)
print(KPMODEL)
print(f'Validation Flag set to {VAL} and Train Flag set to {TRAIN}')


###########################################################################################################
if VAL and not TRAIN:
    h5_path = os.path.join(args.h5_path, f'{DATASET}--{KPMODEL}-Val.hdf5')
elif TRAIN and not VAL:
    h5_path = os.path.join(args.h5_path, f'{DATASET}--{KPMODEL}-Train.hdf5')
else:
    h5_path = os.path.join(args.h5_path, f'{DATASET}--{KPMODEL}.hdf5')

classes, videoName, dataArrs = read_h5_indexes(h5_path)

arrClasses = np.array(classes)
arrVideoNames = np.array(videoName, dtype=object)
arrData = np.array(dataArrs, dtype=object)

print(arrData.shape)
print(arrData[0].shape)
has_zeros = lambda arg: np.any(np.sum(arg, axis=0) == 0) #For a time step has zeros or not
has_consecutive_trues = lambda arg1,arg2: np.any(np.convolve(arg1.astype(int), np.ones(arg2), mode='valid') >= arg2)

###########################################################################################################
from collections import Counter

# Get number of videos
num_videos = len(arrData)

# Get number of frames per video
num_frames_per_video = [data.shape[0] for data in arrData]

# Get number of classes
class_counts = Counter(arrClasses)
num_classes = len(class_counts)
print(class_counts)
print(num_classes)

# Get mean, std, and median of frames per video
mean_frames_per_video = np.mean(num_frames_per_video)
std_frames_per_video = np.std(num_frames_per_video)
median_frames_per_video = np.median(num_frames_per_video)

print(arrClasses.shape)

print("Min number of instances",min(class_counts.values()))
print("Max number of instances",max(class_counts.values()))
print("Mean number of instances",np.mean(list(class_counts.values())))
# import matplotlib.pyplot as plt

# Count the number of classes with each number of instances
instance_counts = {}
for count in class_counts.values():
    if count in instance_counts:
        instance_counts[count] += 1
    else:
        instance_counts[count] = 1

# Create a histogram of the number of classes with each number of instances
plt.bar(instance_counts.keys(), instance_counts.values())
plt.xlabel('Number of Instances')
plt.ylabel('Number of Classes')
plt.title(f'Histogram of Num of Classes for each Num of Instances in {DATASET}')
plt.savefig(f"../../LREC2024/Figures/{DATASET}_class_histogram.png", dpi=300)


# plt.bar(class_counts.keys(), class_counts.values())
# plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# plt.xlabel('Class')
# plt.ylabel('Number of Videos')
# plt.title('Histogram of Number of Videos per Class')
# plt.show()

# min_instances = args.min_instances
# bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
# if DATASET == "AEC":
#     bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
# elif DATASET == "AUTSL":
#     bann = ['apple','computer','fish','kiss','later','no','orange','pizza','purple','secretary','shirt','sunday','take','water','yellow']
# elif DATASET == "PUCP_PSL_DGI156":
#     bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
#     bann += ["sí","ella","uno","ese","ah","dijo","llamar"]
# else:
#     bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
# #PUCP
# # self.list_labels_banned = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
# # self.list_labels_banned += ["sí","ella","uno","ese","ah","dijo","llamar"]
# min_instances = args.min_instances
# bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
# if DATASET == "AEC":
#     bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
# elif DATASET == "AUTSL":
#     bann = ['apple','computer','fish','kiss','later','no','orange','pizza','purple','secretary','shirt','sunday','take','water','yellow']
# elif DATASET == "PUCP_PSL_DGI156":
#     bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
#     bann += ["sí","ella","uno","ese","ah","dijo","llamar"]
# else:
#     bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]

# new_classes, new_videoName, new_arrData,arrData_without_empty = filter_same_landmarks(h5_path,left_hand_slice=slice(501, 521), right_hand_slice=slice(522,542))

# filtered_dataArrs,filtered_reduceArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total, max_consec,num_false_seq,percentage_reduction_categories = getting_filtered_data(arrData_without_empty,new_arrData,new_videoName,new_classes,min_instances= min_instances,banned_classes=bann)


# reduced_lengths = np.array(list(map(lambda x: x.shape[0], filtered_reduceArrs)))
# baseline_lengths = np.array(list(map(lambda x: x.shape[0], filtered_dataArrs)))

# baseline_classes = filtered_classes
# reduced_classes = filtered_classes

# print("\nDimensions:")
# print("Baseline:",len(filtered_dataArrs))
# print("Reduced:",len(filtered_reduceArrs))

# # Calculate percentage reduction for each video
# percentage_reductions = ((baseline_lengths - reduced_lengths) / baseline_lengths) * 100
# labels = ['0-20%', '20-40%', '40-60%','60-80%','80-100%']
# bins = [0, 20, 40, 60, 80, 100]

# # Create a dictionary to organize the data
# data = {'Subset': [], 'Percentage Reduction': [], 'Video Length': [], 'Class': [],'Percentage Reduction Range': []}

# # Populate the dictionary with data from reduced subset
# for i in range(len(reduced_lengths)):
#     data['Subset'].append('Reduced')
#     data['Percentage Reduction'].append(percentage_reductions[i])
#     data['Video Length'].append(reduced_lengths[i])
#     data['Class'].append(reduced_classes[i])
#     if percentage_reductions[i] == 0:
#         data['Percentage Reduction Range'].append(labels[0])
#     else:
#         reduction_range = pd.cut([percentage_reductions[i]], bins=bins, labels=labels, right=False)[0]
#         data['Percentage Reduction Range'].append(reduction_range)


#     data['Subset'].append('Baseline')
#     data['Percentage Reduction'].append(0)
#     data['Video Length'].append(baseline_lengths[i])
#     data['Class'].append(baseline_classes[i])
#     if percentage_reductions[i] == 0:
#         data['Percentage Reduction Range'].append(labels[0])
#     else:
#         reduction_range = pd.cut([percentage_reductions[i]], bins=bins, labels=labels, right=False)[0]
#         data['Percentage Reduction Range'].append(reduction_range)


# # Convert the dictionary to a Pandas DataFrame
# df = pd.DataFrame(data)

# # Group the instances into percentage reduction groups
# # Convert 'Percentage Reduction Range' column to categorical column with sorted order
# df['Percentage Reduction Range'] = pd.Categorical(df['Percentage Reduction Range'], categories=labels, ordered=True)
# grouped_df = df.groupby(['Percentage Reduction Range', 'Subset'])


# # Calculate total number of videos in the dataset
# total_videos = len(df)

# # Calculate percentage of videos in the total dataset for each countplot category
# countplot_percentages = (grouped_df.size() / total_videos) * 100

# # Set Seaborn whitegrid style and grayscale palette
# sns.set_style("whitegrid")
# # own_palette = ['#e66101','#fdb863','#b2abd2','#5e3c99']
# # sns.set_palette(own_palette)

# # Create a figure with nested violin plots
# plt.figure(figsize=(12, 6))
# gs = GridSpec(1, 2, width_ratios=[3, 1])

# # Create the violin plot on the left subplot
# ax1 = plt.subplot(gs[0])
# ax1 = sns.violinplot(x='Percentage Reduction Range', y='Video Length', hue='Subset', data=df, ax=ax1)


# # Perform Kruskal-Wallis test for each group
# for group in labels:
#     group_data = df[df['Percentage Reduction Range'] == group]
#     reduced_data = group_data[group_data['Subset'] == 'Reduced']['Video Length']
#     baseline_data = group_data[group_data['Subset'] == 'Baseline']['Video Length']
#     stat, p_value = stats.kruskal(reduced_data, baseline_data)
#     print(f"Kruskal-Wallis Test for Group '{group}':")
#     print(f"  Statistic: {stat}")
#     print(f"  p-value: {p_value}\n")

# # Calculate sample size for each category in the violin plot
# sample_sizes = df.groupby(['Percentage Reduction Range', 'Subset']).size().reset_index(name='Sample Size')
# sample_sizes['Sample Size'] = sample_sizes['Sample Size'].astype(int)
# sample_sizes = sample_sizes.pivot(index='Percentage Reduction Range', columns='Subset', values='Sample Size')

# print(sample_sizes)


# # Create the countplot on the right subplot
# ax2 = plt.subplot(gs[1])
# ax2 = sns.barplot(x=countplot_percentages.reset_index()['Percentage Reduction Range'], y=countplot_percentages.values, ax=ax2, linewidth=0, width=0.6)

# # Set x-axis tick labels for both subplots
# ax1.tick_params(axis='both', which='major', labelsize=10)
# ax2.tick_params(axis='both', which='major', labelsize=10)

# # Set x label and title for the violin plot
# ax1.set_xlabel("Percentage Reduction Range", fontsize=12)
# ax1.set_ylabel("Video Length [# frames]", fontsize=12)
# ax1.legend(loc="center right")
# # plt.title(f'Impact of Reduction of Frames with Missing Landmarks in {DATASET}', loc='left')

# # Set y label for the countplot
# ax2.set_ylabel("Percentage of Videos [%]", fontsize=12)

# plt.tight_layout()
# if VAL and not TRAIN:
#     plt.savefig(f"../ESANN_2023/Figures/{DATASET}_new_plot-Val.png", dpi=300)
#     plt.savefig(f"../ESANN_2023/Figures/{DATASET}_new_plot-Val.svg", dpi=300)
# elif TRAIN and not VAL:
#     plt.savefig(f"../ESANN_2023/Figures/{DATASET}_new_plot-Train.png", dpi=300)
#     plt.savefig(f"../ESANN_2023/Figures/{DATASET}_new_plot-Train.svg", dpi=300)
# else:
#     plt.savefig(f"../ESANN_2023/Figures/{DATASET}_new_plot.png", dpi=300)
#     plt.savefig(f"../ESANN_2023/Figures/{DATASET}_new_plot.svg", dpi=300)
