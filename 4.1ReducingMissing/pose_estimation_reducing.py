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

# Local imports
sys.path.append('../')
from utils.reduction_functions import read_h5_indexes, filter_same_landmarks, filter_data, saving_reduced_hdf5, plot_length_distribution, get_consecutive_missing_stats, get_descriptive_stats


# Title
parser = argparse.ArgumentParser(description='Analyze distribution of datasets')

#########################################################################################################
# File paths
parser.add_argument('--h5_path', type=str, help='relative path of hdf5 file')
parser.add_argument('--dataset', type=str, help='Dataset name')
parser.add_argument('--train', type=bool, default=False, help='Train Flag')
parser.add_argument('--val', type=bool, default=False, help='Validation Flag')
parser.add_argument('--plot', type=bool, default=False, help='Plot Flag')
parser.add_argument('--save', type=bool, default=False, help='Save Flag')
parser.add_argument('--consecutive_stats', type=bool, default=False, help='Consecutive Stats Flag')
args = parser.parse_args()


#########################################################################################################

DATASET = args.dataset
args.h5_path = os.path.normpath(args.h5_path)
KPMODEL = 'mediapipe'
VAL = args.val
TRAIN = args.train
print(f'Validation Flag set to {VAL} and Train Flag set to {TRAIN}')

print(DATASET)
print(KPMODEL)

h5_path = os.path.join(args.h5_path, f'{DATASET}--{KPMODEL}.hdf5')
print("Opening hdf5 file from path:",h5_path)

classes, videoName, dataArrs = read_h5_indexes(h5_path)

arrClasses = np.array(classes)
arrVideoNames = np.array(videoName, dtype=object)
arrData = np.array(dataArrs, dtype=object)

print(arrData.shape)

has_zeros = lambda arg: np.any(np.sum(arg, axis=0) == 0) #For a time step has zeros or not
has_consecutive_trues = lambda arg1,arg2: np.any(np.convolve(arg1.astype(int), np.ones(arg2), mode='valid') >= arg2)




bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
if DATASET == "AEC":
    bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
elif DATASET == "AUTSL":
    bann = ['apple','computer','fish','kiss','later','no','orange','pizza','purple','secretary','shirt','sunday','take','water','yellow']
elif DATASET == "PUCP_PSL_DGI156":
    bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]
    bann += ["sí","ella","uno","ese","ah","dijo","llamar"]
else:
    bann = ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"]

# Filter data for each dataset

reduced_classes, reduced_videoNames, reduced_dataArrs, baseline_dataArrs = filter_same_landmarks(h5_path,left_hand_slice=slice(501, 521), right_hand_slice=slice(522,542))

# Calculate total number of frames in the whole dataset
total_frames = sum([len(arr) for arr in baseline_dataArrs])

# Calculate total number of frames in the reduced dataset
reduced_total_frames = sum([len(arr) for arr in reduced_dataArrs])

# Filter data for each dataset
if DATASET == "AEC":
    filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total = filter_data(baseline_dataArrs, reduced_videoNames, reduced_classes, min_instances=10, banned_classes=bann)
    reduced_filtered_dataArrs, reduced_filtered_videoNames, reduced_filtered_classes, reduced_valid_classes, reduced_valid_classes_total = filter_data(reduced_dataArrs, reduced_videoNames, reduced_classes, min_instances=10, banned_classes=bann)
elif DATASET == "AUTSL":
    filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total = filter_data(baseline_dataArrs, reduced_videoNames, reduced_classes, top_k_classes=55, banned_classes=bann)
    reduced_filtered_dataArrs, reduced_filtered_videoNames, reduced_filtered_classes, reduced_valid_classes, reduced_valid_classes_total = filter_data(reduced_dataArrs, reduced_videoNames, reduced_classes, top_k_classes=55, banned_classes=bann)
elif DATASET == "INCLUDE":
    filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total = filter_data(baseline_dataArrs, reduced_videoNames, reduced_classes, min_instances=14, banned_classes=bann)
    reduced_filtered_dataArrs, reduced_filtered_videoNames, reduced_filtered_classes, reduced_valid_classes, reduced_valid_classes_total = filter_data(reduced_dataArrs, reduced_videoNames, reduced_classes, min_instances=14, banned_classes=bann)
elif DATASET == "PUCP_PSL_DGI305":
    filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total = filter_data(baseline_dataArrs, reduced_videoNames, reduced_classes, min_instances=10, banned_classes=bann)
    reduced_filtered_dataArrs, reduced_filtered_videoNames, reduced_filtered_classes, reduced_valid_classes, reduced_valid_classes_total = filter_data(reduced_dataArrs, reduced_videoNames, reduced_classes, min_instances=10, banned_classes=bann)
elif DATASET == "LSA64":
    filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes, valid_classes_total = filter_data(baseline_dataArrs, reduced_videoNames, reduced_classes, banned_classes=bann)
    reduced_filtered_dataArrs, reduced_filtered_videoNames, reduced_filtered_classes, reduced_valid_classes, reduced_valid_classes_total = filter_data(reduced_dataArrs, reduced_videoNames, reduced_classes, banned_classes=bann)
else:
    raise ValueError("Invalid dataset name.")

# Save results to log file
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
log_name = f"{DATASET}_{now}.log"
log_path = os.path.join(args.h5_path, log_name)
print(log_path)
with open(log_path, "w") as f:
    f.write(f"Dataset: {DATASET}\n")
    f.write(f"Number of videos before reduction: {len(videoName)}\n")
    f.write(f"Number of videos after reduction: {len(reduced_videoNames)}\n")
    f.write(f"Total number of frames: {total_frames}\n")
    f.write(f"Total number of frames in reduced dataset: {reduced_total_frames}\n")
    f.write("\n")
    f.write("*"*50)
    f.write("\n")
    f.write(f"Filtered data for {DATASET}\n")
    f.write(f"Classes: {len(filtered_classes)}\n")
    f.write(f"Valid classes: {len(valid_classes)}\n")
    f.write(f"Valid classes with non-minimum instances: {len(valid_classes_total)}\n")
    f.write("\n")
    f.write("*"*50)
    f.write("\n")
    f.write(f"Reduced filtered data for {DATASET}\n")
    f.write(f"Reduced classes: {len(reduced_filtered_classes)}\n")
    f.write(f"Reduced valid classes: {len(reduced_valid_classes)}\n")
    f.write(f"Reduced valid classes with non-minimum instances: {len(reduced_valid_classes_total)}\n")

if args.consecutive_stats == True:
    n_frames_tuple, percentage_tuple, num_false_seq = get_consecutive_missing_stats(dataArrs, left_hand_slice=slice(501, 521), right_hand_slice=slice(522,542), consecutive_trues=has_consecutive_trues)

    num_frames_original, num_frames_reduced = n_frames_tuple
    percentage_removed, max_consec_percentage = percentage_tuple
    
    with open("consecutive_missing_frames.log", "w") as f:
        f.write("Descriptive statistics for consecutive missing frames analysis:\n")
        f.write(f"Number of frames in the original dataset: {get_descriptive_stats(num_frames_original)}\n")
        f.write(f"Number of frames in the reduced dataset: {get_descriptive_stats(num_frames_reduced)}\n")
        f.write(f"Percentage of frames removed due to missing landmarks: {get_descriptive_stats(percentage_removed)}\n")
        f.write(f"Maximum percentage of consecutive missing frames per instance: {get_descriptive_stats(max_consec_percentage)}\n")
    print("Consecutive missing frames statistics saved to consecutive_missing_frames.log")

    

if args.save == True:
    out_path = os.path.join(args.h5_path, f"{DATASET}_reduced--{KPMODEL}.hdf5")
    #Saving Baseline subset
    saving_reduced_hdf5(filtered_classes,filtered_videoNames,filtered_dataArrs, out_path)
    #Saving Reduced subset
    h5_reduced_path = args.h5_path+"_reduced"
    out_reduced_path = os.path.join(h5_reduced_path, f"{DATASET}_reduced--{KPMODEL}.hdf5")
    os.makedirs(h5_reduced_path, exist_ok=True)
    saving_reduced_hdf5(reduced_filtered_classes,reduced_filtered_videoNames,reduced_filtered_dataArrs, out_reduced_path)

if args.plot == True:
    reduced_lengths = np.array(list(map(lambda x: x.shape[0], reduced_filtered_dataArrs)))
    baseline_lengths = np.array(list(map(lambda x: x.shape[0], filtered_dataArrs)))

    baseline_classes = filtered_classes
    reduced_classes = filtered_classes

    print("\nDimensions:")
    print("Baseline:",len(filtered_dataArrs))
    print("Reduced:",len(reduced_filtered_dataArrs))

    # Calculate percentage reduction for each video
    percentage_reductions = ((baseline_lengths - reduced_lengths) / baseline_lengths) * 100
    labels = ['0-20%', '20-40%', '40-60%','60-80%','80-100%']
    bins = [0, 20, 40, 60, 80, 100]

    # Create a dictionary to organize the data
    data = {'Subset': [], 'Percentage Reduction': [], 'Video Length': [], 'Class': [],'Percentage Reduction Range': []}

    # Populate the dictionary with data from reduced subset
    for i in range(len(reduced_lengths)):
        data['Subset'].append('Reduced')
        data['Percentage Reduction'].append(percentage_reductions[i])
        data['Video Length'].append(reduced_lengths[i])
        data['Class'].append(reduced_classes[i])
        if percentage_reductions[i] == 0:
            data['Percentage Reduction Range'].append(labels[0])
        else:
            reduction_range = pd.cut([percentage_reductions[i]], bins=bins, labels=labels, right=False)[0]
            data['Percentage Reduction Range'].append(reduction_range)


        data['Subset'].append('Baseline')
        data['Percentage Reduction'].append(0)
        data['Video Length'].append(baseline_lengths[i])
        data['Class'].append(baseline_classes[i])
        if percentage_reductions[i] == 0:
            data['Percentage Reduction Range'].append(labels[0])
        else:
            reduction_range = pd.cut([percentage_reductions[i]], bins=bins, labels=labels, right=False)[0]
            data['Percentage Reduction Range'].append(reduction_range)


    # Convert the dictionary to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Group the instances into percentage reduction groups
    # Convert 'Percentage Reduction Range' column to categorical column with sorted order
    df['Percentage Reduction Range'] = pd.Categorical(df['Percentage Reduction Range'], categories=labels, ordered=True)
    grouped_df = df.groupby(['Percentage Reduction Range', 'Subset'])


    # Calculate total number of videos in the dataset
    total_videos = len(df)

    # Calculate percentage of videos in the total dataset for each countplot category
    countplot_percentages = (grouped_df.size() / total_videos) * 100

    # Set Seaborn whitegrid style and grayscale palette
    sns.set_style("whitegrid")
    # own_palette = ['#e66101','#fdb863','#b2abd2','#5e3c99']
    # sns.set_palette(own_palette)

    # Create a figure with nested violin plots
    plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[3, 1])

    # Create the violin plot on the left subplot
    ax1 = plt.subplot(gs[0])
    ax1 = sns.violinplot(x='Percentage Reduction Range', y='Video Length', hue='Subset', data=df, ax=ax1)


    # Perform Kruskal-Wallis test for each group
    for group in labels:
        group_data = df[df['Percentage Reduction Range'] == group]
        reduced_data = group_data[group_data['Subset'] == 'Reduced']['Video Length']
        baseline_data = group_data[group_data['Subset'] == 'Baseline']['Video Length']
        stat, p_value = stats.kruskal(reduced_data, baseline_data)
        print(f"Kruskal-Wallis Test for Group '{group}':")
        print(f"  Statistic: {stat}")
        print(f"  p-value: {p_value}\n")

    # Calculate sample size for each category in the violin plot
    sample_sizes = df.groupby(['Percentage Reduction Range', 'Subset']).size().reset_index(name='Sample Size')
    sample_sizes['Sample Size'] = sample_sizes['Sample Size'].astype(int)
    sample_sizes = sample_sizes.pivot(index='Percentage Reduction Range', columns='Subset', values='Sample Size')

    print(sample_sizes)


    # Create the countplot on the right subplot
    ax2 = plt.subplot(gs[1])
    ax2 = sns.barplot(x=countplot_percentages.reset_index()['Percentage Reduction Range'], y=countplot_percentages.values, ax=ax2, linewidth=0, width=0.6)

    # Set x-axis tick labels for both subplots
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Set x label and title for the violin plot
    ax1.set_xlabel("Percentage Reduction Range", fontsize=12)
    ax1.set_ylabel("Video Length [# frames]", fontsize=12)
    ax1.legend(loc="center right")
    # plt.title(f'Impact of Reduction of Frames with Missing Landmarks in {DATASET}', loc='left')

    # Set y label for the countplot
    ax2.set_ylabel("Percentage of Videos [%]", fontsize=12)

    plt.tight_layout()
    if VAL and not TRAIN:
        plt.savefig(f"../LREC2024/Figures/{DATASET}_new_plot-Val.png", dpi=300)
        plt.savefig(f"../LREC2024/Figures/{DATASET}_new_plot-Val.svg", dpi=300)
    elif TRAIN and not VAL:
        plt.savefig(f"../LREC2024/Figures/{DATASET}_new_plot-Train.png", dpi=300)
        plt.savefig(f"../LREC2024/Figures/{DATASET}_new_plot-Train.svg", dpi=300)
    else:
        plt.savefig(f"../../LREC2024/Figures/{DATASET}_new_plot.png", dpi=300)
        plt.savefig(f"../../LREC2024/Figures/{DATASET}_new_plot.svg", dpi=300)


    plot_length_distribution(baseline_lengths,reduced_lengths,f'../../LREC2024/Figures/{DATASET}_length_distribution.png')

# fdataArrs, fvideoNames, fclasses, fvalid_classes, fvalid_classes_total= filter_data(arrData, videoName, classes, min_instances = min_instances, banned_classes=bann)
# filtered_dataArrs, filtered_videoNames, filtered_classes, valid_classes,valid_classes_total = filter_data(arrData_without_empty, new_videoName, new_classes, min_instances = min_instances, banned_classes=bann)
# _, _, fnew_classes, fnew_valid_classes, fnew_valid_classes_total= filter_data(arrData_without_empty, new_videoName, new_classes, min_instances = min_instances, banned_classes=[])
# filtered_dataArrs2, filtered_videoNames2, filtered_classes2, valid_classes2, valid_classes_total2 = filter_data(new_arrData, new_videoName, new_classes, min_instances = min_instances, banned_classes=bann)

# print("#################################")
# print("arrData Original Dataset with all videos")
# print('classes:',len(fclasses))
# print('valid_classes:',len(fvalid_classes))
# print('valid_classes with non min instances:',len(fvalid_classes_total))
# print("#################################")
# print("Filtered same landmarks, baseline subset (0 frames videos substracted) without banning")
# print('classes:',len(fnew_classes))
# print('valid_classes:',len(fnew_valid_classes))
# print('valid_classes with non min instances:',len(fnew_valid_classes_total))
# print("#################################")
# print("Filtered same landmarks, baseline subset (0 frames videos substracted) with banning")
# print('classes:',len(filtered_classes))
# print('valid_classes:',len(valid_classes))
# print('valid_classes with non min instances:',len(valid_classes_total))

# print("#################################")
# print("Filtered same landmarks, reduced subset with banning")
# print('classes:',len(filtered_classes2))
# print('valid_classes:',len(valid_classes2))
# print('valid_classes with non min instances:',len(valid_classes_total2))



# #Saving Baseline subset
# saving_reduced_hdf5(filtered_classes,filtered_videoNames,filtered_dataArrs,partial_output_name=DATASET)
# #Saving Reduced subset
# saving_reduced_hdf5(filtered_classes2,filtered_videoNames2,filtered_dataArrs2,partial_output_name=f"{DATASET}_reduced")

####################
# Assuming original dataset lists: classes, videoName, arrData
# Assuming filtered dataset lists: new_classes, new_videoName, new_arrData


# # Step 1: Extract relevant information for each video
# video_info = []

# for i in range(len(new_videoName)):
#     original_video_length = arrData_without_empty[i].shape[0]
#     if new_videoName[i] in new_videoName:
#         index_new_arr = new_videoName.index(new_videoName[i])
#         filtered_video_length = new_arrData[index_new_arr].shape[0]

#         # Calculate number of missing landmark frames
#         num_missing_landmark_frames = original_video_length - filtered_video_length

#         # Calculate percentage of missing landmark frames
#         perc_missing_landmark_frames = (num_missing_landmark_frames / original_video_length) * 100

#     else:
#         num_missing_landmark_frames = original_video_length
#         perc_missing_landmark_frames = 100
    
#     video_info.append({
#         'name': new_videoName[i],
#         'class': new_classes[i],
#         'original_length': original_video_length,
#         'missing_landmark_frames': num_missing_landmark_frames,
#         'perc_missing_landmark_frames': perc_missing_landmark_frames
#     })

# # Step 2: Aggregate data and analyze the distribution of missing landmarks

# # Extract the percentage of missing landmark frames for all videos
# perc_missing_landmark_frames_all = [video['perc_missing_landmark_frames'] for video in video_info]

# # Calculate mean and median percentage of missing landmark frames
# mean_perc_missing_landmark_frames = np.mean(perc_missing_landmark_frames_all)
# median_perc_missing_landmark_frames = np.median(perc_missing_landmark_frames_all)

# print(f"Mean Percentage of Missing Landmark Frames: {mean_perc_missing_landmark_frames}")
# print(f"Median Percentage of Missing Landmark Frames: {median_perc_missing_landmark_frames}")
# sns.set(font_scale=1.2, font='serif')
# sns.set_style('whitegrid')
# # Plot a histogram of the percentage of missing landmark frames
# sns.histplot(perc_missing_landmark_frames_all, stat='percent', bins=10, kde=False, edgecolor='black')
# plt.xlabel('Percentage of Frames with Missing Landmarks')
# plt.ylabel(f'% of Total Frames in Dataset {DATASET} ')
# plt.savefig(f"ESANN_2023/Figures/{DATASET}_Histogram_Distribution.png",dpi=300)


# # # Define the total width and height of each rectangle
# # total_width = 100
# # rect_height = 9

# # # Define the color for non-missing landmarks
# # non_missing_color = '#e6e6e6'

# # # Calculate the width of each rectangle
# # widths = [perc/100 * total_width for perc in perc_missing_landmark_frames_all]

# # # Plot the rectangles
# # fig, ax = plt.subplots(figsize=(10, 7))
# # rects1 = ax.barh(y=range(100, 0, -10), width=total_width, height=rect_height, color=non_missing_color, edgecolor='black')
# # rects2 = ax.barh(y=range(100, 0, -10), width=widths, height=rect_height, color='red', edgecolor='black')

# # # Customize the plot
# # ax.set_xlim(0, 100)
# # ax.set_ylim(0, 105)
# # ax.set_yticks(range(0, 110, 10))
# # ax.set_yticklabels([f'{i}%' for i in range(0, 110, 10)])
# # ax.set_xlabel('Percentage of Frames with Missing Landmarks')
# # ax.set_ylabel(f'% of Total Frames in Dataset {DATASET} ')
# # ax.set_title('Histogram of Percentage of Missing Landmark Frames')
# # plt.show()


# arr_lengths = np.array(list(map(lambda x: x.shape[0], arrData)))
# new_arr_lengths = np.array(list(map(lambda x: x.shape[0], new_arrData)))
# plot_length_distribution(arr_lengths,new_arr_lengths,f'ESANN_2023/Figures/{DATASET}_length_distribution_v1.png')

# arr_lengths = np.array(list(map(lambda x: x.shape[0], filtered_dataArrs)))
# new_arr_lengths = np.array(list(map(lambda x: x.shape[0], filtered_dataArrs2)))

# plot_length_distribution(arr_lengths,new_arr_lengths,f'ESANN_2023/Figures/{DATASET}_length_distribution_v2.png')



# # First, define the filtered_classes
# # Then, create an empty dictionary
# plt.figure()
# class_dict = {}

# # Loop through each class in filtered_classes
# for n,c in enumerate(filtered_classes):
#     # Initialize empty lists for original and reduced lengths
#     class_dict[c] = {
#             'Original Length': [],
#             'Reduced Length': []
#         }
# print(class_dict.keys())
# print(len(filtered_dataArrs))
# print(len(filtered_dataArrs2))
# for n,arr in enumerate(filtered_dataArrs):
#     c = filtered_classes[n]
#     class_dict[c]['Original Length'].append(arr.shape[0])
#     class_dict[c]['Reduced Length'].append(filtered_dataArrs2[n].shape[0])
# print(class_dict)


# df = []
# for c in class_dict:
#     for l in ['Original Length', 'Reduced Length']:
#         for v in class_dict[c][l]:
#             df.append({'Class': c, 'Length': l, 'Value': v})
# df = pd.DataFrame(df)


# print(df.head())
# # Create a new column with the start letter of each class name
# try:
#     df['Start Letter'] = df['Class'].str[0]
# except:
#     df['Start Letter'] = df['Class']
# # Sort the DataFrame by the 'Start Letter' column and the 'Value' column
# df = df.sort_values(['Start Letter', 'Value'])


# print(df.head())



# # Set font size and family
# sns.set(font_scale=1.75, font='Times New Roman')
# sns.set_style('whitegrid')

# # Create nested boxplot
# sns.boxplot(data=df, x='Value', y='Class', hue='Length')
# plt.xlabel('Length')
# plt.ylabel('Class')
# # plt.show()
# plt.savefig(f'ESANN_2023/Figures/{DATASET}_boxplot.png')



# # Calculate the percentage reduction
# df['Percentage Reduction'] = (1 - (df['Value'] / df.groupby('Class')['Value'].transform('mean'))) * 100

# # Define the percentage reduction groups
# bins = [0, 20, 40, 60, 100]
# labels = ['0-20%', '20-40%', '40-60%', '80-100%']

# # Group the instances into percentage reduction groups
# df['Percentage Reduction Group'] = pd.cut(df['Percentage Reduction'], bins=bins, labels=labels)


# plt.figure(figsize=(15,10))
# # Set the Seaborn style
# sns.set_style('whitegrid')

# # Create a violin plot with the percentage reduction group as x, original length as y, and gray palette
# sns.violinplot(x='Percentage Reduction Group', y='Value', data=df, palette='gray')

# # Set the x label and title

# plt.ylabel('\% of Frames Reduction')
# plt.title(f'Impact of Reduction of Frames with Missing Landmarks in {DATASET}')


# # Show the plot
# plt.savefig(f'ESANN_2023/Figures/{DATASET}_violin_plot.png')



# plt.figure(figsize=(15,10))
# # Set font size and family
# sns.set(font_scale=1.75, font='Times New Roman')
# sns.set_style('whitegrid')

# # Create a new column in the dataframe for percentage reduction group
# df['Percentage Reduction Group'] = pd.cut(df['Percentage Reduction'], bins=[0, 20, 40, 60, 100],
#                                           labels=['0-20%', '20-40%', '40-60%', '80-100%'])

# # Create nested boxplot
# sns.boxplot(data=df, y='Value', x='Percentage Reduction Group', hue='Length')
# plt.ylabel('Length')
# plt.xlabel('Percentage Reduction Group')
# plt.title(f'Impact of reduction of frames with missing landmarks in {DATASET}')
# plt.legend(title='Length')
# plt.savefig(f'ESANN_2023/Figures/{DATASET}_nested_boxplot.png')

#########################


    #     class_label = data_orig[0][0][0]
    #     # If the class label matches the current filtered class, add the length to the appropriate list
    #     if class_label == c:
    #         orig_lengths.append(data_orig.shape[0])
    #         reduced_lengths.append(data_reduced.shape[0])
    # # Add the lists to the class_dict dictionary with the current filtered class as the key
    # class_dict[c] = [orig_lengths, reduced_lengths]

# Create a swarm plot


# Create a pandas dataframe from the class_dict
# Calculate the mean value of video length for each class
# df['Original Mean'] = df['Original Length'].apply(lambda x: sum(x)/len(x))
# df['Reduced Mean'] = df['Reduced Length'].apply(lambda x: sum(x)/len(x))
# df['Original SD'] = df['Original Length'].apply(lambda x: np.std(x))
# df['Reduced SD'] = df['Reduced Length'].apply(lambda x: np.std(x))
# df['Original Max'] = df['Original Length'].apply(lambda x: max(x))
# df['Reduced Max'] = df['Reduced Length'].apply(lambda x: max(x))
# df['Original Min'] = df['Original Length'].apply(lambda x: min(x))
# df['Reduced Min'] = df['Reduced Length'].apply(lambda x: min(x))

# # Calculate means and standard deviations for original and reduced lengths
# # Add mean and standard deviation columns for each length type
# df['Mean'] = df.groupby(['Class', 'Length'])['Value'].transform('mean')
# df['SD'] = df.groupby(['Class', 'Length'])['Value'].transform('std')

# # Set seaborn style and font
# sns.set_style('whitegrid')
# sns.set(font_scale=1.25, font='Times New Roman')

# # Create a nested plot using catplot
# g = sns.catplot(
#     data=df,
#     x='Mean',
#     y='Class',
#     hue='Length',
#     kind='point',
#     join=False,
#     dodge=True,
#     capsize=0.2,
#     height=10,
#     aspect=0.8
# )

# # # Add error bars for standard deviation
# # for i, row in enumerate(g.ax.get_children()):
# #     if i % 2 == 1:
# #         err = df.iloc[(i-1)//2]['SD']
# #         x = row.get_xdata()
# #         y = row.get_ydata()
# #         g.ax.errorbar(x, y, xerr=err, fmt='none', color='black')

# # Set plot labels
# plt.xlabel('Video Length')
# plt.ylabel('Class')

# # Show the plot
# plt.show()



# Step 1: Extract relevant information for each video
# video_info = []

# for i in range(len(videoName)):
#     original_video_length = arrData[i].shape[0]
#     if videoName[i] in new_videoName:
#         index_new_arr = new_videoName.index(videoName[i])
#         filtered_video_length = new_arrData[index_new_arr].shape[0]

#         # Calculate number of missing landmark frames
#         num_missing_landmark_frames = original_video_length - filtered_video_length

#         # Calculate percentage of missing landmark frames
#         perc_missing_landmark_frames = (num_missing_landmark_frames / original_video_length) * 100

#     else:
#         num_missing_landmark_frames = original_video_length
#         perc_missing_landmark_frames = 100
    
#     video_info.append({
#         'name': videoName[i],
#         'class': classes[i],
#         'original_length': original_video_length,
#         'missing_landmark_frames': num_missing_landmark_frames,
#         'perc_missing_landmark_frames': perc_missing_landmark_frames
#     })
# print(video_info)

# # Step 2: Aggregate data and analyze the distribution of missing landmarks

# # Extract the percentage of missing landmark frames for all videos
# perc_missing_landmark_frames_all = [video['perc_missing_landmark_frames'] for video in video_info]

# # Calculate mean and median percentage of missing landmark frames
# mean_perc_missing_landmark_frames = np.mean(perc_missing_landmark_frames_all)
# median_perc_missing_landmark_frames = np.median(perc_missing_landmark_frames_all)

# print(f"Mean Percentage of Missing Landmark Frames: {mean_perc_missing_landmark_frames}")
# print(f"Median Percentage of Missing Landmark Frames: {median_perc_missing_landmark_frames}")

# # Plot a histogram of the percentage of missing landmark frames
# plt.hist(perc_missing_landmark_frames_all, bins=10, edgecolor='black')
# plt.xlabel('Percentage of Missing Landmark Frames')
# plt.ylabel('Number of Videos')
# plt.title('Histogram of Percentage of Missing Landmark Frames')
# plt.savefig('percentage_missing_landmark_frames_histogram.png')
# plt.show()

# #########################
# # Assuming the video_info list created in Step 1


# sns.set_style('whitegrid')
# sns.set(font_scale=1.25, font='Times New Roman')

# # Step 3: Analyze video length variation in the total dataset
# original_lengths = [video['original_length'] for video in video_info]
# filtered_lengths = [video['original_length'] - video['missing_landmark_frames']
#                     for video in video_info]

# # Calculate mean and median video length before and after filtering
# mean_original_length = np.mean(original_lengths)
# median_original_length = np.median(original_lengths)
# mean_filtered_length = np.mean(filtered_lengths)
# median_filtered_length = np.median(filtered_lengths)

# print(f"Mean Original Video Length: {mean_original_length}")
# print(f"Median Original Video Length: {median_original_length}")
# print(f"Mean Filtered Video Length: {mean_filtered_length}")
# print(f"Median Filtered Video Length: {median_filtered_length}")

# # Plot box plots comparing video length distributions before and after filtering
# data = {'Original': original_lengths, 'Filtered': filtered_lengths}
# # Convert data to a long-form DataFrame
# data = pd.DataFrame(data)

# print(data)
# print(len(video_info))
# print(video_info[0].keys())
# print(data.keys())
# print(len(data))

# fig, ax = plt.subplots(figsize=(8, 6))

# sns.catplot(data=data, kind="box")
# ax.set_ylabel('Video Length', fontsize=14, fontname='Times New Roman')
# ax.set_title('Box Plot of Video Lengths Before and After Filtering', fontsize=16, fontname='Times New Roman')

# fig.savefig('video_lengths_boxplot.png')
# plt.show()

# # Step 4: Video length variation and missing landmark distribution per class

# from collections import defaultdict

# class_video_info = defaultdict(list)
# selected_classes = []
# i = 0
# for video in video_info:
#     print("video from video_info:",video)
#     class_video_info[video['class']].append(video)
#     i+=1
#     if i>2:
#         break
# print(class_video_info)
# for cls, videos in class_video_info.items():
#     if len(videos) >= 15:
#         selected_classes.append(cls)
#         original_lengths_cls = [video['original_length'] for video in videos]
#         filtered_lengths_cls = [video['original_length'] - video['missing_landmark_frames']
#                                 for video in videos]

#         # Plot box plots comparing video length distributions before and after filtering for the class
#         data_cls = {'Original': original_lengths_cls, 'Filtered': filtered_lengths_cls}
#         data_cls = pd.DataFrame(data_cls)
#         fig_cls, ax_cls = plt.subplots(figsize=(8, 6))
        
#         sns.boxplot(data=data_cls, width=0.5)
#         ax_cls.set_ylabel('Video Length', fontsize=14, fontname='Times New Roman')
#         ax_cls.set_title(f"Box Plot of Video Lengths for Class {cls} Before and After Filtering", fontsize=16, fontname='Times New Roman')

#         fig_cls.savefig(f'video_lengths_class_{cls}_boxplot.png')
#         plt.show()

# selected_classes now contains the classes that have at least 15 instances for analysis
