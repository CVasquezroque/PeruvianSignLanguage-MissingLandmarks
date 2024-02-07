import argparse
import logging
import os
import sys
from typing import Any
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import glob
from collections import Counter
from tqdm import tqdm
sys.path.append('../')
from utils.reduction_functions import (read_h5_indexes, filter_same_landmarks, filter_data, 
                                       saving_reduced_hdf5, plot_length_distribution, 
                                       get_consecutive_missing_stats, get_descriptive_stats,
                                       create_histogram, downsampling)

# global variables
LEFT_HAND_SLICE = slice(501, 521)
RIGHT_HAND_SLICE = slice(522, 542)

# Class to load data from hdf5 file
class DataLoader:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.folder = args.folder
        self.train = args.train
        self.val = args.val
        
        self.kpmodel = args.kpmodel
        self.h5_path = self.set_h5_path()
        self.classes, self.videoName, self.dataArrs = read_h5_indexes(self.h5_path)

    def set_h5_path(self) -> str:
        if self.train and not self.val:
            return os.path.join(self.folder, f'{self.dataset}/Data/H5/Original/{self.dataset}--{self.kpmodel}--Train.hdf5')
        elif self.val and not self.train:
            return os.path.join(self.folder, f'{self.dataset}/Data/H5/Original/{self.dataset}--{self.kpmodel}--Val.hdf5')
        else:
            return os.path.join(self.folder, f'{self.dataset}/Data/H5/Original/{self.dataset}--{self.kpmodel}.hdf5')

# Class to downsampling data from hdf5 file

class DataDownsampler:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.folder = data_loader.folder
        self.dataset = data_loader.dataset
        self.kpmodel = data_loader.kpmodel
        self.train = data_loader.train
        self.val = data_loader.val
        self.bann = self.get_bann_values(self.data_loader.dataset)
    
    def downsample(self):
        self.original_classes, self.original_videoNames, self.original_dataArrs = read_h5_indexes(self.data_loader.h5_path)
        self.downsampled_classes, self.downsampled_videoNames, self.downsampled_dataArrs, self.k_downsample = downsampling(self.original_classes, self.original_videoNames, self.original_dataArrs, k=2)
        
        self.filtered_dataArrs, self.filtered_videoNames, self.filtered_classes, self.valid_classes, self.valid_classes_total = self.selection(self.data_loader.dataset, self.original_dataArrs, self.original_videoNames, self.original_classes)
        self.filtered_downsampled_dataArrs, self.filtered_downsampled_videoNames, self.filtered_downsampled_classes, self.valid_downsampled_classes, self.valid_downsampled_classes_total = self.selection(self.data_loader.dataset, self.downsampled_dataArrs, self.downsampled_videoNames, self.downsampled_classes)
        
        num_frames_original = [len(arr) for arr in self.original_dataArrs]
        num_frames_downsampled = [len(arr) for arr in self.downsampled_dataArrs]

        print(f'Number of frames in the original dataset: {np.sum(num_frames_original)}\n{get_descriptive_stats(num_frames_original)}')
        print(f'Number of frames in the downsampled dataset: {np.sum(num_frames_original)}\n{get_descriptive_stats(num_frames_downsampled)}')


    def save_data(self):
        if self.train and not self.val:
            out_downsampled_path = os.path.join(self.folder, f"{self.dataset}/Data/H5/Downsampled/{self.dataset}_downsampled_by_{self.k_downsample}--{self.kpmodel}--Train.hdf5")
        elif self.val and not self.train:
            out_downsampled_path = os.path.join(self.folder, f"{self.dataset}/Data/H5/Downsampled/{self.dataset}_downsampled_by_{self.k_downsample}--{self.kpmodel}--Val.hdf5")
        else:
            out_downsampled_path = os.path.join(self.folder, f"{self.dataset}/Data/H5/Downsampled/{self.dataset}_downsampled_by_{self.k_downsample}--{self.kpmodel}.hdf5")
        saving_reduced_hdf5(self.downsampled_classes, self.downsampled_videoNames, self.downsampled_dataArrs, out_downsampled_path)
    
    def get_bann_values(self,dataset:str) -> list:
        BANN_VALUES = {
            "AEC": ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"],
            "AUTSL": ['apple','computer','fish','kiss','later','no','orange','pizza','purple','secretary','shirt','sunday','take','water','yellow'],
            "PUCP_PSL_DGI156": ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN","sí","ella","uno","ese","ah","dijo","llamar"],
            "PUCP_PSL_DGI305": [],
            "INCLUDE": [],
            "LSA64": []
        }
        return BANN_VALUES[dataset]

    def selection(self, dataset: str, arr: list, videoNames: list, classes: list) -> tuple:
        
        print(f'Selecting valid classes for {dataset}...')
        print(f'Input classes: {len(classes)}')
        print(f'Input videos: {len(videoNames)}')
        print(f'Input data: {len(arr)}')
        
        SELECTION_PARAMS = {
            "AEC": {"min_instances": 10},
            "INCLUDE": {"min_instances": 14},
            "AUTSL": {"top_k_classes": 55},
            "PUCP_PSL_DGI305": {},
            "PUCP_PSL_DGI156": {},
            "LSA64": {}
        }

        params = SELECTION_PARAMS.get(dataset)

        if params is None:
            raise ValueError("Invalid dataset name.")
        params["banned_classes"] = self.bann

        return filter_data(arr, videoNames, classes, **params)



# Class to filter data from hdf5 file
class DataFilter:
    def __init__(self, data_loader):
        
        self.data_loader = data_loader
        self.bann = self.get_bann_values(self.data_loader.dataset)
        self.folder = self.data_loader.folder
        self.dataset = self.data_loader.dataset
        self.kpmodel = self.data_loader.kpmodel
        self.train = self.data_loader.train
        self.val = self.data_loader.val
        
        if args.csv_meaning:
            print(f'CSV file for {self.dataset} is required ...')
            csv_filename = input("Enter the path of the csv file: ")
            csv_path = os.path.join(args.folder, f'{self.dataset}/Data/CSV/{csv_filename}')
            df = pd.read_csv(csv_path)
            self.handtypes = df[['HID', 'label']].to_dict(orient='list')
        else:
            self.handtypes = None

    def generate_reduced(self):
        self.original_classes, self.original_videoNames, self.original_dataArrs = read_h5_indexes(self.data_loader.h5_path)
        print(f'Number of classes before reduction: {len(self.original_classes)}')
        print(f'Number of videos before reduction: {len(self.original_videoNames)}')

        self.reduced_classes, self.reduced_videoNames, self.reduced_dataArrs, self.baseline_dataArrs = filter_same_landmarks(self.data_loader.h5_path, handtype= self.handtypes,left_hand_slice=LEFT_HAND_SLICE, right_hand_slice=RIGHT_HAND_SLICE)
        print(f'Number of classes after reduction: {len(self.reduced_classes)}')
        print(f'Number of videos after reduction: {len(self.reduced_videoNames)}')

        print('Selecting valid classes...')
        self.filtered_dataArrs, self.filtered_videoNames, self.filtered_classes, self.valid_classes, self.valid_classes_total = self.selection(self.data_loader.dataset, self.baseline_dataArrs, self.reduced_videoNames, self.reduced_classes)
        self.reduced_filtered_dataArrs, self.reduced_filtered_videoNames, self.reduced_filtered_classes, self.reduced_valid_classes, self.reduced_valid_classes_total = self.selection(self.data_loader.dataset, self.reduced_dataArrs, self.reduced_videoNames, self.reduced_classes)

        self.num_frames_original = [len(arr) for arr in self.baseline_dataArrs]
        self.num_frames_reduced = [len(arr) for arr in self.reduced_dataArrs]

        self.percentage_removed, self.max_consec_percentage,self.max_consec_values, self.num_false_seq = get_consecutive_missing_stats(self.filtered_dataArrs, 
                                                                                        self.reduced_filtered_dataArrs, 
                                                                                        self.filtered_classes,
                                                                                        handtype= self.handtypes,
                                                                                        left_hand_slice=LEFT_HAND_SLICE,
                                                                                        right_hand_slice=RIGHT_HAND_SLICE
                                                                                    )
    def save_data(self):
        
        if self.train and not self.val:
            out_reduced_path = os.path.join(self.folder, f"{self.dataset}/Data/H5/Reduced/{self.dataset}_reduced--{self.kpmodel}--Train.hdf5")
            out_baseline_path = os.path.join(self.folder, f"{self.dataset}/Data/H5/Baseline/{self.dataset}--{self.kpmodel}--Train.hdf5")
        elif self.val and not self.train:
            out_reduced_path = os.path.join(self.folder, f"{self.dataset}/Data/H5/Reduced/{self.dataset}_reduced--{self.kpmodel}--Val.hdf5")
            out_baseline_path = os.path.join(self.folder, f"{self.dataset}/Data/H5/Baseline/{self.dataset}--{self.kpmodel}--Val.hdf5")
        else:
            out_reduced_path = os.path.join(self.folder, f"{self.dataset}/Data/H5/Reduced/{self.dataset}_reduced--{self.kpmodel}.hdf5")
            out_baseline_path = os.path.join(self.folder, f"{self.dataset}/Data/H5/Baseline/{self.dataset}--{self.kpmodel}.hdf5")

        print(f'Number of classes baseline: {len(self.valid_classes_total)}')
        print(f'Number of classes after reduction: {len(self.reduced_valid_classes_total)}')
        
        print(f'Number of videos baseline: {len(self.filtered_videoNames)}')
        print(f'Number of videos after reduction: {len(self.reduced_filtered_videoNames)}')

        saving_reduced_hdf5(self.reduced_filtered_classes, self.reduced_filtered_videoNames, self.reduced_filtered_dataArrs, out_reduced_path)
        saving_reduced_hdf5(self.filtered_classes, self.filtered_videoNames, self.filtered_dataArrs, out_baseline_path)
    
    def get_bann_values(self,dataset:str) -> list:
        BANN_VALUES = {
            "AEC": ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"],
            "AUTSL": ['apple','computer','fish','kiss','later','no','orange','pizza','purple','secretary','shirt','sunday','take','water','yellow'],
            "PUCP_PSL_DGI156": ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN","sí","ella","uno","ese","ah","dijo","llamar"],
            "PUCP_PSL_DGI305": [],
            "INCLUDE": [],
            "LSA64": []
        }
        return BANN_VALUES[dataset]

    def selection(self, dataset: str, arr: list, videoNames: list, classes: list) -> tuple:
        
        print(f'Selecting valid classes for {dataset}...')
        print(f'Input classes: {len(classes)}')
        print(f'Input videos: {len(videoNames)}')
        print(f'Input data: {len(arr)}')
        
        SELECTION_PARAMS = {
            "AEC": {"min_instances": 10},
            "INCLUDE": {"min_instances": 14},
            "AUTSL": {"top_k_classes": 55},
            "PUCP_PSL_DGI305": {},
            "PUCP_PSL_DGI156": {},
            "LSA64": {}
        }

        params = SELECTION_PARAMS.get(dataset)

        if params is None:
            raise ValueError("Invalid dataset name.")
        params["banned_classes"] = self.bann

        return filter_data(arr, videoNames, classes, **params)
    
    def has_005_video(self,videonames, class_label):
        # self.reduced_videoNames is a list of video names like 'all/013_007_005.mp4'
        return any('005.mp4' in video_name for video_name in videonames if video_name.startswith(f'all/{str(class_label).zfill(3)}'))

    def generate_csv(self, csv_path):
        # Get a list of all unique classes in both the original and reduced data sets
        original_class_counts = Counter(self.original_classes)
        reduced_class_counts = Counter(self.reduced_classes)
        
        # Get mean percentage removed for each class
        mean_percentage_removed = {cls: np.mean([np.round(self.percentage_removed[i],2) for i in range(len(self.filtered_classes)) if self.filtered_classes[i] == cls]) for cls in self.filtered_classes}
        # Determine if each class has a '005' video
        has_005_r = {cls: self.has_005_video(self.reduced_videoNames,cls) for cls in reduced_class_counts}
        has_005_o = {cls: self.has_005_video(self.original_videoNames,cls) for cls in original_class_counts}

        # Prepare data for CSV
        df_original = pd.DataFrame({
            'Class': list(original_class_counts.keys()),
            'Original': list(original_class_counts.values()),
            'Has_005': [has_005_o[cls] for cls in original_class_counts.keys()]  # Add the 'Has_005' column
        })
        df_reduced = pd.DataFrame({
            'Class': list(reduced_class_counts.keys()),
            'Reduced': list(reduced_class_counts.values()),
            'Has_005': [has_005_r[cls] for cls in reduced_class_counts.keys()],  # Add the 'Has_005' column
            'Mean percentage removed': [mean_percentage_removed[cls] for cls in reduced_class_counts.keys()]
        })

        df_original.to_csv(csv_path, index=False)
        df_reduced.to_csv(csv_path.replace('.csv', '_reduced.csv'), index=False)

class Logger:
    def __init__(self, data_loader, data_filter):
        self.data_loader = data_loader
        self.data_filter = data_filter
        self.train = data_loader.train
        self.val = data_loader.val
        self.subset = "Train" if self.train and not self.val else "Val" if self.val and not self.train else "Original"

        self.log_folder = os.path.join(self.data_loader.folder, f"{self.data_loader.dataset}/Logs")
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        self.log_file = os.path.join(self.log_folder, f"{self.data_loader.dataset}_metadata--{self.subset}.log")
        self.configure_logging()
        self.loggers = {}

    def configure_logging(self):
        logging.basicConfig(level=logging.INFO, filename=self.log_file, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        self.log = logging.getLogger(__name__)

        self.manage_log_files()

    def manage_log_files(self):
        log_files = sorted(glob.glob(os.path.join(self.log_folder, '*.log')), key=os.path.getctime, reverse=True)
        while len(log_files) > 30:
            os.remove(log_files.pop())

    def generate_log_data(self, title: str, data: dict) -> list:
        lines = [f"{title} for {self.data_loader.dataset}"]
        lines.extend([f"{k}: {v}" for k, v in data.items()])
        lines.append("*" * 50)
        return lines

    def log_metadata(self):
        data = self.generate_metadata()
        self.log_data(data, "Metadata")

    def generate_metadata(self):
        total_frames = sum(len(arr) for arr in self.data_filter.baseline_dataArrs)
        reduced_total_frames = sum(len(arr) for arr in self.data_filter.reduced_dataArrs)
        
        data = [
            f"Dataset: {self.data_loader.dataset}",
            f"Number of videos before reduction: {len(self.data_loader.videoName)}",
            f"Number of videos after reduction: {len(self.data_filter.reduced_videoNames)}",
            f"Total number of frames: {total_frames}",
            f"Total number of frames in reduced dataset: {reduced_total_frames}",
            "*" * 50,
        ]
        
        data.extend(self.generate_log_data("Filtered data", {
            "Classes": len(self.data_filter.filtered_classes),
            "Valid classes": len(self.data_filter.valid_classes),
            "Valid classes with non-minimum instances": len(self.data_filter.valid_classes_total),
        }))

        data.extend(self.generate_log_data("Reduced filtered data", {
            "Reduced classes": len(self.data_filter.reduced_filtered_classes),
            "Reduced valid classes": len(self.data_filter.reduced_valid_classes),
            "Reduced valid classes with non-minimum instances": len(self.data_filter.reduced_valid_classes_total),
        }))
        
        return data
    
    def get_logger(self, log_type):
        if log_type not in self.loggers:
            log_file = os.path.join(self.log_folder, f"{self.data_loader.dataset}_{log_type}--{self.subset}.log")
            logger = logging.getLogger(log_type)
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(log_file, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            self.loggers[log_type] = logger
        return self.loggers[log_type]

    
    def log_data(self, data, log_type: str, file_path: str = None):
        logger = self.get_logger(log_type)
        if file_path is None:
            for line in data:
                logger.info(line)
        else:
            with open(file_path, 'w') as f:
                for line in data:
                    f.write(line + '\n')

class StatsGenerator:
    def __init__(self, data_filter, data_loader, logger):
        self.data_filter = data_filter
        self.data_loader = data_loader

        self.logger = logger
        self.train = data_loader.train
        self.val = data_loader.val
        self.folder = data_loader.folder
        self.dataset = data_loader.dataset
        self.log_folder = logger.log_folder
    
    def consecutive_stats(self):
        if self.train and not self.val:
            consec_path = os.path.join(self.log_folder, f"{self.dataset}_consecutive_metadata--Train.log")
        elif self.val and not self.train:
            consec_path = os.path.join(self.log_folder, f"{self.dataset}_consecutive_metadata--Val.log")
        else:
            consec_path = os.path.join(self.log_folder, f"{self.dataset}_consecutive_metadata.log")
        
        data = [
            f"Number of frames in the original dataset: {get_descriptive_stats(self.data_filter.num_frames_original)}",
            f"Number of frames in the reduced dataset: {get_descriptive_stats(self.data_filter.num_frames_reduced)}",
            f"Percentage of frames removed due to missing landmarks: {get_descriptive_stats(self.data_filter.percentage_removed)}",
            f"Maximum number of consecutive missing frames per instance: {get_descriptive_stats(self.data_filter.max_consec_values)}",
            f"Maximum percentage of consecutive missing frames per instance: {get_descriptive_stats(self.data_filter.max_consec_percentage)}",
            f"Number of false sequences: {get_descriptive_stats(self.data_filter.num_false_seq)}"
        ]
        
        self.logger.log_data(data, "Consecutive Stats", consec_path)
        self.create_histograms()

    def create_histograms(self):
        create_histogram(self.data_filter.max_consec_values,
                         "Histogram of Maximum Consecutive Missing Frames",
                         "Number of frames from Largest Block of Consecutive Missing Frames for each instance",
                         "Frequency",
                         os.path.join(self.folder, f'{self.dataset}/Plots/{self.dataset}_max_consec_histogram.png'))
        create_histogram(self.data_filter.max_consec_percentage,
                         "Histogram of Maximum Consecutive Missing Frames [Percentages]",
                         "Percentage of frames from the Largest Block of Consecutive Missing Frames for each instance",
                         "Frequency",
                         os.path.join(self.folder, f'{self.dataset}/Plots/{self.dataset}_max_percent_consec_histogram.png'))
        create_histogram(np.array(self.data_filter.num_false_seq),
                         "Histogram of Number of Blocks of Consecutive Missing Frames",
                         "Number of Blocks of Consecutive Missing Frames for each instance",
                         "Frequency",
                         os.path.join(self.folder, f'{self.dataset}/Plots/{self.dataset}_num_false_seq_histogram.png'))

    def get_non_reduced_instances(self):
        nan_classes = []
        nan_videos = []
        for cls, max_consec_percentage, max_consec_value, video_name in zip(self.data_filter.filtered_classes, self.data_filter.max_consec_percentage, self.data_filter.max_consec_values, self.data_filter.filtered_videoNames):
            if np.isnan(max_consec_percentage) or np.isnan(max_consec_value):
                nan_classes.append(cls)
                nan_videos.append(video_name)
        csv_path = os.path.join(self.folder, f'{self.dataset}/Data/CSV/{self.dataset}_without_missing_values.csv')
        print(f'Exporting classes and videonames without missing values to {csv_path}')
        print(f'Number of classes without missing values: {len(nan_classes)}')
        print(f'Number of videos without missing values: {len(nan_videos)}')
        df = pd.DataFrame({'Class': nan_classes,'Video Name': nan_videos})
        df.to_csv(csv_path, index=False)

class DataVisualizer:
    def __init__(self, dataloader, datafilter, logger):
        self.data_loader = dataloader
        self.data_filter = datafilter
        self.logger = logger
        self.dataset = self.data_loader.dataset
        self.train = self.data_loader.train
        self.val = self.data_loader.val
        self.filtered_classes = self.data_filter.filtered_classes
        self.percentage_removed = self.data_filter.percentage_removed
        self.num_frames_reduced = self.data_filter.num_frames_reduced
        self.num_frames_original = self.data_filter.num_frames_original

        self.labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        self.bins = [0,20,40,60,80,100]
        self.data = {'Subset':[], 'Percentage Reduction':[], 'Video Length':[], 'Class':[], 'Percentage Reduction Range':[]}

    def prepare_data(self):
        for i in range(len(self.filtered_classes)):
            for subset, percentage_reduction, num_frames in zip(['Reduced', 'Baseline'], [self.percentage_removed[i], 0], [self.num_frames_reduced[i], self.num_frames_original[i]]):
                self.data['Subset'].append(subset)
                self.data['Percentage Reduction'].append(percentage_reduction)
                self.data['Video Length'].append(num_frames)
                self.data['Class'].append(self.filtered_classes[i])
                if percentage_reduction == 0:
                    self.data['Percentage Reduction Range'].append(self.labels[0])
                else:
                    reduction_range = pd.cut([percentage_reduction], self.bins, labels=self.labels, right=False)[0]
                    self.data['Percentage Reduction Range'].append(reduction_range)

        self.dataframe = pd.DataFrame(self.data)
        self.dataframe['Percentage Reduction Range'] = pd.Categorical(self.dataframe['Percentage Reduction Range'], categories=self.labels, ordered=True)

    def plot_violin_plot(self):
        sns.set_style("whitegrid")
        sns.set_context("paper")
        plt.figure(figsize=(12, 6))
        self.gs = GridSpec(1, 2, width_ratios=[3, 1])
        ax1 = plt.subplot(self.gs[0])
        ax1 = sns.violinplot(x='Percentage Reduction Range', y='Video Length', hue='Subset', data=self.dataframe, ax=ax1)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.set_xlabel('Percentage Reduction Range', fontsize=12)
        ax1.set_ylabel('Video Length [# of frames]', fontsize=12)
        ax1.legend(loc="center right")

    def plot_count_plot(self):
        grouped_df = self.dataframe.groupby(['Percentage Reduction Range', 'Subset'])
        total_videos = len(self.dataframe)
        countplot_percentages = (grouped_df.size() / total_videos) * 100
        ax2 = plt.subplot(self.gs[1])
        ax2 = sns.barplot(x=countplot_percentages.reset_index()['Percentage Reduction Range'], y=countplot_percentages.values, ax=ax2, linewidth=0, width=0.6)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.set_ylabel('Percentage of Videos', fontsize=12)

    def save_plots(self):
        plt.tight_layout()
        if self.val and not self.train:
            plt.savefig(f'../../LREC2024/Figures/{self.dataset}_stats--val.png', dpi=300)
            plt.savefig(f'../../LREC2024/Figures/{self.dataset}_stats--val.svg', dpi=300)
        elif self.train and not self.val:
            plt.savefig(f'../../LREC2024/Figures/{self.dataset}_stats--train.png', dpi=300)
            plt.savefig(f'../../LREC2024/Figures/{self.dataset}_stats--train.svg', dpi=300)
        else:
            plt.savefig(f'../../LREC2024/Figures/{self.dataset}_stats.png', dpi=300)
            plt.savefig(f'../../LREC2024/Figures/{self.dataset}_stats.svg', dpi=300)
        plot_length_distribution(self.num_frames_original, self.num_frames_reduced, f'../../LREC2024/Figures/{self.dataset}_length_distribution.png')

    def kruskal_wallis(self):
        # Perform Kruskal-Wallis test for each group
        for group in self.labels:
            group_data = self.dataframe[self.dataframe['Percentage Reduction Range'] == group]
            reduced_data = group_data[group_data['Subset'] == 'Reduced']['Video Length']
            baseline_data = group_data[group_data['Subset'] == 'Baseline']['Video Length']
            stat, p_value = stats.kruskal(reduced_data, baseline_data)
            to_log = [f'Kruskal-Wallis test for {group} group',
                      f'Statistic: {stat}',
                      f'p-value: {p_value}']
            self.logger.log_data(to_log, f'Kruskal-Wallis test for {group} group')


def main(args):
    # Data loading
    with tqdm(total=1, desc='Loading data') as pbar:
        data_loader = DataLoader(args)
        pbar.update(1)
    # Data filtering
    with tqdm(total=1, desc='Filtering data') as pbar:
        data_filter = DataFilter(data_loader)
        data_filter.generate_reduced()
        pbar.update(1)
    # Data downsampling
    with tqdm(total=1, desc='Downsampling data') as pbar:
        data_downsampler = DataDownsampler(data_loader)
        data_downsampler.downsample()
        pbar.update(1)
    

    # Data logging
    print("Logging data...")
    logger = Logger(data_loader, data_filter)
    logger.log_metadata()

    
    # Saving data
    if args.save:
        with tqdm(total=1, desc='Saving data') as pbar:
            data_filter.save_data()
            data_downsampler.save_data()
            pbar.update(1)
    # Stats generation
    if args.c:
        with tqdm(total=1, desc='Generating stats') as pbar:
            stats_generator = StatsGenerator(data_filter, data_loader, logger)
            stats_generator.consecutive_stats()
            pbar.update(1)
    # Plotting data
    if args.plot:
        with tqdm(total=1, desc='Plotting data') as pbar:
            data_visualizer = DataVisualizer(data_loader, data_filter, logger)
            data_visualizer.prepare_data()
            data_visualizer.plot_violin_plot()
            data_visualizer.plot_count_plot()
            data_visualizer.save_plots()
            data_visualizer.kruskal_wallis()
            pbar.update(1)

    # Exporting classes and videonames without missing values
    if args.e:
        print("Exporting classes and videonames without missing values...")
        stats_generator = StatsGenerator(data_filter, data_loader, logger)
        stats_generator.get_non_reduced_instances()
    
    if args.monitoring:
        print("Generating monitoring csv...")
        data_filter.generate_csv(os.path.join(data_loader.folder, f'{data_loader.dataset}/Data/CSV/{data_loader.dataset}_monitoring.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze distribution of datasets')
    parser.add_argument('--folder', type=str, help='relative path of hdf5 file')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--train', action='store_true', help='Train Flag')
    parser.add_argument('--val', action='store_true', help='Validation Flag')
    parser.add_argument('--plot', action='store_true', help='Plot Flag')
    parser.add_argument('--save', action='store_true', help='Save Flag')
    parser.add_argument('--c', action='store_true', help='Consecutive Stats Flag')
    parser.add_argument('--kpmodel', type=str, default='mediapipe', help='Keypoint Estimator model')
    parser.add_argument('--e', action='store_true', help='Export classes and videonames without missing values')
    parser.add_argument('--monitoring', action='store_true', help='Monitoring Flag')
    parser.add_argument('--csv_meaning', action='store_true', help='CSV Meaning Flag')
    args = parser.parse_args()
    main(args)
# class DatasetAnalyzer:

#     LEFT_HAND_SLICE = slice(501, 521)
#     RIGHT_HAND_SLICE = slice(522, 542)

#     def __init__(self, args):
#         self.args = args
#         self.train = args.train
#         self.val = args.val
#         self.folder = args.folder
#         #Lets set h5_path as our folder path
#         if self.train and not self.val:
#             self.h5_path = os.path.join(args.folder, f'{args.dataset}/Data/H5/Original/{args.dataset}--{args.kpmodel}--Train.hdf5')
#         elif self.val and not self.train:
#             self.h5_path = os.path.join(args.folder, f'{args.dataset}/Data/H5/Original/{args.dataset}--{args.kpmodel}--Val.hdf5')
#         else:
#             self.h5_path = os.path.join(args.folder, f'{args.dataset}/Data/H5/Original/{args.dataset}--{args.kpmodel}.hdf5')
#         self.classes, self.videoName, self.dataArrs = read_h5_indexes(self.h5_path)
#         self.bann = self.get_bann_values(args.dataset)


#     def get_bann_values(self, dataset):
#         BANN_VALUES = {
#             "AEC": ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN"],
#             "AUTSL": ['apple','computer','fish','kiss','later','no','orange','pizza','purple','secretary','shirt','sunday','take','water','yellow'],
#             "PUCP_PSL_DGI156": ["ya", "qué?", "qué", "bien", "dos", "ahí", "luego", "yo", "él", "tú","???","NNN","sí","ella","uno","ese","ah","dijo","llamar"],
#             "PUCP_PSL_DGI305": [],
#             "INCLUDE": [],
#             "LSA64": []
#         }
#         return BANN_VALUES[dataset]

#     def generate_reduced(self):
#         self.reduced_classes, self.reduced_videoNames, self.reduced_dataArrs, self.baseline_dataArrs = filter_same_landmarks(self.h5_path, left_hand_slice=LEFT_HAND_SLICE, right_hand_slice=RIGHT_HAND_SLICE)
#         self.filtered_dataArrs, self.filtered_videoNames, self.filtered_classes, self.valid_classes, self.valid_classes_total = self.selection(self.args.dataset, self.baseline_dataArrs, self.reduced_videoNames, self.reduced_classes)
#         self.reduced_filtered_dataArrs, self.reduced_filtered_videoNames, self.reduced_filtered_classes, self.reduced_valid_classes, self.reduced_valid_classes_total = self.selection(self.args.dataset, self.reduced_dataArrs, self.reduced_videoNames, self.reduced_classes)

#         self.num_frames_original = [len(arr) for arr in self.baseline_dataArrs]
#         self.num_frames_reduced = [len(arr) for arr in self.reduced_dataArrs]

#         self.percentage_removed, self.max_consec_percentage,self.max_consec_values, self.num_false_seq = get_consecutive_missing_stats(self.filtered_dataArrs, 
#                                                                                         self.reduced_filtered_dataArrs, 
#                                                                                         self.filtered_classes,
#                                                                                         left_hand_slice=LEFT_HAND_SLICE,
#                                                                                         right_hand_slice=RIGHT_HAND_SLICE,
#                                                                                     )
    
#     def selection(self, dataset, arr, videoNames, classes):

#         SELECTION_PARAMS = {
#             "AEC": {"min_instances": 10},
#             "INCLUDE": {"min_instances": 14},
#             "AUTSL": {"top_k_classes": 55},
#             "PUCP_PSL_DGI305": {},
#             "PUCP_PSL_DGI156": {},
#             "LSA64": {}
#         }

#         params = SELECTION_PARAMS.get(dataset)

#         if params is None:
#             raise ValueError("Invalid dataset name.")
#         params["banned_classes"] = self.bann

#         return filter_data(arr, videoNames, classes, **params)

#     def log_data(self):

#         total_frames = sum([len(arr) for arr in self.baseline_dataArrs])
#         reduced_total_frames = sum([len(arr) for arr in self.reduced_dataArrs])

#         log_data_list = [
#             f"Dataset: {self.args.dataset}",
#             f"Number of videos before reduction: {len(self.videoName)}",
#             f"Number of videos after reduction: {len(self.reduced_videoNames)}",
#             f"Total number of frames: {total_frames}",
#             f"Total number of frames in reduced dataset: {reduced_total_frames}",
#             "*"*50,
#             f"Filtered data for {self.args.dataset}",
#             f"Classes: {len(self.filtered_classes)}",
#             f"Valid classes: {len(self.valid_classes)}",
#             f"Valid classes with non-minimum instances: {len(self.valid_classes_total)}",
#             "*"*50,
#             f"Reduced filtered data for {self.args.dataset}",
#             f"Reduced classes: {len(self.reduced_filtered_classes)}",
#             f"Reduced valid classes: {len(self.reduced_valid_classes)}",
#             f"Reduced valid classes with non-minimum instances: {len(self.reduced_valid_classes_total)}"
#         ]

#         self.write_log_data(os.path.join(self.folder, f"{self.args.dataset}/Logs/{self.args.dataset}_metadata.log"), log_data_list)

#     def write_log_data(self, log_path, data):
#         with open(log_path, "w") as f:
#             for line in data:
#                 f.write(line + '\n')

#     def consecutive_stats(self):
#         # has_consecutive_trues = lambda arg1,arg2: np.any(np.convolve(arg1.astype(int), np.ones(arg2), mode='valid') >= arg2)

#         if self.train and not self.val:
#             consec_path = os.path.join(self.folder, f"{self.args.dataset}/Logs/{self.args.dataset}_consecutive_metadata--Train.log")
#         elif self.val and not self.train:
#             consec_path = os.path.join(self.folder, f"{self.args.dataset}/Logs/{self.args.dataset}_consecutive_metadata--Val.log")
#         else:
#             consec_path = os.path.join(self.folder, f"{self.args.dataset}/Logs/{self.args.dataset}_consecutive_metadata.log")

#         self.write_log_data(consec_path, 
#                             [f"Number of frames in the original dataset: {get_descriptive_stats(self.num_frames_original)}",
#                              f"Number of frames in the reduced dataset: {get_descriptive_stats(self.num_frames_reduced)}",
#                              f"Percentage of frames removed due to missing landmarks: {get_descriptive_stats(self.percentage_removed)}",
#                              f"Maximum number of consecutive missing frames per instance: {get_descriptive_stats(self.max_consec_values)}",
#                              f"Maximum percentage of consecutive missing frames per instance: {get_descriptive_stats(self.max_consec_percentage)}",
#                              f"Number of false sequences: {get_descriptive_stats(self.num_false_seq)}"])

#         # Create a histogram of the max_consec_values
#         title = 'Histogram of Maximum Consecutive Missing Frames'
#         x_label = 'Number of frames from Largest Block of Consecutive Missing Frames for each instance'
#         y_label = 'Frequency'
#         path = os.path.join(self.folder, f'{self.args.dataset}/Plots/{self.args.dataset}_max_consec_histogram.png')
#         create_histogram(self.max_consec_values, title, x_label, y_label, path)

#         # Create a histogram of the max_consec_values
#         title = 'Histogram of Maximum Consecutive Missing Frames [Percentages]'
#         x_label = 'Percentage of frames from the Largest Block of Consecutive Missing Frames for each instance'
#         y_label = 'Frequency'
#         path = os.path.join(self.folder, f'{self.args.dataset}/Plots/{self.args.dataset}_max_percent_consec_histogram.png')
#         create_histogram(self.max_consec_percentage, title, x_label, y_label, path)


#         # Create a histogram of the num_false_seq
#         title = 'Histogram of Number of Blocks of Consecutive Missing Frames'
#         x_label = 'Number of Blocks of Consecutive Missing Frames for each instance'
#         y_label = 'Frequency'
#         path = os.path.join(self.folder, f'{self.args.dataset}/Plots/{self.args.dataset}_num_false_seq_histogram.png')
#         create_histogram(np.array(self.num_false_seq), title, x_label, y_label, path)


#     def get_non_reduced_instances(self):
#         nan_classes = []
#         nan_videos = []
#         for cls, max_consec_percentage, max_consec_value, video_name in zip(self.filtered_classes, self.max_consec_percentage, self.max_consec_values, self.filtered_videoNames):
#             if np.isnan(max_consec_percentage) or np.isnan(max_consec_value):
#                 nan_classes.append(cls)
#                 nan_videos.append(video_name)
#         csv_path = os.path.join(self.folder, f'{self.args.dataset}/Data/CSV/{self.args.dataset}_without_missing_values.csv')
#         print(f'Exporting classes and videonames without missing values to {csv_path}')
#         print(f'Number of classes without missing values: {len(nan_classes)}')
#         print(f'Number of videos without missing values: {len(nan_videos)}')
#         df = pd.DataFrame({'Class': nan_classes,'Video Name': nan_videos})
#         df.to_csv(csv_path, index=False)


#     def kruskal_wallis(self):
#         # Perform Kruskal-Wallis test for each group
#         for group in self.labels:
#             group_data = self.dataframe[self.dataframe['Percentage Reduction Range'] == group]
#             reduced_data = group_data[group_data['Subset'] == 'Reduced']['Video Length']
#             baseline_data = group_data[group_data['Subset'] == 'Baseline']['Video Length']
#             stat, p_value = stats.kruskal(reduced_data, baseline_data)
#             print(f'Kruskal-Wallis test for {group} group')
#             print(f'Statistic: {stat}')
#             print(f'p-value: {p_value}')

#     def plot_data(self):
#         self.labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
#         bins = [0, 20, 40, 60, 80, 100]

#         self.data = {'Subset':[], 'Percentage Reduction':[], 'Video Length':[], 'Class':[], 'Percentage Reduction Range':[]}

#         #Populating data dictionary

#         for i in range(len(self.filtered_classes)):
#             self.data['Subset'].append('Reduced')
#             self.data['Percentage Reduction'].append(self.percentage_removed[i])
#             self.data['Video Length'].append(self.num_frames_reduced[i])
#             self.data['Class'].append(self.filtered_classes[i])
#             if self.percentage_removed[i] == 0:
#                 self.data['Percentage Reduction Range'].append(self.labels[0])
#             else:
#                 reduction_range = pd.cut([self.percentage_removed[i]], bins, labels=self.labels, right=False)[0]
#                 self.data['Percentage Reduction Range'].append(reduction_range)
            
#             self.data['Subset'].append('Baseline')
#             self.data['Percentage Reduction'].append(0)
#             self.data['Video Length'].append(self.num_frames_original[i])
#             self.data['Class'].append(self.filtered_classes[i])
#             if self.percentage_removed[i] == 0:
#                 self.data['Percentage Reduction Range'].append(self.labels[0])
#             else:
#                 reduction_range = pd.cut([self.percentage_removed[i]], bins, labels=self.labels, right=False)[0]
#                 self.data['Percentage Reduction Range'].append(reduction_range)
            
#         self.dataframe = pd.DataFrame(self.data)

#         self.dataframe['Percentage Reduction Range'] = pd.Categorical(self.dataframe['Percentage Reduction Range'], categories=self.labels, ordered=True)
#         grouped_df = self.dataframe.groupby(['Percentage Reduction Range', 'Subset'])

#         total_videos = len(self.dataframe)

#         countplot_percentages = (grouped_df.size() / total_videos)*100

#         sns.set_style("whitegrid")
#         sns.set_context("paper")

#         plt.figure(figsize=(12, 6))
#         gs = GridSpec(1, 2, width_ratios=[3, 1])
        
#         ax1 = plt.subplot(gs[0])
#         ax1 = sns.violinplot(x='Percentage Reduction Range', y='Video Length', hue='Subset', data=self.dataframe, ax=ax1)

#         sample_sizes = self.dataframe.groupby(['Percentage Reduction Range', 'Subset']).size().reset_index(name='Sample Size')
#         sample_sizes['Sample Size'] = sample_sizes['Sample Size'].astype(int)
#         sample_sizes = sample_sizes.pivot(index='Percentage Reduction Range', columns='Subset', values='Sample Size')

#         print(sample_sizes)

#         self.kruskal_wallis()

#         # Create the countplot on the right subplot

#         ax2 = plt.subplot(gs[1])
#         ax2 = sns.barplot(x=countplot_percentages.reset_index()['Percentage Reduction Range'], y=countplot_percentages.values, ax=ax2, linewidth=0, width=0.6)

#         # Set x-axis tick labels for both subplots

#         ax1.tick_params(axis='both', which='major', labelsize=10)
#         ax2.tick_params(axis='both', which='major', labelsize=10)

#         # Set x label and title for the violinplot

#         ax1.set_xlabel('Percentage Reduction Range', fontsize=12)
#         ax1.set_ylabel('Video Length [# of frames]', fontsize=12)
#         ax1.legend(loc="center right")

#         # Set y label for the countplot

#         ax2.set_ylabel('Percentage of Videos', fontsize=12)

#         plt.tight_layout()

#         if self.val and not self.train:
#             plt.savefig(f'../../LREC2024/Figures/{self.args.dataset}_new_plot--val.png', dpi=300)
#             plt.savefig(f'../../LREC2024/Figures/{self.args.dataset}_new_plot--val.svg', dpi=300)
#         elif self.train and not self.val:
#             plt.savefig(f'../../LREC2024/Figures/{self.args.dataset}_new_plot--train.png', dpi=300)
#             plt.savefig(f'../../LREC2024/Figures/{self.args.dataset}_new_plot--train.svg', dpi=300)
#         else:
#             plt.savefig(f'../../LREC2024/Figures/{self.args.dataset}_new_plot.png', dpi=300)
#             plt.savefig(f'../../LREC2024/Figures/{self.args.dataset}_new_plot.svg', dpi=300)

#         plot_length_distribution(self.num_frames_original, self.num_frames_reduced, f'../../LREC2024/Figures/{self.args.dataset}_length_distribution.png')

#     def execute(self):
#         self.generate_reduced()
#         self.log_data()

#         if self.args.c:
#             self.consecutive_stats()
        
#         if self.args.save:
#             self.save_data()

#         if self.args.plot:
#             self.plot_data()

#         if self.args.e:
#             self.get_non_reduced_instances()

# def parse_args():
#     parser = argparse.ArgumentParser(description='Analyze distribution of datasets')
#     parser.add_argument('--h5_path', type=str, help='relative path of hdf5 file')
#     parser.add_argument('--dataset', type=str, help='Dataset name')
#     parser.add_argument('--train', action='store_true', help='Train Flag')
#     parser.add_argument('--val', action='store_true', help='Validation Flag')
#     parser.add_argument('--plot', action='store_true', help='Plot Flag')
#     parser.add_argument('--save', action='store_true', help='Save Flag')
#     parser.add_argument('--c', action='store_true', help='Consecutive Stats Flag')
#     parser.add_argument('--kpmodel', type=str, default='mediapipe', help='Keypoint Estimator model')
#     parser.add_argument('--e', action='store_true', help='Export classes and videonames without missing values')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     analyzer = DatasetAnalyzer(args)
#     analyzer.execute()

# if __name__ == "__main__":
#     main()
