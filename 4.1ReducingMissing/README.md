# Dataset Analyzer Script Documentation

This script is designed to analyze the distribution of datasets, especially focusing on the landmarks of different datasets. Its primary purpose is to provide insights into the distribution, identify missing landmarks, and generate visualizations.

## Workflow

1. The script reads an HDF5 file containing datasets of landmarks.
2. The data is then processed to generate a reduced version by filtering out specific landmarks.
3. Various statistics and analyses (like the Kruskal-Wallis test) are performed on the dataset.
4. If specified, the script generates and saves plots visualizing the data distribution.
5. The results, including any reduced datasets, can be saved to disk.

## Required Folder Structure

The script assumes a certain folder structure to function properly. The folder structure is as follows:

```bash
.

├── LREC2024
│   ├── Figures
├── PeruvianSignLanguage
│   ├── 4.1ReducedMissing
│   │   ├── landmark_dataset_analysis.py
│   │   ├── other scripts...
│   └── Data
│       ├── AEC
│       │   ├── AEC--mediapipe.hdf5
│       │   └── other_dataset_files...
│       ├── AUTSL
│       │   ├── AUTSL--mediapipe.hdf5
│       │   └── other_dataset_files...
│       └── other_datasets...
└── other folders with scripts of PeruvianSignLanguage ...
```

## DatasetAnalyzer Class

### Attributes:

- **LEFT_HAND_SLICE**: Slice indicating the landmarks of the left hand.
- **RIGHT_HAND_SLICE**: Slice indicating the landmarks of the right hand.
- **h5_path**: Path to the hdf5 file.
- **classes, videoName, dataArrs**: Data read from the hdf5 file.
- **bann**: Banned values based on the dataset.
- **train, val**: Flags indicating whether to use training or validation data.

## Methods:

#### `__init__(self, args)`
Initializes the DatasetAnalyzer object.

**Parameters**:

- args: Command line arguments passed to the script.

**Attributes set during initialization**:

- args: Stores the passed command line arguments.
- h5_path: Constructs the path to the hdf5 file based on the args.
- classes, videoName, dataArrs: Reads data from the hdf5 file.
- bann: Calls get_bann_values to get banned values for the dataset specified in args.
- train, val: Flags from args indicating whether to use training or validation data.




#### `get_bann_values(dataset)`

Returns banned values for the given dataset. Banned values are specific landmark labels that are excluded from the analysis.

#### `generate_reduced()`

Generates a reduced dataset after applying filters. Filters include removing specific landmarks or videos based on criteria like minimum instances or banned classes.

#### `selection(dataset, arr, videoNames, classes)`

Selects and returns a filtered subset of the data based on criteria specific to the dataset.

#### `log_data()`

Logs various statistics about the dataset to a file. This includes details like the number of videos, frames, and details about the reduced dataset.

#### `consecutive_stats()`

Calculates statistics related to consecutive missing frames in the dataset.

#### `get_non_reduced_instances()`

Exports the names of videos and their classes that do not have any missing landmarks.

#### `save_data()`

Saves the processed and reduced dataset to an HDF5 file.

#### `kruskal_wallis()`

Performs the Kruskal-Wallis test on the data. This is a non-parametric method used to test the equality of population medians among groups.

#### `plot_data()`

Generates and saves visualizations about the dataset distribution.

#### `execute()`

The main function that orchestrates the entire flow of the analysis. It calls the other methods in a sequence to read the data, process it, analyze it, and optionally save the results and visualizations.


### How to Execute:

1. Ensure both `landmark_dataset_analysis.py` and `run_landmark_analysis.sh` are in the same directory.
2. Grant execute permissions to the `.sh` file:
3. Run the script: `sh run_landmark_analysis.sh`
4. During the execution, the `.sh` script will prompt you for:

    - **Dataset Names List**: Enter the names of datasets you wish to analyze, separated by spaces.
    - **Additional Parameters**: You can enter any additional parameters (like `--save` or `--plot`, etc) to be passed to the `landmark_dataset_analysis.py` script, they will be showed at the beggining in the command line showing all the additional parameters you can use.

    For example:
    
        ```bash
        $ Available parameters for landmark_dataset_analysis.py:
        $ --train: Train Flag
        $ --val: Validation Flag
        $ --plot: Plot Flag
        $ --save: Save Flag
        $ --consecutive_stats: Consecutive Stats Flag
        $ --export_non_reduced: Flag for Export classes and videonames without missing values

        $ Enter the list of dataset names (separated by spaces):
        > AEC PUCP_PSL_DGI305 AUTSL INCLUDE LSA64

        $ Enter any additional parameters for landmark_dataset_analysis.py (e.g. --train --plot):
        > --plot --save --consecutive_stats --export_non_reduced
        ```