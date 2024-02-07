#!/bin/bash

# Display the available parameters and their descriptions to the user
echo "Available parameters for landmark_dataset_analysis.py:"
echo "--train: Train Flag"
echo "--val: Validation Flag"
echo "--plot: Plot Flag"
echo "--save: Save Flag"
echo "--c: Consecutive Stats Flag"
echo "--e: Flag for Export classes and videonames without missing values"
echo ""

# Prompt the user for the list of dataset names
echo "Enter the list of dataset names (separated by spaces):"
read -a DATASETS

# Prompt the user for any additional parameters they want to pass to the Python script
echo "Enter any additional parameters for landmark_dataset_analysis.py (e.g. --train --plot):"
read PARAMETERS

# Loop over the dataset names and run the Python script for each one
for DATASET in "${DATASETS[@]}"
do
    # Set the path to the hdf5 file
    H5_PATH="../Datasets/"

    # Run the Python script with the provided dataset and any additional parameters
    echo "Running dataset ${DATASET}"
    python landmark_dataset_analysis.py --folder "${H5_PATH}" --dataset "${DATASET}" $PARAMETERS

    # echo "Running dataset ${DATASET}"
    # python landmark_dataset_analysis.py --folder "${H5_PATH}" --dataset "${DATASET}" --train $PARAMETERS

    # echo "Running dataset ${DATASET}"
    # python landmark_dataset_analysis.py --folder "${H5_PATH}" --dataset "${DATASET}" --val $PARAMETERS
done
