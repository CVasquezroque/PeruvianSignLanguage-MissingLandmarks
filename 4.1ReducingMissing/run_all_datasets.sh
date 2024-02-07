#!/bin/bash

# Prompt the user for the list of dataset names
echo "Enter the list of dataset names (separated by spaces):"
read -a DATASETS

# Prompt the user for the mode of operation
echo "Select mode of operation:"
echo "1. Run all scripts"
echo "2. Run analyzing_distribution.py only"
echo "3. Run pose_estimation_reducing.py only"
read -p "Enter mode number: " MODE

# Prompt the user for the values of the --plot and --save parameters for pose_estimation_reducing.py
if [ "$MODE" -eq 1 ] || [ "$MODE" -eq 3 ]; then
    read -p "Enter value for --plot (True/False): " PLOT
    read -p "Enter value for --save (True/False): " SAVE
    read -p "Enter value for --consecutive_stats (True/False): " CONSECUTIVE_STATS
fi

# Loop over the dataset names and run the appropriate script for each one
for DATASET in "${DATASETS[@]}"
do
    # Set the path to the hdf5 file
    H5_PATH="../Data/${DATASET}"

    # Run the appropriate script based on the mode of operation
    echo "Running dataset ${DATASET}"
    if [ "$MODE" -eq 1 ]; then
        echo "Running script analyzing_distribution.py"
        python analyzing_distribution.py --h5_path "${H5_PATH}" --dataset "${DATASET}"
        
        echo "Running script pose_estimation_reducing.py"
        python pose_estimation_reducing.py --h5_path "${H5_PATH}" --dataset "${DATASET}" --plot "${PLOT}" --save "${SAVE}" --consecutive_stats "${CONSECUTIVE_STATS}"
    elif [ "$MODE" -eq 2 ]; then
        echo "Running script analyzing_distribution.py"
        python analyzing_distribution.py --h5_path "${H5_PATH}" --dataset "${DATASET}"
    elif [ "$MODE" -eq 3 ]; then
        echo "Running script pose_estimation_reducing.py"
        python pose_estimation_reducing.py --h5_path "${H5_PATH}" --dataset "${DATASET}" --plot "${PLOT}" --save "${SAVE}" --consecutive_stats "${CONSECUTIVE_STATS}"
    else
        echo "Invalid mode number. Exiting."
        exit 1
    fi
done