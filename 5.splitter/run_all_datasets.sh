# Prompt the user for the list of dataset names
echo "Enter the list of dataset names (separated by spaces):"
read -a DATASETS
# Set the folder path
FOLDER_PATH="../Datasets"

# Loop over the dataset names and run the appropriate script for each one
for DATASET in "${DATASETS[@]}"
do
    # Set the path to the hdf5 file
    H5_PATH="${FOLDER_PATH}/${DATASET}"

    # Run the appropriate script based on the mode of operation
    echo "Running dataset ${DATASET}"
    if [ "${DATASET}" == "LSA64" ]; then
        CSV_FILENAME="${DATASET}_meaning.csv"
        python split_by_hdf5.py --folder "${FOLDER_PATH}" --dataset "${DATASET}" --csv_filename "${CSV_FILENAME}"
    else
        python split_by_hdf5.py --folder "${FOLDER_PATH}" --dataset "${DATASET}"
    fi
done
