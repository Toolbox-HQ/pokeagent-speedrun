#!bin/bash

# WLOG we choose the first bucket
bucket=$(.venv/bin/s3cmd ls | head -n 1 | awk '{print $NF}' | sed 's/s3:\/\///; s/\///')

# Define the directory you want to loop through
DATA_DIR="./Data"

# Check if the directory exists
if [ -d "$DATA_DIR" ]; then

    for file in "$DATA_DIR"/*; do
        if [ -f "$file" ]; then
            echo "Uploading file: $file"
            .venv/bin/s3cmd put $file s3://$bucket/pokeagent/emulator-data/
        fi
    done
else
    echo "Error: Directory '$DATA_DIR' not found."
fi