import os
import json

# Ask the user for the key to remove
key_to_remove = input("Enter the key to remove from each JSON file: ")

# Get all JSON files in the current directory
json_files = [f for f in os.listdir('.') if f.endswith('.json')]

for filename in json_files:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure the JSON is a list of dicts
        if isinstance(data, list):
            new_data = [{k: v for k, v in d.items() if k != key_to_remove} for d in data]
        else:
            print(f"Skipping {filename}: not a list of dicts.")
            continue

        # Overwrite the original file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2)
        
        print(f"Processed {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
