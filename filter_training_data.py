import os
import json

def delete_unlisted_pngs(metadata_file, directory):
    # Read the metadata file and extract the listed file names
    listed_files = set()
    with open(metadata_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            listed_files.add(data["file_name"])
    
    # Iterate over PNG files in the directory
    for file in os.listdir(directory):
        if file.endswith(".png") and file not in listed_files:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Deleted: {file}")

# Usage
delete_unlisted_pngs("SMB1\\metadata.jsonl", "SMB1")