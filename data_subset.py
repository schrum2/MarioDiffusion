import os
import shutil
import random
import argparse

def copy_random_files(source_dir, dest_dir, num_files=500):
    """
    Randomly selects files from a source directory and copies them to a destination directory.

    Args:
        source_dir: The path to the source directory.
        dest_dir: The path to the destination directory.
        num_files: The number of files to copy (default: 500).
    """

    try:
        # Get a list of all files in the source directory
        files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

        if not files:
            print(f"Error: No files found in the source directory: {source_dir}")
            return

        # Ensure we don't try to sample more files than exist
        num_files_to_copy = min(num_files, len(files)) #Handles edge case where less than 500 files exist

        # Randomly select files
        selected_files = random.sample(files, num_files_to_copy)

        # Create the destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)  # exist_ok prevents error if dir exists

        # Copy the selected files
        for file_name in selected_files:
            source_path = os.path.join(source_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.copy2(source_path, dest_path) #copy2 preserves metadata
            print(f"Copied: {file_name}")

        print(f"Successfully copied {num_files_to_copy} files to {dest_dir}")

    except FileNotFoundError:
        print(f"Error: Source directory not found: {source_dir}")
    except Exception as e:  # Catch other potential errors (e.g., permissions)
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy random files from one directory to another.")
    parser.add_argument("source_dir", help="The source directory.")
    parser.add_argument("dest_dir", help="The destination directory.")
    parser.add_argument("-n", "--num_files", type=int, default=500, help="The number of files to copy (default: 500).")

    args = parser.parse_args()

    copy_random_files(args.source_dir, args.dest_dir, args.num_files)