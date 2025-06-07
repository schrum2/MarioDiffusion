import os
import glob

def get_sample_files(directory):
    """Recursively find all files starting with 'sample_' and ending with '.png'"""
    pattern = os.path.join(directory, '**', 'sample_*.png')
    return glob.glob(pattern, recursive=True)

def chunk_list(lst, chunk_size):
    """Split list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    # Get all matching files
    all_files = get_sample_files('.')
    
    if not all_files:
        print("No sample_*.png files found.")
        return

    print(f"Found {len(all_files)} files matching the pattern 'sample_*.png'")
    
    # Process files in chunks of 20 for better readability
    chunk_size = 20
    file_chunks = list(chunk_list(all_files, chunk_size))
    
    for i, chunk in enumerate(file_chunks):
        print(f"\nBatch {i+1} of {len(file_chunks)}:")
        print("Files to be deleted:")
        for file in chunk:
            print(f"  {file}")
            
        if i < len(file_chunks) - 1:
            choice = input("\nWhat would you like to do?\n"
                         "1) Delete these files and show next batch\n"
                         "2) Exit without deleting\n"
                         "3) Delete all remaining files without asking\n"
                         "Enter choice (1/2/3): ").strip()
            
            if choice == '1':
                # Delete current batch
                for file in chunk:
                    try:
                        os.remove(file)
                        print(f"Deleted: {file}")
                    except Exception as e:
                        print(f"Error deleting {file}: {e}")
            elif choice == '2':
                print("Exiting without deleting files.")
                return
            elif choice == '3':
                # Delete current and all remaining batches
                remaining_files = [f for chunk in file_chunks[i:] for f in chunk]
                for file in remaining_files:
                    try:
                        os.remove(file)
                        print(f"Deleted: {file}")
                    except Exception as e:
                        print(f"Error deleting {file}: {e}")
                return
            else:
                print("Invalid choice. Exiting without deleting files.")
                return
        else:
            # Last batch
            choice = input("\nThis is the final batch. Delete these files? (y/n): ").strip().lower()
            if choice == 'y':
                for file in chunk:
                    try:
                        os.remove(file)
                        print(f"Deleted: {file}")
                    except Exception as e:
                        print(f"Error deleting {file}: {e}")
            else:
                print("Skipping deletion of final batch.")

if __name__ == "__main__":
    main()
