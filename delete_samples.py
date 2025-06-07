import os
import glob
from datetime import datetime
from send2trash import send2trash

def get_files_to_delete(directory):
    """Recursively find all target files to delete"""
    # Find sample_*.png files
    sample_pattern = os.path.join(directory, '**', 'sample_*.png')
    sample_files = glob.glob(sample_pattern, recursive=True)
    
    # Find desktop.ini files
    desktop_pattern = os.path.join(directory, '**', 'desktop.ini')
    desktop_files = glob.glob(desktop_pattern, recursive=True)
    
    return sample_files + desktop_files

def chunk_list(lst, chunk_size):
    """Split list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def cleanup_empty_dirs(directory):
    """Recursively remove empty directories"""
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                # Check if directory is empty
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")

def main():
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"deleted_files_{timestamp}.txt"
    
    # Get all matching files
    all_files = get_files_to_delete('.')
    
    if not all_files:
        print("No files found to delete.")
        return

    print(f"Found {len(all_files)} files to delete")
    
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
            
            if choice == '1':                # Delete current batch
                with open(log_file, 'a') as log:
                    for file in chunk:
                        try:
                            send2trash(file)
                            deletion_msg = f"Moved to Recycle Bin: {file}"
                            print(deletion_msg)
                            log.write(deletion_msg + "\n")
                        except Exception as e:
                            error_msg = f"Error deleting {file}: {e}"
                            print(error_msg)
                            log.write(error_msg + "\n")
            elif choice == '2':
                print("Exiting without deleting files.")
                return
            elif choice == '3':                # Delete current and all remaining batches
                remaining_files = [f for chunk in file_chunks[i:] for f in chunk]
                with open(log_file, 'a') as log:
                    for file in remaining_files:
                        try:
                            send2trash(file)
                            deletion_msg = f"Moved to Recycle Bin: {file}"
                            print(deletion_msg)
                            log.write(deletion_msg + "\n")
                        except Exception as e:
                            error_msg = f"Error deleting {file}: {e}"
                            print(error_msg)
                            log.write(error_msg + "\n")
                # Cleanup empty directories after deleting all files
                cleanup_empty_dirs('.')
                return
            else:
                print("Invalid choice. Exiting without deleting files.")
                return
        else:
            # Last batch
            choice = input("\nThis is the final batch. Delete these files? (y/n): ").strip().lower()
            if choice == 'y':
                with open(log_file, 'a') as log:
                    for file in chunk:
                        try:
                            send2trash(file)
                            deletion_msg = f"Moved to Recycle Bin: {file}"
                            print(deletion_msg)
                            log.write(deletion_msg + "\n")
                        except Exception as e:
                            error_msg = f"Error deleting {file}: {e}"
                            print(error_msg)
                            log.write(error_msg + "\n")
                # Cleanup empty directories after deleting all files
                cleanup_empty_dirs('.')
            else:
                print("Skipping deletion of final batch.")

if __name__ == "__main__":
    main()
