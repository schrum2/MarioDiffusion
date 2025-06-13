import os
import glob
from datetime import datetime
from send2trash import send2trash
import argparse

def get_files_to_delete(directory):
    """Generator that yields files to delete as they are found"""
    # Find sample_*.png files
    sample_pattern = os.path.join(directory, '**', 'sample_*.png')
    for file in glob.iglob(sample_pattern, recursive=True):
        yield file
    
    # Find desktop.ini files
    desktop_pattern = os.path.join(directory, '**', 'desktop.ini')
    for file in glob.iglob(desktop_pattern, recursive=True):
        yield file

    # Find early_stop_state.json files
    early_stop_pattern = os.path.join(directory, '**', 'early_stop_state.json')
    for file in glob.iglob(early_stop_pattern, recursive=True):
        yield file

    # Find lr_scheduler.pt files
    lr_scheduler_pattern = os.path.join(directory, '**', 'lr_scheduler.pt')
    for file in glob.iglob(lr_scheduler_pattern, recursive=True):
        yield file

    # Find optimizer.pt files
    optimizer_pattern = os.path.join(directory, '**', 'optimizer.pt')
    for file in glob.iglob(optimizer_pattern, recursive=True):
        yield file

def delete_file(file_path, permanent=False):
    """Delete a file, either permanently or to recycle bin"""
    try:
        if permanent:
            os.remove(file_path)
            return f"Permanently deleted: {file_path}"
        else:
            send2trash(file_path)
            return f"Moved to Recycle Bin: {file_path}"
    except Exception as e:
        return f"Error deleting {file_path}: {e}"

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Delete sample files and empty directories")
    parser.add_argument("--permanent", action="store_true", 
                       help="Permanently delete files instead of moving to recycle bin")
    args = parser.parse_args()

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"deleted_files_{timestamp}.txt"
    
    # Process files as they are found
    file_generator = get_files_to_delete('.')
    chunk = []
    chunk_size = 20
    batch_num = 1
    last_batch = False
    
    delete_mode = "permanently" if args.permanent else "to Recycle Bin"
    print("Searching for files to delete...")
    print(f"Files will be deleted {delete_mode} in batches of {chunk_size}")
    
    try:
        while True:
            # Fill the current chunk
            while len(chunk) < chunk_size:
                try:
                    file = next(file_generator)
                    chunk.append(file)
                except StopIteration:
                    if not chunk:  # No more files found and chunk is empty
                        if batch_num == 1:  # No files were found at all
                            print("No files found to delete.")
                        return
                    last_batch = True
                    break  # Process the final partial chunk
            
            print(f"\nBatch {batch_num}:")
            print("Files to be deleted:")
            for file in chunk:
                print(f"  {file}")
            
            if not last_batch:
                choice = input("\nWhat would you like to do?\n"
                             "1) Delete these files and show next batch\n"
                             "2) Exit without deleting\n"
                             "3) Delete all remaining files without asking\n"
                             "Enter choice (1/2/3): ").strip()
                
                if choice == '1':
                    with open(log_file, 'a') as log:
                        for file in chunk:
                            deletion_msg = delete_file(file, args.permanent)
                            print(deletion_msg)
                            log.write(deletion_msg + "\n")
                    chunk = []  # Clear the chunk for next batch
                    batch_num += 1
                elif choice == '2':
                    print("Exiting without deleting files.")
                    return
                elif choice == '3':
                    # Open log file once for all deletions
                    with open(log_file, 'a') as log:
                        # Delete current batch
                        for file in chunk:
                            deletion_msg = delete_file(file, args.permanent)
                            print(deletion_msg)
                            log.write(deletion_msg + "\n")
                        
                        # Delete all remaining files as they are found
                        for file in file_generator:
                            deletion_msg = delete_file(file, args.permanent)
                            print(deletion_msg)
                            log.write(deletion_msg + "\n")
                    
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
                            deletion_msg = delete_file(file, args.permanent)
                            print(deletion_msg)
                            log.write(deletion_msg + "\n")
                    # Cleanup empty directories after deleting all files
                    cleanup_empty_dirs('.')
                return
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return

if __name__ == "__main__":
    main()
