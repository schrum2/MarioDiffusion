import os
from huggingface_hub import hf_hub_download

# Helper to get file path (local or HF Hub)
def get_file(filename, load_directory, subfolder):
    # Try local path first
    if os.path.isdir(load_directory):
        model_dir = os.path.join(load_directory, subfolder) if subfolder else load_directory
        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path):
            return file_path
    # Otherwise, try Hugging Face Hub
    repo_id = load_directory
    subpath = f"{subfolder}/{filename}" if subfolder else filename
    return hf_hub_download(repo_id=repo_id, filename=subpath)
