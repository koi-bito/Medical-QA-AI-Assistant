from huggingface_hub import HfApi
import os

HF_USERNAME = "koi-bito" 

api = HfApi()
repo_id = f"{HF_USERNAME}/medical-qa-vectorstore"

print(f"Creating HuggingFace dataset repository: {repo_id}...")
api.create_repo(
    repo_id=repo_id, 
    repo_type="dataset", 
    private=False, 
    exist_ok=True
)

print("Uploading chroma_db directory (this might take a minute)...")
api.upload_folder(
    folder_path="data/chroma_db",
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="chroma_db"
)
print("Vector store uploaded successfully!")
