from huggingface_hub import HfApi, whoami

def main():
    # Get current HF username programmatically from your active session
    try:
        user_info = whoami()
        hf_username = user_info['name']
        print(f"Logged in as Hugging Face user: {hf_username}")
    except Exception as e:
        print("Error: Could not retrieve Hugging Face user info. Make sure your HF_TOKEN environment variable is set.")
        print(e)
        return

    api = HfApi()
    repo_id = f"{hf_username}/phi3-medical-lora"

    print(f"Creating repository '{repo_id}' if it doesn't exist...")
    api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
    print(f"Repository ready at: https://huggingface.co/{repo_id}")

    print("Uploading adapter and tokenizer files (excluding checkpoint folders)...")
    api.upload_folder(
        folder_path="models/phi3-medical-lora",
        repo_id=repo_id,
        ignore_patterns=["checkpoint-*"],
    )
    print(f"\n✅ Successfully uploaded! Your model is live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
