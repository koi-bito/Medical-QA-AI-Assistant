import time
import urllib3.util.timeout
from huggingface_hub import snapshot_download

# Force connect and read timeouts at the urllib3 socket layer to prevent hanging during streaming reads
original_init = urllib3.util.timeout.Timeout.__init__

def new_init(self, total=None, connect=None, read=None):
    # If read/connect timeouts are None, default, or not set, force them to 15 seconds
    if read is None or read is urllib3.util.timeout.SUBSECTION_DEFAULTS:
        read = 15.0
    if connect is None or connect is urllib3.util.timeout.SUBSECTION_DEFAULTS:
        connect = 15.0
    original_init(self, total=total, connect=connect, read=read)

urllib3.util.timeout.Timeout.__init__ = new_init

def main():
    repo_id = "microsoft/Phi-3-mini-4k-instruct"
    print(f"Starting resilient download of {repo_id}...")
    print("Timeouts are forced at 10s (connect) and 30s (read) to prevent hangs.")
    
    max_retries = 30
    retry_delay = 5
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[Attempt {attempt}/{max_retries}] Downloading model files...")
            snapshot_download(
                repo_id=repo_id,
                max_workers=2
            )
            print("\n✓ Download completed successfully!")
            break
        except Exception as e:
            print(f"\n[Warning] Download interrupted: {e}")
            if attempt < max_retries:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
            else:
                print("\n[Error] Max retries reached. Download failed.")

if __name__ == "__main__":
    main()
