# upload_to_hf.py
import os
from pathlib import Path
import modal

# ---- Modal setup ----
app = modal.App("upload-model-to-huggingface")

# Install git-lfs (recommended) and huggingface_hub inside the Modal image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs")
    .run_commands("git lfs install --system || true")
    .pip_install("huggingface_hub>=0.24.0")
)

# Change this to your existing Modal Volume name that contains the model files
VOLUME_NAME = "gpt-oss-models"

# Mount the volume at /models inside the container
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={"/models": volume},
    timeout=60 * 60,   # plenty of time for large uploads
)
def upload_to_hf(
    repo_id: str,
    model_subdir: str = ".",
    repo_private: bool = True,
    repo_type: str = "model",
    commit_message: str = "Add model files",
):
    """
    Uploads files from /models/<model_subdir> (Modal Volume) to Hugging Face Hub.
    """
    from huggingface_hub import HfApi, create_repo, upload_folder

    token = os.environ.get("HF_TOKEN") 
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Add a Modal secret named 'huggingface-token' that sets HF_TOKEN."
        )

    api = HfApi(token=token)

    # Ensure repo exists (no-op if it already does)
    create_repo(
        repo_id=repo_id,
        private=repo_private,
        repo_type=repo_type,
        exist_ok=True,
        token=token,
    )

    # Path inside the mounted volume
    local_folder = Path("/models") / model_subdir
    if not local_folder.exists():
        raise FileNotFoundError(f"Model path not found in volume: {local_folder}")

    # Optional: create a minimal README if none exists
    readme_path = local_folder / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            f"# {repo_id}\n\nModel uploaded from Modal volume `{VOLUME_NAME}`.\n"
        )

    print(f"Uploading folder: {local_folder} -> hf://{repo_type}s/{repo_id}")
    # upload_folder handles large files (LFS) and resumable uploads
    upload_folder(
        folder_path=str(local_folder),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="",  # upload contents to repo root
        commit_message=commit_message,
        # Skip junk; add more patterns if needed
        ignore_patterns=[
            "**/__pycache__/**",
            "**/.ipynb_checkpoints/**",
            "**/.DS_Store",
            "**/*.tmp",
        ],
        token=token,
    )

    # Print final URL
    base = "https://huggingface.co"
    if repo_type != "model":
        base = f"{base}/{repo_type}s"
    print(f"✅ Upload complete: {base}/{repo_id}")

@app.local_entrypoint()
def main(
    repo_id: str,
    model_subdir: str = ".",
    public: bool = False,
    repo_type: str = "model",
):
    """
    CLI entrypoint. Example:
      modal run upload_to_hf.py --repo-id your-user/your-model --model-subdir . --public
    """
    repo_private = not public
    upload_to_hf.remote(
        repo_id=repo_id,
        model_subdir=model_subdir,
        repo_private=repo_private,
        repo_type=repo_type,
        commit_message="Add model files",
    )
