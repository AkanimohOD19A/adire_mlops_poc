"""HuggingFace deployment"""
from zenml import step
from typing import Dict
from huggingface_hub import HfApi, create_repo
import os


@step
def push_to_huggingface(
        model_path: str,
        promotion_result: Dict,
        repo_id: str,
        token: str = None
) -> str:
    """Push to HuggingFace if promoted"""

    if not promotion_result["promoted"]:
        print("Skipping HF push - model not promoted")
        return "skipped"

    if token is None:
        token = os.getenv("HF_TOKEN")

    print(f"Pushing to HuggingFace: {repo_id}")

    try:
        api = HfApi(token=token)

        # Create repo
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

        # Upload weights
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message="Upload SD 1.5 LoRA weights - Nigerian Adire style"
        )

        print(f"✓ Pushed to https://huggingface.co/{repo_id}")
        return f"https://huggingface.co/{repo_id}"

    except Exception as e:
        print(f"✗ HF push failed: {e}")
        return f"failed: {e}"