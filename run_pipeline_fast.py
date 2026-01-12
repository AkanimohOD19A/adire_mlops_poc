"""Fast pipeline - skip evaluation for testing"""
import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow

# Load environment
load_dotenv()

from steps.model_deployer import push_to_huggingface


def main():
    print("=" * 60)
    print("Fast Deploy Pipeline - Skipping Evaluation")
    print("=" * 60)

    model_path = "models/lora_weights"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Mock promotion result (as if it passed)
    mock_promotion = {
        "promoted": True,
        "reason": "Manual deployment test"
    }

    # Get token
    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment")

    print(f"\nDeploying to HuggingFace...")

    # Deploy directly without ZenML pipeline
    from huggingface_hub import HfApi, create_repo

    repo_id = "AfroLogicInsect/sd-lora-nigerian-adire"

    try:
        api = HfApi(token=hf_token)
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=hf_token)

        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message="Upload SD 1.5 LoRA weights - Nigerian Adire style"
        )

        print(f"\n✓ Successfully deployed to: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")


if __name__ == "__main__":
    main()