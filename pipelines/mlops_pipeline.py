"""Main MLOps pipeline"""
import os
from zenml import pipeline
from steps.model_evaluator import evaluate_model
from steps.model_promoter import promote_model
from steps.model_deployer import push_to_huggingface


@pipeline
def sd_mlops_pipeline(
        model_path: str,
        model_name: str = "sd-lora-nigerian-adire",
        hf_repo_id: str = "AfroLogicInsect/sd-lora-nigerian-adire"
):
    """Complete MLOps pipeline for SD 1.5"""

    # Fetch HF Token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")


    # Test prompts
    test_prompts = [
        "a nigerian_adire_style painting of Lagos cityscape at sunset",
        "a nigerian_adire_style portrait of a woman",
        "a nigerian_adire_style pattern with geometric shapes",
        "a nigerian_adire_style landscape scene"
    ]

    # Step 1: Evaluate
    metrics = evaluate_model(
        model_path=model_path,
        test_prompts=test_prompts
    )

    # Step 2: Promotion logic
    promotion = promote_model(
        evaluation_metrics=metrics,
        model_name=model_name,
        quality_threshold=0.75
    )

    # Step 3: Deploy to HuggingFace
    hf_url = push_to_huggingface(
        model_path=model_path,
        promotion_result=promotion,
        repo_id=hf_repo_id,
        token=hf_token  # Pass token explicitly
    )

    return hf_url

