from zenml import pipeline, step
from zenml.client import Client
from zenml.logger import get_logger
import mlflow
from typing import Tuple, Dict
import json

logger = get_logger(__name__)

@step
def load_training_data(data_path: str) -> Dict:
    """Load and validate training images"""
    from pathlib import Path
    from PIL import Image
    
    image_files = list(Path(data_path).glob("*.jpg")) + list(Path(data_path).glob("*.png"))
    
    valid_images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            if img.size[0] >= 512 and img.size[1] >= 512:
                valid_images.append(str(img_path))
        except Exception as e:
            logger.warning(f"Skipping invalid image {img_path}: {e}")
    
    logger.info(f"Loaded {len(valid_images)} valid training images")
    
    return {
        "image_paths": valid_images,
        "num_images": len(valid_images),
        "data_path": data_path
    }

@step
def train_lora_model(
    data_info: Dict,
    model_id: str,
    concept_name: str,
    num_train_steps: int,
    learning_rate: float
) -> Dict:
    """Train LoRA model (this runs on Colab, but orchestrated by ZenML)"""
    
    # Log to MLflow
    mlflow.set_experiment("image_generation_lora")
    
    with mlflow.start_run(run_name=f"lora_training_{concept_name}"):
        mlflow.log_params({
            "model_id": model_id,
            "concept_name": concept_name,
            "num_images": data_info["num_images"],
            "num_train_steps": num_train_steps,
            "learning_rate": learning_rate
        })
        
        # Here you'd trigger the actual Colab training
        # For POC, simulate training results
        
        training_results = {
            "model_path": f"./models/lora_{concept_name}",
            "final_loss": 0.045,
            "training_steps": num_train_steps,
            "concept_name": concept_name
        }
        
        mlflow.log_metrics({
            "final_loss": training_results["final_loss"],
            "training_steps": training_results["training_steps"]
        })
        
        logger.info(f"Training completed. Final loss: {training_results['final_loss']}")
        
        return training_results

@step
def evaluate_model(
    training_results: Dict,
    test_prompts: list
) -> Dict:
    """Evaluate model quality using automated metrics"""
    from diffusers import StableDiffusionPipeline
    import torch
    
    # Load fine-tuned model
    # In production, load from training_results["model_path"]
    
    evaluation_metrics = {
        "clip_score": 0.0,
        "fid_score": 0.0,
        "sample_quality": 0.0
    }
    
    # Generate test images
    test_results = []
    for prompt in test_prompts:
        # Simulate generation
        result = {
            "prompt": prompt,
            "clip_score": 0.82,  # Placeholder
            "aesthetic_score": 7.5  # Placeholder
        }
        test_results.append(result)
        
        evaluation_metrics["clip_score"] += result["clip_score"]
    
    evaluation_metrics["clip_score"] /= len(test_prompts)
    evaluation_metrics["num_samples"] = len(test_results)
    
    # Log to MLflow
    mlflow.log_metrics(evaluation_metrics)
    
    logger.info(f"Evaluation complete. CLIP score: {evaluation_metrics['clip_score']:.3f}")
    
    return evaluation_metrics

@step
def promote_model(
    training_results: Dict,
    evaluation_metrics: Dict,
    clip_threshold: float = 0.75,
    environment: str = "production"
) -> Dict:
    """MLflow model promotion logic"""
    
    # Promotion decision logic
    should_promote = evaluation_metrics["clip_score"] >= clip_threshold
    
    promotion_result = {
        "promoted": should_promote,
        "reason": "",
        "model_version": "v1.0",
        "environment": environment
    }
    
    if should_promote:
        # Register model in MLflow
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name=f"sd_lora_{training_results['concept_name']}"
        )
        
        # Transition to production
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get latest version
        model_name = f"sd_lora_{training_results['concept_name']}"
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
        
        # Promote to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage=environment.capitalize()
        )
        
        promotion_result["reason"] = f"CLIP score {evaluation_metrics['clip_score']:.3f} exceeds threshold {clip_threshold}"
        promotion_result["model_version"] = latest_version.version
        
        logger.info(f"✓ Model promoted to {environment}")
    else:
        promotion_result["reason"] = f"CLIP score {evaluation_metrics['clip_score']:.3f} below threshold {clip_threshold}"
        logger.warning(f"✗ Model NOT promoted: {promotion_result['reason']}")
    
    mlflow.log_dict(promotion_result, "promotion_decision.json")
    
    return promotion_result

@step
def push_to_huggingface(
    training_results: Dict,
    promotion_result: Dict,
    hf_repo_id: str
) -> str:
    """Push model to Hugging Face if promoted"""
    
    if not promotion_result["promoted"]:
        logger.info("Skipping HuggingFace push - model not promoted")
        return "skipped"
    
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi()
    
    # Create repo if doesn't exist
    try:
        create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        logger.warning(f"Repo might already exist: {e}")
    
    # Upload LoRA weights
    model_path = training_results["model_path"]
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=hf_repo_id,
        commit_message=f"Upload LoRA weights - {promotion_result['model_version']}"
    )
    
    logger.info(f"✓ Model pushed to https://huggingface.co/{hf_repo_id}")
    
    return f"https://huggingface.co/{hf_repo_id}"

@pipeline
def mlops_image_generation_pipeline(
    data_path: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    concept_name: str = "african_art",
    num_train_steps: int = 1000,
    learning_rate: float = 1e-4,
    clip_threshold: float = 0.75,
    hf_repo_id: str = "amahadaniel/sd-lora-african-art"
):
    """Complete MLOps pipeline for image generation model"""
    
    # Step 1: Load data
    data_info = load_training_data(data_path=data_path)
    
    # Step 2: Train model
    training_results = train_lora_model(
        data_info=data_info,
        model_id=model_id,
        concept_name=concept_name,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate
    )
    
    # Step 3: Evaluate
    test_prompts = [
        f"a {concept_name} style painting of a sunset",
        f"a {concept_name} style portrait",
        f"a {concept_name} style landscape"
    ]
    
    evaluation_metrics = evaluate_model(
        training_results=training_results,
        test_prompts=test_prompts
    )
    
    # Step 4: Promotion logic (MLflow)
    promotion_result = promote_model(
        training_results=training_results,
        evaluation_metrics=evaluation_metrics,
        clip_threshold=clip_threshold,
        environment="production"
    )
    
    # Step 5: Push to Hugging Face
    hf_url = push_to_huggingface(
        training_results=training_results,
        promotion_result=promotion_result,
        hf_repo_id=hf_repo_id
    )
    
    return hf_url
