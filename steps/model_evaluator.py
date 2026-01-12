"""Model evaluation step"""
from zenml import step
from typing import Dict, List
import mlflow
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime

@step
def evaluate_model(
        model_path: str,
        test_prompts: List[str],
        base_model: str = "runwayml/stable-diffusion-v1-5"
) -> Dict:
    """Evaluate trained SD 1.5 model"""

    print("Loading model for evaluation...")

    # Check device and set appropriate dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device} with dtype: {dtype}")

    # Load Model
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype = dtype #torch.float16
    )

    # Load LoRA weights
    pipe.load_lora_weights(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Evaluate with reduced steps for CPU
    results = []
    total_time = 0

    # Use fewer steps on CPU to speed up
    num_steps = 20 if device == "cpu" else 50

    for prompt in test_prompts:
        start = datetime.now()

        try:
            image = pipe(
                prompt,
                num_inference_steps=num_steps,
                guidance_scale=7.5
            ).images[0]

            generation_time = (datetime.now() - start).total_seconds()
            total_time += generation_time

            # Placeholder quality
            quality_score = 0.82

            results.append({
                "prompt": prompt,
                "generation_time": generation_time,
                "quality_score": quality_score,
                "success": True
            })

            print(f"✓ Generated: {prompt[:50]}... ({generation_time:.2f}s)")

        except Exception as e:
            print(f" Failed: {prompt[:50]}... - {e}")
            results.append({
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })

    # Calc. metrics
    successful = [r for r in results if r.get("success")]

    metrics = {
        "num_tests": len(test_prompts),
        "num_successful": len(successful),
        "success_rate": len(successful) / len(test_prompts) if test_prompts else 0,
        "avg_generation_time": total_time / len(successful) if successful else 0,
        "avg_quality_score": sum(r["quality_score"] for r in successful) / len(successful) if successful else 0,
        "device": device,
        "num_inference_steps": num_steps
    }

    # Log to mlflow
    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    mlflow.log_param("device", device)

    print(f"\n✓ Evaluation complete: {metrics}")

    return metrics