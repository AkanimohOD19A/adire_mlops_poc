"""Run the MLOps pipeline"""
import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow
from pipelines.mlops_pipeline import sd_mlops_pipeline

# Load environment
try:
    load_dotenv()
except UnicodeError:
    print("Warning: .env file has encoding issues. Please check the file.")
    print("Continuing without .env file...")
except FileNotFoundError:
    print("No .env file found. Using system environment variables.")



def main():
    print("=" * 60)
    print("SD 1.5 MLOps Pipeline - Nigerian Adire Style")
    print("=" * 60)

    # Configuration
    model_path = "models/lora_weights"

    # Verify path exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Set MLflow tracking
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("sd-nigerian-adire")

    print(f"\n Model path: {model_path}")
    print()

    # Run pipeline
    with mlflow.start_run(run_name="full_pipeline"):
        result = sd_mlops_pipeline(
            model_path=model_path,
            hf_repo_id="AfroLogicInsect/sd-lora-nigerian-adire"  # Change this!
        )

    print("\n" + "=" * 60)
    print(f"✓ Pipeline complete!")
    print(f"✓ Result: {result}")
    print("=" * 60)

    print("\n View results:")
    print("  - MLflow UI: mlflow ui --port 5000")
    print("  - ZenML UI: zenml up")


if __name__ == "__main__":
    main()