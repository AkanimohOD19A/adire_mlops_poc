if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "deploy"], required=True)
    parser.add_argument("--data_path", default="./data/training_images")
    parser.add_argument("--hf_repo", default="amahadaniel/sd-lora-african-art")
    args = parser.parse_args()
    
    if args.mode == "train":
        # Initialize ZenML
        from zenml.client import Client
        client = Client()
        
        # Run pipeline
        pipeline_instance = mlops_image_generation_pipeline(
            data_path=args.data_path,
            concept_name="african_art",
            hf_repo_id=args.hf_repo
        )
        
        print("✓ Pipeline completed!")
        print(f"Check ZenML dashboard: zenml up")
        print(f"Check MLflow dashboard: mlflow ui")
        
    elif args.mode == "deploy":
        # Launch inference app
        app = MonitoredInferenceApp(
            model_id="runwayml/stable-diffusion-v1-5",
            lora_weights_path="./models/lora_african_art"
        )
        app.launch()


