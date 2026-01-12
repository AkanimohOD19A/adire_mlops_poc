"""Gradio inference app for Nigerian Adire Style SD model"""
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import mlflow
from datetime import datetime
from pathlib import Path

# Configuration
MODEL_PATH = "models/lora_weights"  # Local model path
HF_MODEL_ID = "AfroLogicInsect/sd-lora-nigerian-adire"  # HuggingFace model
USE_HF_MODEL = True  # Set to True to load from HuggingFace, False for local

print("🚀 Loading Nigerian Adire Style Generator...")

# Determine device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"📱 Using device: {device} with dtype: {dtype}")

# Load base model
print("📦 Loading base Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)

# Load LoRA weights
if USE_HF_MODEL:
    print(f"🤗 Loading LoRA weights from HuggingFace: {HF_MODEL_ID}")
    try:
        pipe.load_lora_weights(HF_MODEL_ID)
        print("✅ Successfully loaded from HuggingFace!")
    except Exception as e:
        print(f"⚠️ Failed to load from HuggingFace: {e}")
        print(f"🔄 Falling back to local model: {MODEL_PATH}")
        if Path(MODEL_PATH).exists():
            pipe.load_lora_weights(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Local model not found at {MODEL_PATH}")
else:
    print(f"📂 Loading LoRA weights from local: {MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    pipe.load_lora_weights(MODEL_PATH)

pipe = pipe.to(device)

# Setup MLflow tracking
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("sd-inference")

print("✅ Model loaded successfully!")


def generate(prompt, steps, guidance):
    """Generate image from prompt"""
    start = datetime.now()

    try:
        # Generate image
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=int(steps),
                guidance_scale=guidance
            ).images[0]

        duration = (datetime.now() - start).total_seconds()

        # Log to MLflow
        try:
            with mlflow.start_run():
                mlflow.log_params({
                    "prompt": prompt,
                    "steps": steps,
                    "guidance": guidance,
                    "device": device
                })
                mlflow.log_metrics({"duration": duration})
        except Exception as e:
            print(f"MLflow logging failed: {e}")

        return image, f"✅ Generated in {duration:.2f}s on {device}"

    except Exception as e:
        return None, f"❌ Generation failed: {str(e)}"


# Example prompts
examples = [
    ["a nigerian_adire_style painting of Lagos cityscape at sunset", 30, 7.5],
    ["a nigerian_adire_style portrait of a woman wearing traditional attire", 30, 7.5],
    ["a nigerian_adire_style pattern with geometric shapes and indigo colors", 30, 7.5],
    ["a nigerian_adire_style landscape with palm trees and ocean", 30, 7.5],
]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 Nigerian Adire Style Generator
        ### Fine-tuned Stable Diffusion 1.5 with LoRA + Full MLOps Pipeline
        
        Generate beautiful images in the traditional Nigerian Adire textile art style!
        """
    )

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="a nigerian_adire_style painting of...",
                lines=3
            )
            steps = gr.Slider(
                minimum=10,
                maximum=50,
                value=20 if device == "cpu" else 30,
                step=5,
                label="Inference Steps (lower = faster)"
            )
            guidance = gr.Slider(
                minimum=1,
                maximum=15,
                value=7.5,
                step=0.5,
                label="Guidance Scale"
            )
            generate_btn = gr.Button("🎨 Generate Image", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")
            status = gr.Textbox(label="Status")

    gr.Examples(
        examples=examples,
        inputs=[prompt, steps, guidance],
        outputs=[output_image, status],
        fn=generate,
        cache_examples=False,
        label="Example Prompts"
    )

    gr.Markdown(
        """
        ### 💡 Tips:
        - Always include `nigerian_adire_style` in your prompt for best results
        - Lower steps (10-20) for faster generation on CPU
        - Higher guidance scale (7-10) for more style adherence
        - Be patient on CPU - each image takes 2-5 minutes
        """
    )

    generate_btn.click(
        fn=generate,
        inputs=[prompt, steps, guidance],
        outputs=[output_image, status]
    )

if __name__ == "__main__":
    print("\n🌐 Starting Gradio interface...")
    demo.launch(
        share=True,  # Creates public link
        server_name="0.0.0.0",  # Makes it accessible on network
        server_port=7860
    )