# python - << EOF
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

image = pipe("a cinematic african village at sunrise").images[0]
image.save("test.png")

print("Stable Diffusion CPU OK")
# EOF