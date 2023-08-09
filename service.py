import time

import torch
from diffusers import DiffusionPipeline
from octoai.service import Service
from octoai.types import Image, Text


class DiffusionOctoService(Service):
    """An OctoAI service extends octoai.service.Service."""
    def __init__(self):
        self.pipe = None

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        start_time = time.time()
        print(f'Model loaded in {(time.time() - start_time) * 1000:.2f}ms')
        DiffusionPipeline.download("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
        pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
        if torch.cuda.is_available():
            self.pipe = pipe.to('cuda')

    def infer(self, prompt: Text) -> Image:
        """Run a single prediction on the model"""
        prompt = prompt.text
        print("Prompt:", prompt)
        print(f"Running on '{torch.cuda.get_device_name()}'")
        start_time = time.time()
        image = self.pipe(prompt, num_inference_steps=30).images[0]
        print(f"Inference done in {(time.time() - start_time) * 1000:.2f}ms")
        return Image.from_pil(image)
