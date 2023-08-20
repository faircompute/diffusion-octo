import time

import torch
from diffusers import DiffusionPipeline
from octoai.service import Service
from octoai.types import Image, Text
from threading import Lock, Thread
import time
import os


class AutoExit(Thread):
    def __init__(self):
        super().__init__()
        self.last_active = time.time()
        self.max_active_time = 60*10
        self.mutex = Lock()

    def remaining_active_time(self):
        self.mutex.acquire()
        active_time = time.time() - self.last_active
        time_left = self.max_active_time - active_time
        self.mutex.release()
        return time_left

    def update_activity(self):
        self.mutex.acquire()
        self.last_active = time.time()
        self.mutex.release()

    def run(self):
        while True:
            time_left = self.remaining_active_time()
            if time_left >= 0:
                time.sleep(time_left + 10)
            else:
                os._exit(0)


class DiffusionOctoService(Service):
    # It takes a lot of time to compile the model on the first run
    # The speed-up as tested on 4070Ti is insignificant
    run_compile = False

    """An OctoAI service extends octoai.service.Service."""
    def __init__(self):
        super().__init__()
        self.pipe = None

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        start_time = time.time()
        print(f'Model loaded in {(time.time() - start_time) * 1000:.2f}ms')
        DiffusionPipeline.download("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
        self.pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
        if torch.cuda.is_available():
            self.pipe = self.pipe.to('cuda')
            cuda_capability = torch.cuda.get_device_capability(device=None)
            if self.run_compile and cuda_capability[0] >= 7:
                self.pipe.unet.to(memory_format=torch.channels_last)
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

        self.auto_exit = AutoExit()
        self.auto_exit.start()
        self.auto_exit.update_activity()

    def infer(self, prompt: Text) -> Image:
        """Run a single prediction on the model"""
        self.auto_exit.update_activity()
        prompt = prompt.text
        print("Prompt:", prompt)
        print(f"Running on '{torch.cuda.get_device_name()}'")
        start_time = time.time()
        image = self.pipe(prompt, num_inference_steps=30).images[0]
        print(f"Inference done in {(time.time() - start_time) * 1000:.2f}ms")
        return Image.from_pil(image)
