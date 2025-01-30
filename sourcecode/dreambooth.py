import os
import json
import boto3
import requests
import shutil
import torch
import gradio as gr
from glob import glob
from tqdm import tqdm
from botocore import UNSIGNED
from botocore.client import Config
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from huggingface_hub import login
from diffusers import StableDiffusionPipeline, DDIMScheduler


class DreamBoothTrainer:
    def __init__(self, huggingface_token, model_name, output_dir, concepts_list):
        self.huggingface_token = huggingface_token
        self.model_name = model_name
        self.output_dir = output_dir
        self.concepts_list = concepts_list
        self.weights_dir = ""

        self._setup_directories()
        self._download_best_params()
        login(token=self.huggingface_token)
        self._save_concepts_list()

    def _setup_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for concept in self.concepts_list:
            os.makedirs(concept["instance_data_dir"], exist_ok=True)

    def _download_best_params(self):
        local_dir = os.path.join(os.getcwd(), 'best_checkpoints', 'dreambooth')
        os.makedirs(local_dir, exist_ok=True)
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket='data255-cartoongan', Prefix='4000/')

        for page in tqdm(pages):
            for obj in page.get('Contents', []):
                key = obj['Key']
                local_file_path = os.path.join(local_dir, os.path.relpath(key, '4000/'))
                local_file_dir = os.path.dirname(local_file_path)
                
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)
                s3.download_file('data255-cartoongan', key, local_file_path)
                print(f"Downloaded {key} to {local_file_path}")

    def _save_concepts_list(self):
        with open("concepts_list.json", "w") as f:
            json.dump(self.concepts_list, f, indent=4)

    def download_scripts(self):
        scripts = [
            "https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py",
            "https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py"
        ]
        for script in scripts:
            self._download_file(script)

    def _download_file(self, url):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        filename = url.split("/")[-1]
        with open(filename, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                file.write(chunk)
        print(f"Downloaded {filename}")

    def train_model(self):
        command = """
        python train_dreambooth.py \
        --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
        --pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse \
        --output_dir=checkpoints/dreambooth \
        --revision=main \
        --with_prior_preservation \
        --prior_loss_weight=1.0 \
        --seed=2024 \
        --resolution=512 \
        --train_batch_size=1 \
        --train_text_encoder \
        --mixed_precision=fp16 \
        --gradient_accumulation_steps=1 \
        --learning_rate=1e-6 \
        --lr_scheduler=constant \
        --lr_warmup_steps=0 \
        --num_class_images=50 \
        --sample_batch_size=4 \
        --max_train_steps=1 \
        --save_interval=1 \
        --save_sample_prompt='photo of deva' \
        --concepts_list=concepts_list.json
        """
        print("Training Starting!")
        os.system(command)
        print("Training Finished!")

    def generate_images(self, prompt, negative_prompt, num_samples, guidance_scale, num_inference_steps, height=512, width=512):
        pipe = StableDiffusionPipeline.from_pretrained(self.model_name, safety_checker=None).to("cuda" if torch.cuda.is_available() else 'cpu')
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        
        g_cuda = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(2024)
        
        with torch.autocast("cuda" if torch.cuda.is_available() else 'cpu'), torch.inference_mode():
            images = pipe(prompt, height=height, width=width, negative_prompt=negative_prompt, num_images_per_prompt=num_samples,
                          num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=g_cuda).images
        return images

    def save_grid(self, folder_path):
        folders = sorted([f for f in os.listdir(folder_path) if f != "0"], key=lambda x: int(x))
        row = len(folders)
        col = len(os.listdir(os.path.join(folder_path, folders[0], "samples")))
        scale = 4
        fig, axes = plt.subplots(row, col, figsize=(col * scale, row * scale), gridspec_kw={'hspace': 0, 'wspace': 0})

        for i, folder in enumerate(folders):
            image_folder = os.path.join(folder_path, folder, "samples")
            images = [f for f in os.listdir(image_folder)]
            for j, image in enumerate(images):
                curr_axes = axes[i, j] if row > 1 else axes[j]
                curr_axes.imshow(mpimg.imread(os.path.join(image_folder, image)), cmap='gray')
                curr_axes.axis('off')

        plt.tight_layout()
        plt.savefig('grid.png', dpi=72)

    def convert_to_ckpt(self, fp16=True):
        ckpt_path = os.path.join(self.output_dir, "model.ckpt")
        half_arg = "--half" if fp16 else ""
        command = f"python convert_diffusers_to_original_stable_diffusion.py --model_path {self.output_dir} --checkpoint_path {ckpt_path} {half_arg}"
        os.system(command)
        print(f"[*] Converted ckpt saved at {ckpt_path}")

    def launch_gradio_demo(self):
        def inference(prompt, negative_prompt, num_samples, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
            return self.generate_images(prompt, negative_prompt, num_samples, guidance_scale, num_inference_steps, height, width)

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", value="photo of deva playing with a dog")
                    negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                    run = gr.Button(value="Generate")
                    with gr.Row():
                        num_samples = gr.Number(label="Number of Samples", value=4)
                        guidance_scale = gr.Number(label="Guidance Scale", value=7.5)
                    with gr.Row():
                        height = gr.Number(label="Height", value=512)
                        width = gr.Number(label="Width", value=512)
                    num_inference_steps = gr.Slider(label="Steps", value=24)
                with gr.Column():
                    gallery = gr.Gallery()

            run.click(inference, inputs=[prompt, negative_prompt, num_samples, height, width, num_inference_steps, guidance_scale], outputs=gallery)

        demo.launch(share=True)



if __name__ == "__main__":
    HUGGINGFACE_TOKEN = "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"
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    OUTPUT_DIR = os.path.join('checkpoints', 'dreambooth')
    
    concepts_list = [
        {
            "instance_prompt": "photo of deva person",
            "class_prompt": "photo of a person",
            "instance_data_dir": os.path.join(os.getcwd(), 'datasets', 'dreambooth', 'deva'),
            "class_data_dir": os.path.join(os.getcwd(), 'datasets', 'dreambooth', 'person')
        },
    ]

    trainer = DreamBoothTrainer(HUGGINGFACE_TOKEN, MODEL_NAME, OUTPUT_DIR, concepts_list)
    trainer.download_scripts()
    trainer.train_model()
    trainer.convert_to_ckpt()
    trainer.launch_gradio_demo()
