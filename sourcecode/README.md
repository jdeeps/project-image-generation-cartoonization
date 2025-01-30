# Image-Generation-and-Cartoonization

This project has 2 stages - Finetuning a pretrained Text-to-image model [StableDiffusion V-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) using [DreamBoothing](https://dreambooth.github.io/) technique and [Cartoon GAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf).

## Files
- ```download_datasets.py``` file downloads the datasets for Cartoon GAN and Dreambooth networks.
- ```download_checkpoints.py``` file downloads the model checkpoints.
- ```cartoongan.py``` file contains the training and inference code for Cartoon GAN.
- ```dreambooth.py``` file contains the training and inference code for Dreambooth architecture in OOPS format.
- ```main.py``` file is the entry point to run the inference of Cartoon GAN.
- ```notebooks/``` directory contains the raw notebook files to run on Colab.
- ```notebooks/DreamBooth_Stable_Diffusion-1.ipynb``` Notebook for finetuning and inference of Dreambooth network.

## Important Links
- Cartoon GAN Dataset: https://drive.google.com/file/d/1esNF4ZDtdQ0-UpIqNq4OHSkD-8Yp0JFP/view?usp=sharing
- Project Repo: https://github.com/gdevakumar/Image-Generation-and-Cartoonization
- Project Raw Artifacts: https://drive.google.com/drive/folders/1h52-b2ieE26NLJcQxnbzfc2B5WFK1LA7?usp=sharing

## To run this project
1. Clone the repository
```
git clone https://github.com/gdevakumar/Image-Generation-and-Cartoonization.git
cd Image-Generation-and-Cartoonization
```

2. Create a virtual environment for best practice (*optional*)
- On **Windows** machines
```
python -m venv env
env\Scripts\activate
```

- On **Linux/Mac** machines
```
python -m venv env
source env/bin/activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Download datasets
```
python download_datasets.py
```


## Inference
1. Download the best checkpoints 
```
python download_checkpoints.py
```

2. Run the inference script of Cartoon GAN
```
python main.py --test cartoongan
```

3. Run the training script of Cartoon GAN
```
python main.py --train cartoongan
```

### Note: 
- Finetuning Dreambooth model requires Nvidia GPUs, else its gonna throw errors due to the usage of fp16 precision format supported only on few hardware! 
- Use `notebooks/DreamBooth_Stable_Diffusion-1.ipynb` notebook for this case.

