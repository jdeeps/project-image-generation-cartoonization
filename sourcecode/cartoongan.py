import os
import cv2
import math
import time
import torch
import numpy as np
import gradio as gr
import torch.nn as nn
from tqdm import tqdm
from typing import Union
from torch import sigmoid
import torch.optim as optim
from torch.nn import BCELoss
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image, ImageFilter
from skimage.filters import gaussian
from skimage import io, img_as_ubyte
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from download_checkpoints import CheckpointsDownloader

class ResidualBlock(nn.Module):
    """
    Residual Network Block with 2 Convolution and BatchNorm layers
    """
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm_1 = nn.BatchNorm2d(256)
        self.norm_2 = nn.BatchNorm2d(256)

    def forward(self, x):
        output = self.norm_2(self.conv_2(F.relu(self.norm_1(self.conv_1(x)))))
        return output + x 

class Generator(nn.Module):
    """
    Generator Network
    """
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.norm_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm_2 = nn.BatchNorm2d(128)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm_3 = nn.BatchNorm2d(256)
        self.residual_blocks = [ResidualBlock() for _ in range(8)]
        self.res = nn.Sequential(*self.residual_blocks)
        self.conv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm_4 = nn.BatchNorm2d(128)
        self.conv_8 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_9 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm_5 = nn.BatchNorm2d(64)
        self.conv_10 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.norm_1(self.conv_1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm_2(self.conv_3(self.conv_2(x))))
        x = F.relu(self.norm_3(self.conv_5(self.conv_4(x))))
        x = self.res(x)
        x = F.relu(self.norm_4(self.conv_7(self.conv_6(x))))
        x = F.relu(self.norm_5(self.conv_9(self.conv_8(x))))
        x = self.dropout(x)
        x = self.conv_10(x)
        x = sigmoid(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator Network
    """
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm_1 = nn.BatchNorm2d(128)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm_2 = nn.BatchNorm2d(256)
        self.conv_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm_3 = nn.BatchNorm2d(256)
        self.conv_7 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x))
        x = F.leaky_relu(self.norm_1(self.conv_3(F.leaky_relu(self.conv_2(x)))), negative_slope=0.2)
        x = F.leaky_relu(self.norm_2(self.conv_5(F.leaky_relu(self.conv_4(x)))), negative_slope=0.2)
        x = F.leaky_relu(self.norm_3(self.conv_6(x)), negative_slope=0.2)
        x = self.conv_7(x)
        x = sigmoid(x)
        return x


class CartoonGAN:
    def __init__(self):
        self.image_size = 256
        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = os.path.join(os.getcwd(), 'datasets', 'danbooru', 'data', 'train')

        self._create_directories()
        self._prepare_datasets()
        self._prepare_models()
        self._prepare_optimizers()
        self._prepare_losses()

    def _create_directories(self) -> None:
        """
        Creates directory to store intermediate training results (pair of images at the end of each epoch)
        """
        if not os.path.exists('intermediate_results'):
            os.makedirs('intermediate_results')

    def _prepare_datasets(self) -> None:
        """
        Apply transformation and load the training images(original, cartoon, real) into a Dataset Loader object 
        and split into train and validation splits
        """
        transformer = T.Compose([
            T.CenterCrop(self.image_size),
            T.ToTensor()
        ])

        cartoon_dataset = ImageFolder(os.path.join(self.data_dir, 'cartoons'), transformer)
        len_training_set = math.floor(len(cartoon_dataset) * 0.9)
        len_valid_set = len(cartoon_dataset) - len_training_set
        training_set, _ = random_split(cartoon_dataset, (len_training_set, len_valid_set))
        self.cartoon_image_dataloader_train = DataLoader(training_set, self.batch_size, shuffle=True, num_workers=0)

        smoothed_cartoon_dataset = ImageFolder(os.path.join(self.data_dir, 'cartoons_smoothed'), transformer)
        len_training_set = math.floor(len(smoothed_cartoon_dataset) * 0.9)
        len_valid_set = len(smoothed_cartoon_dataset) - len_training_set
        training_set, _ = random_split(smoothed_cartoon_dataset, (len_training_set, len_valid_set))
        self.smoothed_cartoon_image_dataloader_train = DataLoader(training_set, self.batch_size, shuffle=True, num_workers=0)

        real_dataset = ImageFolder(os.path.join(self.data_dir, 'real'), transformer)
        len_training_set = math.floor(len(real_dataset) * 0.9)
        len_valid_set = len(real_dataset) - len_training_set
        training_set, validation_set = random_split(real_dataset, (len_training_set, len_valid_set))
        self.photo_dataloader_train = DataLoader(training_set, self.batch_size, shuffle=True, num_workers=0)
        self.photo_dataloader_valid = DataLoader(validation_set, self.batch_size, shuffle=True, num_workers=0)

    def _prepare_models(self):
        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)

        vgg16 = models.vgg16(weights='DEFAULT')
        self.feature_extractor = vgg16.features[:24].to(self.device)
        for param in self.feature_extractor.parameters():
            param.require_grad = False

    def _prepare_optimizers(self):
        lr = 0.0002
        beta1 = 0.5
        beta2 = 0.999
        self.d_optimizer = optim.Adam(self.D.parameters(), lr, [beta1, beta2])
        self.g_optimizer = optim.Adam(self.G.parameters(), lr, [beta1, beta2])

    def _prepare_losses(self):
        self.discriminator_loss = self.DiscriminatorLoss(self.device)
        self.generator_loss = self.GeneratorLoss(self.feature_extractor, self.device)

    class DiscriminatorLoss(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.bce_loss = BCELoss()
            self.device = device

        def forward(self, discriminator_output_of_cartoon_input, discriminator_output_of_cartoon_smoothed_input, discriminator_output_of_generated_image_input, epoch, write_to_tensorboard=False):
            actual_batch_size = discriminator_output_of_cartoon_input.size()[0]
            zeros = torch.zeros([actual_batch_size, 1, 64, 64]).to(self.device)
            ones = torch.ones([actual_batch_size, 1, 64, 64]).to(self.device)

            d_loss_cartoon = self.bce_loss(discriminator_output_of_cartoon_input, ones)
            d_loss_cartoon_smoothed = self.bce_loss(discriminator_output_of_cartoon_smoothed_input, zeros)
            d_loss_generated_input = self.bce_loss(discriminator_output_of_generated_image_input, zeros)

            d_loss = d_loss_cartoon + d_loss_cartoon_smoothed + d_loss_generated_input

            return d_loss

    class GeneratorLoss(nn.Module):
        def __init__(self, feature_extractor, device):
            super().__init__()
            self.w = 0.000005
            self.bce_loss = BCELoss()
            self.feature_extractor = feature_extractor
            self.device = device

        def forward(self, discriminator_output_of_generated_image_input, generator_input, generator_output, epoch, is_init_phase=False, write_to_tensorboard=False):
            if is_init_phase:
                g_content_loss = self._content_loss(generator_input, generator_output)
                g_adversarial_loss = 0.0
                g_loss = g_content_loss
            else:
                g_adversarial_loss = self._adversarial_loss_generator_part_only(discriminator_output_of_generated_image_input)
                g_content_loss = self._content_loss(generator_input, generator_output)
                g_loss = g_adversarial_loss + self.w * g_content_loss

            return g_loss

        def _adversarial_loss_generator_part_only(self, discriminator_output_of_generated_image_input):
            actual_batch_size = discriminator_output_of_generated_image_input.size()[0]
            ones = torch.ones([actual_batch_size, 1, 64, 64]).to(self.device)
            return self.bce_loss(discriminator_output_of_generated_image_input, ones)

        def _content_loss(self, generator_input, generator_output):
            return (self.feature_extractor(generator_output) - self.feature_extractor(generator_input)).norm(p=1)

    def train(self, num_epochs: int, checkpoint_dir: str) -> (list, list):
        """
        Training loop for Cartoon GAN.

        Args:
            num_epochs (int): The number of epochs to train the model.
            checkpoint_dir (str): The directory to save checkpoints.

        Returns:
            tuple: A tuple containing two lists:
                - losses (list): Training losses for each epoch of generator and discriminator.
                - validation_losses (list): Validation losses for each epoch of generator.
        """
        best_valid_loss = math.inf
        epochs_already_done = 0
        losses = []
        validation_losses = []
        os.makedirs('checkpoints', exist_ok=True)
        checkpoints = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
        if checkpoints:
            last_checkpoint = sorted(checkpoints)[-1]
            checkpoint = torch.load(os.path.join(checkpoint_dir, last_checkpoint), map_location=self.device)
            best_valid_loss = checkpoint['best_valid_loss']
            epochs_already_done = checkpoint['last_epoch']
            losses = checkpoint['losses']
            validation_losses = checkpoint['validation_losses']
            self.D.load_state_dict(checkpoint['d_state_dict'])
            self.G.load_state_dict(checkpoint['g_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            print(f'Loaded checkpoint {last_checkpoint} with g_valid_loss {checkpoint["g_valid_loss"]}, best_valid_loss {best_valid_loss}, {epochs_already_done} epochs, and total number of losses {len(losses)}')

        init_epochs = 10
        print_every = 100
        start_time = time.time()

        for epoch in range(num_epochs - epochs_already_done):
            epoch += epochs_already_done

            for index, ((photo_images, _), (smoothed_cartoon_images, _), (cartoon_images, _)) in enumerate(zip(self.photo_dataloader_train, self.smoothed_cartoon_image_dataloader_train, self.cartoon_image_dataloader_train)):
                batch_size = photo_images.size(0)
                photo_images = photo_images.to(self.device)
                smoothed_cartoon_images = smoothed_cartoon_images.to(self.device)
                cartoon_images = cartoon_images.to(self.device)

                # training the discriminator
                self.d_optimizer.zero_grad()
                d_of_cartoon_input = self.D(cartoon_images)
                d_of_cartoon_smoothed_input = self.D(smoothed_cartoon_images)
                d_of_generated_image_input = self.D(self.G(photo_images))
                write_only_one_loss_from_epoch_not_every_batch_loss = (index == 0)
                d_loss = self.discriminator_loss(d_of_cartoon_input, d_of_cartoon_smoothed_input, d_of_generated_image_input, epoch, write_to_tensorboard=write_only_one_loss_from_epoch_not_every_batch_loss)
                d_loss.backward()
                self.d_optimizer.step()

                # training the generator
                self.g_optimizer.zero_grad()
                g_output = self.G(photo_images)
                d_of_generated_image_input = self.D(g_output)
                init_phase = epoch < init_epochs
                g_loss = self.generator_loss(d_of_generated_image_input, photo_images, g_output, epoch, is_init_phase=init_phase, write_to_tensorboard=write_only_one_loss_from_epoch_not_every_batch_loss)
                g_loss.backward()
                self.g_optimizer.step()

                if (index % print_every) == 0:
                    losses.append((d_loss.item(), g_loss.item()))
                    now = time.time()
                    current_run_time = now - start_time
                    start_time = now
                    print(f"Epoch {epoch + 1}/{num_epochs} | d_loss {d_loss.item():.4f} | g_loss {g_loss.item():.4f} | time {current_run_time:.0f}s | total number of losses {len(losses)}")

                self._save_training_result(photo_images, g_output)

            with torch.no_grad():
                self.D.eval()
                self.G.eval()
                for batch_index, (photo_images, _) in enumerate(self.photo_dataloader_valid):
                    photo_images = photo_images.to(self.device)
                    g_output = self.G(photo_images)
                    d_of_generated_image_input = self.D(g_output)
                    g_valid_loss = self.generator_loss(d_of_generated_image_input, photo_images, g_output, epoch, is_init_phase=init_phase, write_to_tensorboard=write_only_one_loss_from_epoch_not_every_batch_loss)

                    if batch_index % print_every == 0:
                        validation_losses.append(g_valid_loss.item())
                        now = time.time()
                        current_run_time = now - start_time
                        start_time = now
                        print(f"Epoch {epoch + 1}/{num_epochs} | validation loss {g_valid_loss.item():.4f} | time {current_run_time:.0f}s | total number of losses {len(validation_losses)}")

                self.D.train()
                self.G.train()

                if g_valid_loss.item() < best_valid_loss:
                    print(f"Generator loss improved from {best_valid_loss} to {g_valid_loss.item()}")
                    best_valid_loss = g_valid_loss.item()

                checkpoint = {
                    'g_valid_loss': g_valid_loss.item(),
                    'best_valid_loss': best_valid_loss,
                    'losses': losses,
                    'validation_losses': validation_losses,
                    'last_epoch': epoch + 1,
                    'd_state_dict': self.D.state_dict(),
                    'g_state_dict': self.G.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1:03d}.pth'))
                if best_valid_loss == g_valid_loss.item():
                    print("Overwrite best checkpoint")
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_checkpoint.pth'))

        return losses, validation_losses

    def _save_training_result(self, input, output):
        image_input = input[0].detach().cpu().numpy()
        image_output = output[0].detach().cpu().numpy()
        image_input = np.transpose(image_input, (1, 2, 0))
        image_output = np.transpose(image_output, (1, 2, 0))
        filename = str(int(time.time()))
        path_input = os.path.join('intermediate_results', f'{filename}_input.jpg')
        path_output = os.path.join('intermediate_results', f'{filename}.jpg')
        plt.imsave(path_input, image_input)
        plt.imsave(path_output, image_output)


def plot_loss_curves(losses: list, val_losses: list, save_path: str = "plots") -> None:
    """
    Plots and saves the training and validation loss curves for the generator and discriminator.

    Args:
        losses (list): A list of tuples, where each tuple contains the discriminator loss and generator loss.
        val_losses (list): A list of generator validation losses.
        save_path (str): The path where the plot will be saved. Defaults to "./plots".
    """
    os.makedirs(save_path, exist_ok=True)
    d_losses = [x[0] for x in losses]
    g_losses = [x[1] for x in losses]

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator training loss')
    plt.plot(g_losses, label='Generator training loss')
    plt.plot(val_losses, label='Generator validation loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend(frameon=False)
    
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()


def load_checkpoint(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads the checkpoint for the Generator model.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device to load the model on.

    Returns:
        torch.nn.Module: The loaded Generator model.
    """
    if not os.path.exists('best_checkpoints'):
        downloader = CheckpointsDownloader()
        downloader.download()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Generator().to(device)
    model.load_state_dict(checkpoint['g_state_dict'])
    return model

def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Preprocesses the input image for the model.

    Args:
        image_path (str): Path to the input image.
        device (torch.device): The device to load the image on.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    img = Image.open(image_path)
    transform = T.Compose([
        T.Resize([256, 256]),
        T.ToTensor()
    ])
    img = transform(img).unsqueeze(0).to(device)
    return img

def cartoon_gan(input_image: str) -> np.ndarray:
    """
    Generates a cartoon image using the trained Generator model.

    Args:
        input_image (str): Path to the input image.

    Returns:
        np.ndarray: The generated cartoon image.
    """
    img = preprocess_image(input_image, device)
    result_image_checkpoint = G_inference(img)
    cartoon = np.transpose(result_image_checkpoint[0].cpu().detach().numpy(), (1, 2, 0))
    return cartoon

def create_gradio_interface() -> gr.Interface:
    """
    Creates the Gradio interface for the Cartoon GAN.

    Returns:
        gr.Interface: The Gradio interface.
    """
    return gr.Interface(fn=cartoon_gan, inputs=gr.Image(type='filepath'), outputs="image", title="Cartoon GAN")

def cartoongan_inference() -> None:
    """
    Main function to load the model, create the Gradio interface, and launch it.
    """
    global device, G_inference

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(os.getcwd(), 'best_checkpoints', 'cartoongan', 'model_3_checkpoint_ep220.pth')

    G_inference = load_checkpoint(checkpoint_path, device)
    cartoon_gan_interface = create_gradio_interface()
    cartoon_gan_interface.launch(share=True)



if __name__ == "__main__":
    gan = CartoonGAN()
    losses, validation_losses = gan.train(1, 'checkpoints')
    plot_loss_curves(losses, validation_losses)
