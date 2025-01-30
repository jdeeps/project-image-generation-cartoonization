import os
import requests
from tqdm import tqdm


class CheckpointsDownloader:
    def __init__(self, chkpt_dir: str = os.path.join(os.getcwd(), 'best_checkpoints')):
        """
        Initializes the CheckpointsDownloader with the specified checkpoint directory.

        Args:
            chkpt_dir (str): The directory where checkpoints will be stored.
        """
        self.chkpt_dir = chkpt_dir
        self._create_checkpoint_directory()

    def _create_checkpoint_directory(self) -> None:
        """Creates the checkpoint directory if it does not exist."""
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
            os.makedirs(os.path.join(self.chkpt_dir, 'cartoongan'))
            os.makedirs(os.path.join(self.chkpt_dir, 'dreambooth'))
            print(f"Created a directory to store checkpoints: {self.chkpt_dir}")
        else:
            print(f"Directory for checkpoints exists!")

    def download(self) -> None:
        """
        Downloads the checkpoints for the Cartoon GAN network.

        Returns:
            None
        """
        # self._download_file_from_s3(MODEL_1, os.path.join(os.getcwd(), 'best_checkpoints', 'cartoongan', 'model_1_checkpoint_ep210.pth'))
        # self._download_file_from_s3(MODEL_2, os.path.join(os.getcwd(), 'best_checkpoints', 'cartoongan', 'model_2_checkpoint_ep210.pth'))
        self._download_file_from_s3(MODEL_3, os.path.join(os.getcwd(), 'best_checkpoints', 'cartoongan', 'model_3_checkpoint_ep220.pth'))

    def _download_file_from_s3(self, url: str, local_file_path: str) -> None:
        """
        Downloads a file from an S3 URL to a local file path.

        Args:
            url (str): The URL of the file to download.
            local_file_path (str): The local path where the file will be saved.

        Returns:
            None

        Raises:
            requests.exceptions.RequestException: If there is an issue with the download request.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_file_path, 'wb') as file:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f'Downloading {os.path.basename(local_file_path)}'):
                    file.write(chunk)

            print(f'Successfully downloaded {url} to {local_file_path}')
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f'Error downloading file: {e}')



MODEL_1 = "https://data255-cartoongan.s3.amazonaws.com/checkpoints/model_1_checkpoint_ep210.pth"
MODEL_2 = "https://data255-cartoongan.s3.amazonaws.com/checkpoints/model_2_checkpoint_ep210.pth"
MODEL_3 = "https://data255-cartoongan.s3.amazonaws.com/checkpoints/model_3_checkpoint_ep220.pth"



if __name__ == "__main__":
    downloader = CheckpointsDownloader()
    downloader.download()
