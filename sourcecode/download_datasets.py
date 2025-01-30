import os
import zipfile
import requests
from tqdm import tqdm


class DatasetDownloader:
    def __init__(self, data_dir: str = os.path.join(os.getcwd(), 'datasets')):
        """
        Initializes the DatasetDownloader with the specified data directory.

        Args:
            data_dir (str): The directory where datasets will be stored.
        """
        self.data_dir = data_dir
        self._create_data_directory()

    def _create_data_directory(self) -> None:
        """Creates the data directory if it does not exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created a directory to store dataset: {self.data_dir}")
        else:
            print(f"Directory for dataset exists!")

    def download_and_unzip(self, dataset: str) -> None:
        """
        Downloads and unzips the specified dataset for the Cartoon GAN network or the Dreambooth network and deletes zip file

        Args:
            dataset (str): The name of the dataset to download. 
                           Acceptable values are "danbooru" or "dreambooth".

        Returns:
            None

        Raises:
            ValueError: If the dataset name is not valid.
        """
        if dataset == "danbooru":
            self._download_file_from_s3(DANBOORU_DATA_URL, os.path.join(self.data_dir, 'danbooru.zip'))
        elif dataset == "dreambooth":
            self._download_file_from_s3(DREAMBOOTH_DATA_URL, os.path.join(self.data_dir, 'dreambooth.zip'))
        else:
            raise ValueError("[ERROR] The dataset name is not valid. Please specify either 'danbooru' or 'dreambooth'.")

        self._unzip_file(os.path.join(self.data_dir, f"{dataset}.zip"), os.path.join(self.data_dir, dataset))
        os.remove(os.path.join(self.data_dir, f"{dataset}.zip"))

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

    def _unzip_file(self, zip_file_path: str, extract_to_folder: str) -> None:
        """
        Unzips a file to the specified folder.

        Args:
            zip_file_path (str): The path to the zip file.
            extract_to_folder (str): The folder where the files will be extracted.

        Returns:
            None
        """
        if not os.path.isfile(zip_file_path):
            print(f"The file {zip_file_path} does not exist.")
            return

        if not os.path.exists(extract_to_folder):
            os.makedirs(extract_to_folder)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_folder)
            print(f"Files have been extracted to {extract_to_folder}")


DANBOORU_DATA_URL = "https://data255-cartoongan.s3.amazonaws.com/danbooru/danbooru.zip"
DREAMBOOTH_DATA_URL = "https://data255-cartoongan.s3.amazonaws.com/dreambooth/dreambooth.zip"



if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), 'datasets')
    downloader = DatasetDownloader(data_dir)
    downloader.download_and_unzip("danbooru")
    downloader.download_and_unzip("dreambooth")
