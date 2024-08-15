import os
from ml_Project import logger
from urllib import request
import zipfile
from ml_Project.utils.common import create_directories

class DataIngestion:
    def __init__(self, data_dir, file_path, url, unzip_filepath):
        self.data_dir = data_dir
        self.file_path = file_path
        self.url = url
        self.unzip_filepath = unzip_filepath

    def create_dir(self):
        create_directories([self.data_dir])

    def download_file(self):
        if not os.path.exists(self.file_path):
            filename, headers = request.urlretrieve(
                url = self.url,
                filename = self.file_path
            )
            logger.info(f"{filename} download! with the following info: \n{headers}")
        else:
            logger.info(f"File already exists.")

    def extract_zip_file(self):
        unzip_file_path = self.unzip_filepath
        os.makedirs(unzip_file_path, exist_ok = True)
        with zipfile.ZipFile(self.file_path, "r") as zip_ref:
            zip_ref.extractall(unzip_file_path)