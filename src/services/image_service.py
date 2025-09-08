import logging
import os
import aiofiles
from fastapi import UploadFile
from src.ml.run_inference import inference_pipeline

# Configure Logger:
# ANSI Escape Code for white letters
WHITE = "\033[37m"
RESET = "\033[0m"  # reset of color

# Logger configure
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console-Handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter with ANSI Escape Code for white letters
formatter = logging.Formatter(f'{WHITE}%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s{RESET}')
handler.setFormatter(formatter)

# Handler for Logger added
logger.addHandler(handler)

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(dir_path, os.pardir)
uploads_dir = os.path.abspath(os.path.join(parent_dir, "uploads"))

class ImageService:
    def __init__(self, file: UploadFile):
        self.file = file

    async def store_image(self) -> str:
        """
        Stores file to the upload folder in the file dictory

        :return: Response if the file was successfully saved
        """
        print("test")
        if not (".png" in self.file.filename or ".jpeg" in self.file.filename or ".jpg" in self.file.filename):
            raise TypeError("This is not a image, the file has to be of type: .png, jpeg or jpg")
        else:
            async with aiofiles.open(f"{uploads_dir}/{self.file.filename}", "wb") as f:
                logger.info(f"Storing image in uploads folder: {dir_path}/uploads/{self.file.filename}")
                content = await self.file.read()
                await f.write(content)
            return "File was saved successfully"

    async def create_pgn_file(self, response):
        if response == "File was saved successfully":
            # load inference pipeline
            filepath = f"{uploads_dir}/{self.file.filename}"

            # Define model name
            model_name = "trocr-base-handwritten-with-pre-and-post-processing"

            logger.info(f"Starting the ml inference with image with filepath: {filepath}")

            # get prediction pgn string
            pgn_file_str = inference_pipeline(filepath, model_name)

            # Write pgn file into the directory
            filename_without_type = self.file.filename.split(".")[0]
            file_path = f"pgn_files/{filename_without_type}.pgn"
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                logger.info(f"Directory '{directory}' does not exist. Creating the directory...")
                os.makedirs(directory, exist_ok=True)

            logger.info(f"Current dir: {os.getcwd()} and File path: {file_path}")

            try:
                async with aiofiles.open(file_path, "w") as f:
                    logger.info(f"Writing PGN file to: {file_path}")
                    await f.write(pgn_file_str)
                logger.info("File was saved successfully.")
            except Exception as e:
                logger.error(f"An error occurred while saving the file: {e}")

            return file_path
        else:
            return "No file to process"

