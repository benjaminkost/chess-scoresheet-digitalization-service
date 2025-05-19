import logging
import os

import aiofiles
from fastapi import UploadFile
from ..pipelines.inference_pipeline import inference_pipeline

logging.basicConfig(
    level=logging.INFO,  # Log-Ebene (z. B. DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log-Format
)
logger = logging.getLogger(__name__)  # Logger mit Modulnamen beziehen

class ImageService:
    def __init__(self, file: UploadFile):
        self.file = file
    async def store_image(self):
        logger.info(f"{os.getcwd()}")
        if not (".png" in self.file.filename or ".jpeg" in self.file.filename or ".jpg" in self.file.filename):
            return "This is not a image, the file has to of type: .png, jpeg or jpg"
        else:
            async with aiofiles.open(f"./uploads/{self.file.filename}", "wb") as f:
                logger.info(f"Storing image in uploads folder: ./uploads/{self.file.filename}")
                content = await self.file.read()
                await f.write(content)
            return "File was saved successfully"

    async def create_pgn_file(self, response):
        if response == "File was saved successfully":
            # load inference pipeline
            filepath = f"./uploads/{self.file.filename}"

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

