import logging
import os
import time
from pathlib import Path

import aiofiles
from chess.pgn import Game
from fastapi import UploadFile
from src.ml.pipelines.inference_pipeline import inference_pipeline
from utils.config import get_root_dir_path

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

uploads_dir = os.path.abspath(os.path.join(get_root_dir_path(), "src/uploads"))

def generate_chess_game_from_image(image_file_path: Path) -> Game:
    """
    Get prediction for an image of a scoresheet to a chess game in pgn format

    :param image_file_path: Path to image of a scoresheet
    :return: Game object of the generated chess game
    """
    logger.info(f"Starting the ml inference with image with filepath: {str(image_file_path)}")

    chess_game = inference_pipeline(str(image_file_path))

    if len(chess_game.errors) is not 0:
        raise Exception("Generated PGN is not in valid PGN format")
    else:
        return chess_game


async def save_pgn_file(chess_game: Game) -> Path | None:
    """
    Write PGN file into the directory

    :param chess_game: Chess Game that needs to be saved
    :return: Path to the saved Chess Game in PGN format
    """
    filename = int(time.time()*10000)
    pgn_file_path = f"{get_root_dir_path()}/src/pgn_files/{filename}.pgn"
    directory = os.path.dirname(pgn_file_path)
    if not os.path.exists(directory):
        logger.info(f"Directory '{directory}' does not exist. Creating the directory...")
        os.makedirs(directory, exist_ok=True)

    try:
        async with aiofiles.open(pgn_file_path, "w") as f:
            logger.info(f"Writing PGN file to: {pgn_file_path}")
            await f.write(str(chess_game))
        logger.info(f"File was saved successfully")

        return Path(pgn_file_path)
    except Exception as ex:
        logger.error(f"An error occurred while saving the file: {ex}")

class ImageService:
    def __init__(self, file: UploadFile):
        self.file = file

    async def save_image(self) -> Path:
        """
        Stores file to the upload folder in the file directory

        :return: Path of the saved file
        """
        if not (".png" in self.file.filename or ".jpeg" in self.file.filename or ".jpg" in self.file.filename):
            raise TypeError(f"File has to be type of: .png, jpeg or jpg BUT was {self.file.filename.split(".")[-1]}")
        else:
            file_path = f"{uploads_dir}/{self.file.filename}"
            async with aiofiles.open(file_path, "wb") as f:
                logger.info(f"Storing image in uploads folder: {uploads_dir}/{self.file.filename}")
                content = await self.file.read()
                await f.write(content)
            return Path(file_path)

    async def create_pgn_file(self) -> Path:
        """
        Create a chess game from a chess scoresheet with a ml inference

        :return: Path of the generated chess game (pgn format)
        """
        path_to_saved_file = await self.save_image()
        chess_game = generate_chess_game_from_image(path_to_saved_file)
        pgn_file_path = await save_pgn_file(chess_game)

        return pgn_file_path

