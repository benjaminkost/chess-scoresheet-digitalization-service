import http

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from src.services.image_service import ImageService
import logging

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

ImageController = APIRouter(
    prefix="/api/image"
)

@ImageController.post(
    "/upload",
    response_class=FileResponse,
    response_model=None,
    status_code=http.HTTPStatus.CREATED
)
async def upload_image(file: UploadFile = File(...)) -> dict[str, str] | FileResponse:
    """
    Create a chess game in PGN format from an uploaded scoresheet of a chess game

    :param file: image of a chess scoresheet
    :return: The generated Chess Game in PGN format as a file or error messages
    """
    try:
        pgn_file_path = await ImageService(file=file).create_pgn_file()

        logger.info(f"Path to generated PGN file is: {pgn_file_path}")

        if pgn_file_path.exists():
            return FileResponse(
                path=str(pgn_file_path),
                media_type="text/plain",
                filename=pgn_file_path.name
            )
        else:
            logger.error(f"Path to PGN file '{logger}' does not exist")
            return {"error": "No PGN-File found"}

    except TypeError as te:
        logger.error(f"TypeError when saving: {te}")
        return {"error": str(te)}
    except Exception as e:
        logger.error(f"Exception when saving: {str(e)}")
        return {"error": str(e)}
