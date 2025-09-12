import http

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from src.services.image_service import ImageService
import logging

logging.basicConfig(level=logging.INFO)

image_controller = APIRouter(
    prefix="/api/image"
)

@image_controller.post(
    "/upload",
    response_class=FileResponse,
    response_model=None,
    status_code=http.HTTPStatus.CREATED
)
async def upload_image(file: UploadFile = File(...)) -> dict[str, str] | FileResponse:
    """
    Create a chess game in PGN format from uploaded game

    :param file: image of a chess scoresheet
    :return:
    """
    try:
        pgn_file_path = await ImageService(file=file).create_pgn_file()

        # Check if PGN-File actually exists
        if pgn_file_path.exists():
            return FileResponse(
                path=str(pgn_file_path),
                media_type="text/plain",
                filename=pgn_file_path.name
            )
        else:
            return {"error": "No PGN-File found"}

    except TypeError as te:
        logging.error(f"TypeError when saving: {te}")
        return {"error": str(te)}
    except Exception as e:
        logging.error(f"Error when saving: {str(e)}")
        return {"error": str(e)}
