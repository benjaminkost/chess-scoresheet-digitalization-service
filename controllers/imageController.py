import http

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from ..services.imageService import ImageService
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

image_controller = APIRouter(
    prefix="/api/image"
)

@image_controller.post(
    "/upload",
    status_code=http.HTTPStatus.CREATED
)
async def upload_image(file: UploadFile = File(...)):
    try:
        image_service = ImageService(file=file)
        response = await image_service.store_image()
        # Create PGN File
        pgn_file_path = await image_service.create_pgn_file(response)

        # Check if PGN-File actually exists
        if pgn_file_path and Path(pgn_file_path).exists():
            return FileResponse(
                path=pgn_file_path,
                media_type="text/plain",
                filename=Path(pgn_file_path).name
            )
        else:
            return {"error": "No PGN-File found"}

    except TypeError as te:
        return te
    except Exception as e:
        logging.error(f"Error when saving: {str(e)}")
        return {"error": {str(e)}}
