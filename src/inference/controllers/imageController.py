import http

from fastapi import APIRouter, UploadFile, File
from services.imageService import ImageService
import logging
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
        image_service = ImageService()
        response = await image_service.store_image(file)
        return {"result": f"{response}"}
    except TypeError as te:
        return te
    except Exception as e:
        logging.error(f"Fehler beim Speichern: {str(e)}")
        return {"error": {str(e)}}
