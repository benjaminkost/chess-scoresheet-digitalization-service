import aiofiles
from fastapi import UploadFile

class ImageService:
    async def store_image(self, file: UploadFile):
        if not (".png" in file.filename or ".jpeg" in file.filename or ".jpg" in file.filename):
            return "This is not a image, the file has to of type: .png, jpeg or jpg"
        else:
            async with aiofiles.open(f"/app/uploads/{file.filename}", "wb") as f:
                content = await file.read()
                await f.write(content)
            return "File was saved successfully"

