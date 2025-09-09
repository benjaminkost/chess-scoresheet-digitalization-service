from io import BytesIO
import logging
from pathlib import Path
import pytest
from starlette.datastructures import UploadFile

from src.services import image_service
from src.services.image_service import ImageService

logger = logging.getLogger(__name__)
abs_path_to_sut = image_service.__file__

def create_upload_file(file_path: str) -> UploadFile:
    file = open(file_path, "rb")
    file = BytesIO(file.read())

    upload_file = UploadFile(file=file,
                            size=file.__sizeof__(),
                            filename=Path(file_path).name)

    return upload_file

def create_dummy_file() -> UploadFile:
    file = BytesIO(b"Hello World")
    file.close()

    upload_file = UploadFile(file=file,
                            size=file.__sizeof__(),
                            filename="dummy.txt")

    return upload_file

@pytest.mark.asyncio
class TestImageService:

    async def test_store_image_use_non_compatible_file_extension_raises_type_error(self):
        file = create_dummy_file()
        sut = ImageService(file)

        with pytest.raises(TypeError) as te:
            await sut.store_image()

        assert te.value.args[0] == f"File has to be type of: .png, jpeg or jpg BUT was {file.filename.split(".")[-1]}"

    async def test_store_image_use_compatible_saves_file(self, mocker, caplog, get_data_path):
        file = create_upload_file(f"{str(get_data_path)}/images/001_0.png")
        mock_file = mocker.patch("aiofiles.open")
        mock_file.return_value.write.return_value = None
        spy_log = caplog.at_level(logging.INFO)
        abs_path_to_sut_without_script = abs_path_to_sut.replace("/image_service.py", "")
        sut = ImageService(file)

        result = await sut.store_image()

        assert len(spy_log.args[0].messages) == 1
        assert spy_log.args[0].messages[0] == f"Storing image in uploads folder: {abs_path_to_sut_without_script}/uploads/{file.filename}"
        assert result == "File was saved successfully"