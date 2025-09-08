from io import BytesIO
import logging
from pathlib import Path
import pytest
from starlette.datastructures import UploadFile

from src.services.image_service import ImageService

logger = logging.getLogger(__name__)

def create_upload_file(file_path: str) -> UploadFile:
    file = open(file_path, "rb")
    file = BytesIO(file.read())
    file.close()

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


class TestImageService:

    @pytest.fixture
    def create_ImageService_with_dummy_file(self) -> ImageService:

        file = create_dummy_file()

        return ImageService(file)

    @pytest.fixture
    def create_ImageService_with_real_file(self, get_data_path: Path) -> ImageService:
        path_file = get_data_path / "images" / "001_0.png"

        file = create_upload_file(str(path_file))

        return ImageService(file)

    @pytest.mark.asyncio
    async def test_store_image_use_non_compatible_file_extension_raises_type_error(self, caplog, create_ImageService_with_dummy_file: ImageService):
        caplog.set_level(logging.ERROR)
        file = create_dummy_file()
        sut = ImageService(file)

        with pytest.raises(TypeError):
            await sut.store_image()