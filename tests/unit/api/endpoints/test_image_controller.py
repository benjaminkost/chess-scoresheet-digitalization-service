from io import BytesIO
from pathlib import Path
import pytest
from fastapi import UploadFile

from src.api.endpoints.controllers.image_controller import upload_image
from src.services.image_service import ImageService


@pytest.fixture
def create_dummy_file() -> UploadFile:
    file = BytesIO(b"Hello World")
    file.close()

    upload_file = UploadFile(file=file,
                            size=file.__sizeof__(),
                            filename="dummy.txt")

    return upload_file

@pytest.mark.asyncio
async def test_upload_image_valid_file_returns_file(mocker, get_test_data_path, create_dummy_file):
    pgn_test_file_path = Path(f"{get_test_data_path}/pgn_files/test_game.pgn")

    mocker.patch.object(
        ImageService,
        "create_pgn_file",
        new=mocker.AsyncMock(return_value=pgn_test_file_path)
    )
    result = await upload_image(create_dummy_file)

    assert result.filename == "test_game.pgn"