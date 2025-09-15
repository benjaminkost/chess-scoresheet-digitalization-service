import logging
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
async def test_upload_image_use_valid_file_returns_file(mocker, get_test_data_path, create_dummy_file):
    pgn_test_file_path = Path(f"{get_test_data_path}/pgn_files/test_game.pgn")
    mock_image_service = mocker.patch.object(
        ImageService,
        "create_pgn_file",
        new=mocker.AsyncMock(return_value=pgn_test_file_path)
    )

    result = await upload_image(create_dummy_file)

    assert mock_image_service.call_count == 1
    assert result.filename == "test_game.pgn"
    assert result.path == str(pgn_test_file_path)
    assert result.media_type == "text/plain"

@pytest.mark.asyncio
async def test_upload_image_use_non_existing_path_returns_error_dict(mocker, create_dummy_file):
    pgn_test_file_path = Path(f"non/existing/path")
    mock_image_service = mocker.patch.object(ImageService, "create_pgn_file",
                        new=mocker.AsyncMock(return_value=pgn_test_file_path))

    result = await upload_image(create_dummy_file)

    assert mock_image_service.call_count == 1
    assert type(result) == dict
    assert result["error"] == "No PGN-File found"

@pytest.mark.asyncio
async def test_upload_use_error_on_create_pgn_file_method_returns_typeerror_log(mocker, create_dummy_file, caplog):
    type_error_message = "File has to be type of: .png, jpeg or jpg BUT was test_file"
    type_error = TypeError(type_error_message)
    mock_image_service = mocker.patch.object(ImageService, "create_pgn_file",
                        new=mocker.AsyncMock(side_effect=type_error))
    spy_log = caplog.at_level(logging.ERROR)

    result = await upload_image(create_dummy_file)

    assert len(spy_log.args[0].messages) == 1
    assert spy_log.args[0].messages[0] == f"TypeError when saving: {type_error_message}"
    assert mock_image_service.call_count == 1
    assert type(result) == dict
    assert result["error"] == type_error_message

@pytest.mark.asyncio
async def test_upload_use_error_on_create_pgn_file_method_returns_error_log(mocker, create_dummy_file, caplog):
    type_error_message = "Generated PGN is not in valid PGN format"
    type_error = Exception(type_error_message)
    mock_image_service = mocker.patch.object(ImageService, "create_pgn_file",
                        new=mocker.AsyncMock(side_effect=type_error))
    spy_log = caplog.at_level(logging.ERROR)

    result = await upload_image(create_dummy_file)

    assert len(spy_log.args[0].messages) == 1
    assert spy_log.args[0].messages[0] == f"Exception when saving: {type_error_message}"
    assert mock_image_service.call_count == 1
    assert type(result) == dict
    assert result["error"] == type_error_message