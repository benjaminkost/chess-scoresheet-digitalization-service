import io
from io import BytesIO
import logging
from pathlib import Path

import chess.pgn
import pytest
from starlette.datastructures import UploadFile

from src.services.image_service import ImageService, generate_chess_game_from_image, save_pgn_file
from utils.config import get_root_dir_path

logger = logging.getLogger(__name__)
path_of_sut_script = "src.services.image_service"

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

    async def test_save_image_use_non_compatible_file_extension_raises_type_error(self):
        file = create_dummy_file()
        sut = ImageService(file)

        with pytest.raises(TypeError) as te:
            await sut.save_image()

        assert te.value.args[0] == f"File has to be type of: .png, jpeg or jpg BUT was {file.filename.split(".")[-1]}"

    async def test_save_image_use_compatible_saves_file(self, mocker, caplog, get_data_path):
        file = create_upload_file(f"{str(get_data_path)}/images/001_0.png")
        mock_file = mocker.patch("aiofiles.open")
        mock_file.return_value.write.return_value = None
        spy_log = caplog.at_level(logging.INFO)
        sut = ImageService(file)

        result = await sut.save_image()

        assert len(spy_log.args[0].messages) == 1
        assert spy_log.args[0].messages[0] == f"Storing image in uploads folder: {get_root_dir_path()}/src/uploads/{file.filename}"
        assert result == Path(f"{get_root_dir_path()}/src/uploads/{file.filename}")

def test_generate_chess_game_from_image_use_no_pgn_formated_file_raises_error(mocker, get_data_path):
    pgn = io.StringIO("1. piacpq e5 2. d4")
    mock_chess_game_wrong_pgn = chess.pgn.read_game(pgn)
    mock_inference_pipeline = mocker.patch(f"{path_of_sut_script}.inference_pipeline")
    mock_inference_pipeline.return_value = mock_chess_game_wrong_pgn
    image_file_path = Path(__file__)

    with pytest.raises(Exception) as ex:
        generate_chess_game_from_image(image_file_path)

    assert ex.value.args[0] == "Generated PGN is not in valid PGN format"
    assert mock_inference_pipeline.call_count == 1

def test_generate_chess_game_from_image_use_valid_pgn_formated_file_raises_error(mocker, get_data_path):
    pgn = io.StringIO("1. e4 e5 2. d4")
    mock_chess_game_wrong_pgn = chess.pgn.read_game(pgn)
    mock_inference_pipeline = mocker.patch(f"{path_of_sut_script}.inference_pipeline")
    mock_inference_pipeline.return_value = mock_chess_game_wrong_pgn
    mock_image_file_path = Path(__file__)

    result = generate_chess_game_from_image(mock_image_file_path)

    assert type(result) == chess.pgn.Game
    assert result == mock_chess_game_wrong_pgn

@pytest.mark.asyncio
async def test_save_pgn_file_use_valid_game_write_to_filesystem(mocker, caplog):
    pgn = io.StringIO("1. e4 e5 2. d4")
    chess_game = chess.pgn.read_game(pgn)
    mock_file = mocker.patch("aiofiles.open")
    mock_file.return_value.write.return_value = None
    spy_log = caplog.at_level(logging.INFO)
    mock_time = mocker.patch("time.time")
    mocked_current_time = float(1234567890.1234)
    mock_time.return_value = mocked_current_time

    result = await save_pgn_file(chess_game)

    assert len(spy_log.args[0].messages) == 2
    assert spy_log.args[0].messages[0] == f"Writing PGN file to: {get_root_dir_path()}/src/pgn_files/12345678901234.pgn"
    assert spy_log.args[0].messages[1] == f"File was saved successfully"
    assert result == Path(f"{get_root_dir_path()}/src/pgn_files/12345678901234.pgn")