import logging

import pytest
from PIL import PngImagePlugin

from src.ml.steps.dynamic_importer import load_png_image


def test_load_png_image_use_valid_file_path_returns_image(caplog, get_test_data_path):
    file_path = get_test_data_path / "images" / "001_0.png"
    spy_log = caplog.at_level(logging.INFO)

    result = load_png_image(file_path)

    assert len(spy_log.args[0].messages) == 2
    assert spy_log.args[0].messages[0] == f"File from path '{file_path}' is about to get loaded..."
    assert spy_log.args[0].messages[1] == f"Image from path '{file_path}' was successfully loaded"
    assert type(result) == PngImagePlugin.PngImageFile

def test_load_png_image_use_no_path_returns_exception(caplog):
    spy_log = caplog.at_level(logging.INFO)

    with pytest.raises(Exception) as ex:
        load_png_image(None)

    assert len(spy_log.args[0].messages) == 1
    assert spy_log.args[0].messages[0] == f"File from path '{None}' is about to get loaded..."
    assert ex.value.args[0] == "No file path provided."

def test_load_png_image_use_path_with_non_existing_file_returns_fileexistserror(caplog, get_test_data_path):
    path_with_non_existing_file = get_test_data_path / "images" / "001_1.png"
    spy_log = caplog.at_level(logging.INFO)

    with pytest.raises(FileExistsError) as ex:
        load_png_image(path_with_non_existing_file)

    assert len(spy_log.args[0].messages) == 1
    assert spy_log.args[0].messages[0] == f"File from path '{path_with_non_existing_file}' is about to get loaded..."
    assert ex.value.args[0] == f"No file at path '{path_with_non_existing_file}'"

def test_load_png_image_use_path_with_wrong_file_extension_returns_typeerror(caplog, get_test_data_path):
    path_with_wrong_file_extension = get_test_data_path / "other_files" / "test_text_file.txt"
    spy_log = caplog.at_level(logging.INFO)

    with pytest.raises(TypeError) as ex:
        load_png_image(path_with_wrong_file_extension)

    assert len(spy_log.args[0].messages) == 1
    assert spy_log.args[0].messages[0] == f"File from path '{path_with_wrong_file_extension}' is about to get loaded..."
    assert ex.value.args[0] == "File path must point to a file with a file extension '.png', '.jpeg' or '.jpg'"




