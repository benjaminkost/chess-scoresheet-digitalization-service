import torch
from PIL import Image

from src.ml.pipelines.inference_pipeline import inference_pipeline


def test_inference_pipeline(mocker, get_test_data_path):
    # Give
    path_of_sut_script = "src.ml.pipelines.inference_pipeline"
    image_file_path = get_test_data_path / "images" / "001_0.png"
    mock_image = mocker.patch(f"{path_of_sut_script}.load_png_image")
    mock_image.return_value = Image.new("RGB", (10, 10))

    mock_move_boxes = mocker.patch(f"{path_of_sut_script}.preprocessing_image")
    mock_move_boxes.return_value = [Image.new("RGB", (10, 10))]

    processor_mock = mocker.patch(f"{path_of_sut_script}.load_processor")
    processor_mock.return_value = None

    mock_tokenize_image = mocker.patch(f"{path_of_sut_script}.tokenize_image")
    mock_tokenize_image.return_value = torch.tensor([[[1, 2, 3],[4, 5, 6],]])

    mock_prediction_ids = mocker.patch(f"{path_of_sut_script}.predict_chess_move")
    mock_prediction_ids.return_value = torch.tensor([[[1, 2, 3],[4, 5, 6],]])

    mock_prediction = mocker.patch(f"{path_of_sut_script}.decode_prediction")
    mock_prediction.return_value = "abc"

    mock_postprocessing_prediction = mocker.patch(f"{path_of_sut_script}.postprocessing_prediction")
    mock_postprocessing_prediction.return_value = "abc"

    # When
    result = inference_pipeline(str(image_file_path))

    # Then
    assert result == "abc"
    assert mock_image.call_count == 1
    assert mock_move_boxes.call_count == 1
    assert processor_mock.call_count == 1
    assert mock_tokenize_image.call_count == 1
    assert mock_prediction_ids.call_count == 1
    assert mock_prediction.call_count == 1
    assert mock_postprocessing_prediction.call_count == 1