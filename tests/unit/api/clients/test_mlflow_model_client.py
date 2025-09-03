from pathlib import Path
from PIL import Image
from src.api.clients.mlflow_model_client import predict

def test_predict(mocker, get_data_path: Path):
    # Arrange
    image = Image.open(get_data_path / "images" / "001_0.png")
    api_json_response = {"prediction": "abc"}
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.json.return_value = api_json_response

    # Act
    res = predict(image)

    # Assert
    mock_post.assert_called_once()
    assert res == api_json_response
