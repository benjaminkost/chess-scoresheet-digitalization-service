import torch
from src.api.clients.mlflow_model_client import predict, check_health

def test_predict_with_torch_tensor_return_prediction(mocker):
    # Arrange
    pixel_values = torch.tensor([[0, 0, 0], [1, 1, 1]])
    api_json_response = {"prediction": "abc"}
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.json.return_value = api_json_response

    # Act
    res = predict(pixel_values)

    # Assert
    mock_post.assert_called_once()
    assert res == api_json_response

def test_check_health_returns_ok(mocker):
    # Arrange
    mock_post = mocker.patch("requests.get")
    mock_post.return_value.status_code = 200

    # Act
    res = check_health()

    # Assert
    mock_post.assert_called_once()
    assert res == True
