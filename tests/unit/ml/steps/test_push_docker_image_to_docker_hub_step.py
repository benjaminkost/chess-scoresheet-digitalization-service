from src.ml.steps.push_docker_image_to_docker_hub_step import push_docker_image_to_docker_hub


def test_push_docker_image_to_docker_hub_returns_true(mocker):
    mock_subprocess = mocker.patch("subprocess.run")
    mock_strip = mocker.Mock()
    mock_stdout = mocker.Mock()
    mock_strip.strip.return_value = "test"
    mock_stdout.return_value = mock_strip
    mock_subprocess.return_value = mock_stdout

    result = push_docker_image_to_docker_hub()

    assert result == True

def test_push_docker_image_to_docker_hub_returns_false(mocker):
    mocker.patch("subprocess.run", side_effect=Exception)

    result = push_docker_image_to_docker_hub()

    assert result == False
