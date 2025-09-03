from pathlib import Path
import pytest

@pytest.fixture
def get_data_path() -> Path:
    return Path(__file__).parent / "data"