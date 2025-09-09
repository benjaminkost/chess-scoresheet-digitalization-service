from pathlib import Path
import pytest
from utils.config import get_root_dir_path

@pytest.fixture
def get_data_path() -> Path:
    return get_root_dir_path() / "tests" / "data"