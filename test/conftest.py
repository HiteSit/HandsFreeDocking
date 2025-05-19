import pytest
from pathlib import Path
import shutil

@pytest.fixture
def temp_workdir(tmp_path):
    workdir = tmp_path / "docking_run"
    workdir.mkdir()
    return workdir