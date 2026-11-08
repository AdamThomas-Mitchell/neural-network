import re
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from neural_network.config import loader
from neural_network.config.data import DatasetConfig
from neural_network.config.loader import load_dataset_config
from neural_network.errors import DatasetConfigError


VALID_YAML = """
name: test
mirrors:
  - https://mirror.example.com/
files:
  - label: images
    file_name: images.gz
    checksum: abc123
raw_data_loc: data/test/raw
processed_data_loc: data/test/processed
"""

MINIMAL_YAML = """
name: test
mirrors:
  - https://mirror.example.com/
files:
  - label: images
    file_name: images.gz
"""


def write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    return path


class TestLoadDatasetConfig:
    def test_loads_valid_config(self, tmp_path: Path):
        config = load_dataset_config(write_config(tmp_path, VALID_YAML))

        assert isinstance(config, DatasetConfig)
        assert config.name == "test"
        assert [str(m) for m in config.mirrors] == ["https://mirror.example.com/"]
        assert config.files[0].label == "images"
        assert config.files[0].file_name == "images.gz"
        assert config.files[0].checksum == "abc123"
        assert config.raw_data_loc == Path("data/test/raw")
        assert config.processed_data_loc == Path("data/test/processed")

    def test_optional_fields_use_defaults(self, tmp_path: Path):
        config = load_dataset_config(write_config(tmp_path, MINIMAL_YAML))

        assert config.files[0].checksum is None
        assert config.raw_data_loc == DatasetConfig.model_fields["raw_data_loc"].default
        assert (
            config.processed_data_loc
            == DatasetConfig.model_fields["processed_data_loc"].default
        )

    def test_data_locations_need_not_exist(self, tmp_path: Path):
        missing_dir = tmp_path / "nope"
        contents = MINIMAL_YAML + (
            f"raw_data_loc: {missing_dir / 'raw'}\n"
            f"processed_data_loc: {missing_dir / 'processed'}\n"
        )

        config = load_dataset_config(write_config(tmp_path, contents))

        assert config.raw_data_loc == missing_dir / "raw"
        assert config.processed_data_loc == missing_dir / "processed"
        assert not missing_dir.exists()

    def test_missing_file_raises(self, tmp_path: Path):
        missing = tmp_path / "missing.yaml"

        with pytest.raises(DatasetConfigError, match=re.escape(str(missing))) as exc:
            load_dataset_config(missing)

        assert isinstance(exc.value.__cause__, FileNotFoundError)

    def test_directory_raises(self, tmp_path: Path):
        with pytest.raises(DatasetConfigError, match=re.escape(str(tmp_path))) as exc:
            load_dataset_config(tmp_path)

        assert isinstance(exc.value.__cause__, OSError)

    def test_unreadable_file_raises(self, tmp_path: Path):
        config_path = write_config(tmp_path, VALID_YAML)

        with patch("builtins.open", side_effect=PermissionError("denied")):
            with pytest.raises(DatasetConfigError, match="denied") as exc:
                load_dataset_config(config_path)

        assert isinstance(exc.value.__cause__, PermissionError)

    def test_invalid_yaml_raises(self, tmp_path: Path):
        config_path = write_config(tmp_path, "name: [unclosed")

        with pytest.raises(DatasetConfigError) as exc:
            load_dataset_config(config_path)

        assert isinstance(exc.value.__cause__, yaml.YAMLError)

    @pytest.mark.parametrize(
        "contents",
        [
            "",  # empty file
            "name: test\n",  # missing mirrors and files
            "name: test\nmirrors: []\nfiles: []\n",  # empty lists
            "name: test\nmirrors: [not-a-url]\nfiles:\n  - {label: a, file_name: a}\n",
            "name: test\nmirrors: [https://x.com/]\nfiles:\n  - {label: a}\n",  # no file_name
            "- just\n- a\n- list\n",  # wrong top-level type
        ],
    )
    def test_invalid_schema_raises(self, tmp_path: Path, contents: str):
        config_path = write_config(tmp_path, contents)

        with pytest.raises(
            DatasetConfigError, match=re.escape(str(config_path))
        ) as exc:
            load_dataset_config(config_path)

        assert isinstance(exc.value.__cause__, ValidationError)

    def test_logs_warning_before_raising(self, tmp_path: Path):
        with patch.object(loader, "logger") as mock_logger:
            with pytest.raises(DatasetConfigError):
                load_dataset_config(tmp_path / "missing.yaml")

        mock_logger.warning.assert_called_once()
