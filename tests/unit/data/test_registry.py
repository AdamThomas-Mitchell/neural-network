from pathlib import Path

import pytest
import yaml

from neural_network.config.data import DatasetConfig
from neural_network.data.registry import DATASET_REGISTRY


class TestDatasetRegistry:
    @pytest.mark.parametrize("config_path", DATASET_REGISTRY.values())
    def test_entry_points_to_valid_config(self, config_path: Path):
        with open(config_path) as f:
            yaml_contents = yaml.safe_load(f)

        DatasetConfig.model_validate(yaml_contents)

    @pytest.mark.parametrize("config_path", DATASET_REGISTRY.values())
    def test_entry_does_not_depend_on_working_directory(self, config_path: Path):
        assert config_path.is_absolute()
