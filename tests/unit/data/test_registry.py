from pathlib import Path

import pytest

from neural_network.config.loader import load_dataset_config
from neural_network.data.registry import DATASET_REGISTRY


class TestDatasetRegistry:
    @pytest.mark.parametrize("config_path", DATASET_REGISTRY.values())
    def test_entry_points_to_valid_config(self, config_path: Path):
        load_dataset_config(config_path)

    @pytest.mark.parametrize("config_path", DATASET_REGISTRY.values())
    def test_entry_does_not_depend_on_working_directory(self, config_path: Path):
        assert config_path.is_absolute()
