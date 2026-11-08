from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from neural_network.config.data import DatasetConfig
from neural_network.errors import DatasetConfigError


def load_dataset_config(config_path: Path) -> DatasetConfig:

    try:
        with open(config_path) as f:
            contents = yaml.safe_load(f)

        return DatasetConfig.model_validate(contents)

    except (OSError, yaml.YAMLError, ValidationError) as ex:
        logger.warning(f"Invalid dataset config: {config_path}: {ex}")
        raise DatasetConfigError(f"Invalid dataset config: {config_path}: {ex}") from ex
