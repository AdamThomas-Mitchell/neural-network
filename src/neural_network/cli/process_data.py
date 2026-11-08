import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from neural_network.config.data import DatasetConfig
from neural_network.data.process import process_mnist_dataset
from neural_network.data.registry import DATASET_REGISTRY
from neural_network.errors import DataError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=[*DATASET_REGISTRY.keys()],
        help="Name of the dataset to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether existing files should be overwritten.",
    )
    args = parser.parse_args()

    dataset_config_file: Path = DATASET_REGISTRY[args.dataset_name]

    if not dataset_config_file.is_file():
        logger.error(f"{dataset_config_file} is not a valid file")
        sys.exit(1)

    try:
        with open(dataset_config_file) as f:
            yaml_contents = yaml.safe_load(f)

        dataset_config: DatasetConfig = DatasetConfig.model_validate(yaml_contents)

        process_mnist_dataset(
            dataset_config=dataset_config,
            overwrite=args.overwrite,
        )
    except (DataError, FileNotFoundError) as ex:
        logger.error(f"Failed to process dataset: {ex}")
        sys.exit(1)
    except (yaml.YAMLError, OSError) as ex:
        logger.error(f"Failed to parse YAML file {dataset_config_file}: {ex}")
        sys.exit(1)
    except ValidationError as ex:
        logger.error(
            f"Dataset configuration validation failed for {dataset_config_file}: {ex}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
