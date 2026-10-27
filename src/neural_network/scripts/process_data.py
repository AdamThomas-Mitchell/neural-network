import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from neural_network.config.data import DatasetConfig
from neural_network.data.process import process_mnist_dataset
from neural_network.errors import DataError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_config_path",
        type=Path,
        help="Path to the dataset configuration file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether existing files should be overwritten.",
    )
    args = parser.parse_args()

    if not args.dataset_config_path.is_file():
        logger.error(f"{args.dataset_config_path} is not a valid file")
        sys.exit(1)

    try:
        with open(args.dataset_config_path) as f:
            yaml_contents = yaml.safe_load(f)

        dataset_config: DatasetConfig = DatasetConfig.model_validate(yaml_contents)

        process_mnist_dataset(
            dataset_config=dataset_config,
            overwrite=args.overwrite,
        )
    except (DataError, FileNotFoundError) as ex:
        logger.error(f"Failed to download dataset: {ex}")
        sys.exit(1)
    except (yaml.YAMLError, OSError) as ex:
        logger.error(f"Failed to parse YAML file {args.dataset_config_path}: {ex}")
        sys.exit(1)
    except ValidationError as ex:
        logger.error(
            f"Dataset configuration validation failed for {args.dataset_config_path}: {ex}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
