import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

from neural_network.config.data import DatasetConfig
from neural_network.data import download_dataset
from neural_network.errors import DataError


def main() -> None:
    """Main function to download the dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_config_path",
        help="Path to the dataset configuration file.",
    )
    parser.add_argument(
        "output_dirpath",
        help="Path to local directory where files will be downloaded.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether existing files should be overwritten.",
    )
    args = parser.parse_args()

    dataset_config_path = Path(args.dataset_config_path)
    if not dataset_config_path.exists() or not dataset_config_path.is_file():
        logger.error(f"{dataset_config_path} is not a valid file")
        raise FileNotFoundError

    with open(dataset_config_path) as f:
        yaml_contents = yaml.safe_load(f)
    dataset_config: DatasetConfig = DatasetConfig.model_validate(yaml_contents)
    output_dirpath: Path = Path(args.output_dirpath)
    overwrite: bool = args.overwrite

    try:
        download_dataset(
            dataset_config=dataset_config,
            output_dirpath=output_dirpath,
            overwrite=overwrite,
        )
    except DataError, NotADirectoryError, FileNotFoundError:
        sys.exit(1)


if __name__ == "__main__":
    main()
