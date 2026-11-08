import argparse
import sys

from loguru import logger

from neural_network.config.loader import load_dataset_config
from neural_network.data.download import download_dataset
from neural_network.data.registry import DATASET_REGISTRY
from neural_network.errors import DataError


def main() -> None:
    """Run the ``download-data`` command to download a dataset's raw files.

    Raises:
        SystemExit: With status 1 if the config cannot be loaded or the download
            fails, or status 2 if the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download a dataset's raw files from the mirrors listed in its config. "
            "Files are saved to the config's raw_data_loc."
        ),
        epilog="Example: download-data mnist",
    )
    parser.add_argument(
        "dataset_name",
        choices=list(DATASET_REGISTRY),
        help="Name of the dataset to download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download files again even if they already exist.",
    )
    args = parser.parse_args()

    dataset_config_file = DATASET_REGISTRY[args.dataset_name]

    try:
        dataset_config = load_dataset_config(dataset_config_file)

        download_dataset(
            dataset_config=dataset_config,
            overwrite=args.overwrite,
        )
    except (DataError, FileNotFoundError) as ex:
        logger.error(f"Failed to download dataset: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()
