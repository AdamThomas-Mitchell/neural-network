import argparse
import sys

from loguru import logger

from neural_network.config.loader import load_dataset_config
from neural_network.data.process import process_mnist_dataset
from neural_network.data.registry import DATASET_REGISTRY
from neural_network.errors import DataError


def main() -> None:
    """Run the ``process-data`` command to convert a dataset's raw files to NumPy.

    Raises:
        SystemExit: With status 1 if the config cannot be loaded or processing
            fails, or status 2 if the command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert a dataset's raw gzipped IDX files to NumPy .npy files, reading "
            "from the config's raw_data_loc and writing to its processed_data_loc."
        ),
        epilog="Run download-data first. Example: process-data mnist",
    )
    parser.add_argument(
        "dataset_name",
        choices=list(DATASET_REGISTRY),
        help="Name of the dataset to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Process files again even if the processed files already exist.",
    )
    args = parser.parse_args()

    dataset_config_file = DATASET_REGISTRY[args.dataset_name]

    try:
        dataset_config = load_dataset_config(dataset_config_file)

        process_mnist_dataset(
            dataset_config=dataset_config,
            overwrite=args.overwrite,
        )
    except DataError as ex:
        logger.error(f"Failed to process dataset: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()
