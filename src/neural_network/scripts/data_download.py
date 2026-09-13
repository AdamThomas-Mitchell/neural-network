import argparse
import sys
from pathlib import Path

from neural_network.data import download_mnist_dataset
from neural_network.errors import DataError


def main() -> None:
    """Main function to download the MNIST dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dirpath",
        help="Path to local directory where MNIST files will be downloaded.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether existing MNIST files should be overwritten.",
    )
    args = parser.parse_args()

    output_dirpath: Path = Path(args.output_dirpath)
    overwrite: bool = args.overwrite

    try:
        download_mnist_dataset(output_dirpath, overwrite=overwrite)
    except DataError, NotADirectoryError, FileNotFoundError:
        sys.exit(1)


if __name__ == "__main__":
    main()
