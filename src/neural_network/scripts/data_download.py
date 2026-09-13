import argparse
from pathlib import Path

from neural_network.data import download_mnist_dataset


def main() -> None:
    """Main function to download the MNIST dataset."""

    parser = argparse.ArgumentParser(
        prog="neural_network", description="Download the MNIST dataset"
    )
    parser.add_argument("output_dirpath")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dirpath: Path = Path(args.output_dirpath)
    overwrite: bool = args.overwrite

    download_mnist_dataset(output_dirpath, overwrite=overwrite)


if __name__ == "__main__":
    main()
