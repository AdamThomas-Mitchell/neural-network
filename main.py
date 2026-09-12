from pathlib import Path

from neural_network.data import download_mnist_dataset


if __name__ == "__main__":
    download_mnist_dataset(
        output_dirpath=Path("data/MNIST/raw"),
        overwrite=False,
    )
