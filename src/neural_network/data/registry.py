from pathlib import Path


dataset_registry: dict[str, Path] = {
    "mnist": Path("configs/datasets/mnist.yaml"),
    "fashion-mnist": Path("configs/datasets/fashion_mnist.yaml"),
}
