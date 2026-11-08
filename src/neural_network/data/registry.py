from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[3]
DATASET_REGISTRY: dict[str, Path] = {
    "mnist": PROJECT_ROOT / "configs/datasets/mnist.yaml",
    "fashion-mnist": PROJECT_ROOT / "configs/datasets/fashion_mnist.yaml",
}
