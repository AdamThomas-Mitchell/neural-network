import numpy as np
from numpy.typing import NDArray


def relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x.clip(0)


def leaky_relu() -> None:
    pass


def sigmoid() -> None:
    pass


def hyperbolic_tangent() -> None:
    pass


def softmax() -> None:
    pass
