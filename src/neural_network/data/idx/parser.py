import math
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from neural_network.data.idx.models import (
    DIM_SIZE_DTYPE,
    MAGIC,
    DataType,
    Header,
    byte_dtype_map,
    dtype_byte_length_map,
    dtype_ntype_map,
)
from neural_network.data.idx.reader import BinaryReader
from neural_network.errors import ParseError


def _parse_header(idx_file: Path) -> Header:
    """Read header of custom IDX file format.

    Args:
        idx_file (Path): Path to IDX file

    Raises:
        ParseError: If magic number is incorrect

    Returns:
        Header: File header for given IDX file
    """
    with open(idx_file, "rb") as f:
        reader = BinaryReader(f, endian=">")

        if reader.read_exact(2) != MAGIC:
            logger.error(f"IDX file {idx_file} has incorrect magic number")
            raise ParseError(f"IDX file {idx_file} has incorrect magic number")

        data_type: DataType = byte_dtype_map[reader.read_exact(1)]
        n_dim: int = reader.u8()
        dim_sizes: list[int] = [reader.i32() for _ in range(n_dim)]

    header = Header(
        data_type=data_type,
        n_dim=n_dim,
        dim_sizes=dim_sizes,
    )

    return header


def parse_sample(idx_file: Path, sample_idx: int) -> NDArray:
    """Retrieve a given sample from an IDX file.

    Args:
        idx_file (Path): Path to IDX file
        sample_idx (int): Index of sample to retreive

    Raises:
        ValueError: If given index is greater than number of samples in file

    Returns:
        NDArray: The sample at the given index
    """
    header: Header = _parse_header(idx_file)

    if sample_idx > header.dim_sizes[0]:
        raise ValueError(
            f"Sample index {sample_idx} out of range - file contains {header.dim_sizes[0]} samples"
        )

    header_length: int = 4 + (header.n_dim * dtype_byte_length_map[DIM_SIZE_DTYPE])
    sample_length: int = dtype_byte_length_map[header.data_type] * math.prod(
        header.dim_sizes[1:]
    )

    bytes_to_skip = header_length + ((sample_idx - 1) * sample_length)

    with open(idx_file, "rb") as f:
        sample: NDArray = np.frombuffer(
            buffer=f.read(),
            dtype=dtype_ntype_map[header.data_type],
            count=sample_length,
            offset=bytes_to_skip,
        )

    if header.n_dim > 1:
        sample: NDArray = sample.reshape(*header.dim_sizes[1:])

    return sample


def load_data(idx_file: Path) -> NDArray:
    """Load the data in an IDX file into a NumPy array.

    Args:
        idx_file (Path): Path to IDX file

    Returns:
        NDArray: The data in the file as an array
    """
    header: Header = _parse_header(idx_file)
    header_length: int = 4 + (header.n_dim * dtype_byte_length_map[DIM_SIZE_DTYPE])

    with open(idx_file, "rb") as f:
        data: NDArray = np.frombuffer(
            buffer=f.read(),
            dtype=dtype_ntype_map[header.data_type],
            count=-1,
            offset=header_length,
        ).reshape(*header.dim_sizes)

    return data
