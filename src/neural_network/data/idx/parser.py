import math
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from neural_network.data.idx.models import (
    DataType,
    Header,
    byte_dtype_map,
    dtype_byte_length_map,
    dtype_ntype_map,
)
from neural_network.data.idx.reader import BinaryReader
from neural_network.errors import InvalidFileError, ReadError


MAGIC = b"\x00\x00"
DIM_SIZE_DTYPE = DataType.i32


def _parse_header(idx_file: Path) -> Header:
    """Read header of custom IDX file format.

    Args:
        idx_file (Path): Path to IDX file

    Raises:
        ReadError: If the file cannot be read, e.g. because it does not exist
        ParseError: If the header is truncated
        InvalidFileError: If the magic number or data type code is invalid

    Returns:
        Header: File header for given IDX file
    """
    try:
        with open(idx_file, "rb") as f:
            reader = BinaryReader(f, endian=">")

            if reader.read_exact(2) != MAGIC:
                logger.warning(f"IDX file {idx_file} has incorrect magic number")
                raise InvalidFileError(
                    f"IDX file {idx_file} has incorrect magic number"
                )

            data_type_code: bytes = reader.read_exact(1)
            if data_type_code not in byte_dtype_map:
                logger.warning(
                    f"IDX file {idx_file} has unknown data type code {data_type_code!r}"
                )
                raise InvalidFileError(
                    f"IDX file {idx_file} has unknown data type code {data_type_code!r}"
                )

            data_type: DataType = byte_dtype_map[data_type_code]
            n_dim: int = reader.u8()
            dim_sizes: list[int] = [reader.i32() for _ in range(n_dim)]
    except OSError as ex:
        logger.warning(f"Unable to read IDX file {idx_file}: {ex}")
        raise ReadError(f"Unable to read IDX file {idx_file}: {ex}") from ex

    return Header(
        data_type=data_type,
        n_dim=n_dim,
        dim_sizes=dim_sizes,
    )


def _read_file_bytes(idx_file: Path) -> bytes:
    """Read the full contents of an IDX file.

    Args:
        idx_file (Path): Path to IDX file

    Raises:
        ReadError: If the file cannot be read

    Returns:
        bytes: The raw contents of the file
    """
    try:
        with open(idx_file, "rb") as f:
            return f.read()
    except OSError as ex:
        logger.warning(f"Unable to read IDX file {idx_file}: {ex}")
        raise ReadError(f"Unable to read IDX file {idx_file}: {ex}") from ex


def parse_sample(idx_file: Path, sample_idx: int) -> NDArray:
    """Retrieve a given sample from an IDX file.

    Args:
        idx_file (Path): Path to IDX file
        sample_idx (int): Zero-based index of sample to retrieve

    Raises:
        ValueError: If given index is outside the range of samples in file
        ReadError: If the file cannot be read
        ParseError: If the header is truncated
        InvalidFileError: If the file is invalid or holds less data than its
            header describes

    Returns:
        NDArray: The sample at the given index
    """
    header: Header = _parse_header(idx_file)

    if not 0 <= sample_idx < header.dim_sizes[0]:
        logger.warning(
            f"Sample index {sample_idx} out of range - file contains {header.dim_sizes[0]} samples"
        )
        raise ValueError(
            f"Sample index {sample_idx} out of range - file contains {header.dim_sizes[0]} samples"
        )

    header_length: int = 4 + (header.n_dim * dtype_byte_length_map[DIM_SIZE_DTYPE])
    sample_length: int = math.prod(header.dim_sizes[1:])
    sample_bytes: int = sample_length * dtype_byte_length_map[header.data_type]

    bytes_to_skip = header_length + (sample_idx * sample_bytes)

    buffer: bytes = _read_file_bytes(idx_file)
    try:
        sample: NDArray = np.frombuffer(
            buffer=buffer,
            dtype=dtype_ntype_map[header.data_type],
            count=sample_length,
            offset=bytes_to_skip,
        )
        if header.n_dim > 1:
            sample = sample.reshape(*header.dim_sizes[1:])
    except ValueError as ex:
        logger.warning(f"IDX file {idx_file} does not match its header: {ex}")
        raise InvalidFileError(
            f"IDX file {idx_file} does not match its header: {ex}"
        ) from ex

    return sample.astype(sample.dtype.newbyteorder("="))


def load_data(idx_file: Path) -> NDArray:
    """Load the data in an IDX file into a NumPy array.

    Args:
        idx_file (Path): Path to IDX file

    Raises:
        ReadError: If the file cannot be read
        ParseError: If the header is truncated
        InvalidFileError: If the file is invalid or its data does not match the
            shape its header describes

    Returns:
        NDArray: The data in the file as an array
    """
    header: Header = _parse_header(idx_file)
    header_length: int = 4 + (header.n_dim * dtype_byte_length_map[DIM_SIZE_DTYPE])

    buffer: bytes = _read_file_bytes(idx_file)
    try:
        data: NDArray = np.frombuffer(
            buffer=buffer,
            dtype=dtype_ntype_map[header.data_type],
            count=-1,
            offset=header_length,
        ).reshape(*header.dim_sizes)
    except ValueError as ex:
        logger.warning(f"IDX file {idx_file} does not match its header: {ex}")
        raise InvalidFileError(
            f"IDX file {idx_file} does not match its header: {ex}"
        ) from ex

    return data.astype(data.dtype.newbyteorder("="))
