from pathlib import Path

from neural_network.data.idx.models import (
    DataType,
    Header,
    IDXFile,
    byte_dtype_mapping,
)
from neural_network.data.idx.reader import BinaryReader


def read_header(idx_file: Path) -> Header:
    # TODO: validate magic numbers
    # TODO: error handing

    with open(idx_file, "rb") as f:
        reader = BinaryReader(f, endian=">")

        magic_1: int = reader.u8()
        magic_2: int = reader.u8()
        data_type: DataType = byte_dtype_mapping[reader.read_exact(1)]
        n_dim: int = reader.u8()
        dim_sizes: list[int] = [reader.i32() for _ in range(n_dim)]

    header = Header(
        magic_1=magic_1,
        magic_2=magic_2,
        data_type=data_type,
        n_dim=n_dim,
        dim_sizes=dim_sizes,
    )

    return header


def load_data(idx_file: Path):
    pass


def iter_data(idx_file: Path):
    pass


def load_file(idx_file: Path) -> IDXFile:
    pass
