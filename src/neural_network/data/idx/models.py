from dataclasses import dataclass
from enum import StrEnum

from numpy.typing import NDArray


class DataType(StrEnum):
    """Allowed data types in custom IDX format."""

    u8 = "u8"
    i8 = "i8"
    i16 = "i16"
    i32 = "i32"
    f32 = "f32"
    f64 = "f64"


byte_dtype_mapping: dict[bytes, DataType] = {
    b"0x08": DataType.u8,
    b"0x09": DataType.i8,
    b"0x0B": DataType.i16,
    b"0x0C": DataType.i32,
    b"0x0D": DataType.f32,
    b"0x0E": DataType.f64,
}


@dataclass(frozen=True)
class Header:
    magic_1: int
    magic_2: int
    data_type: DataType
    n_dim: int
    dim_sizes: list[int]


@dataclass(frozen=True)
class IDXFile:
    header: Header
    data: NDArray
