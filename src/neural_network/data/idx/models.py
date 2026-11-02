from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class DataType(StrEnum):
    """Allowed data types in custom IDX format."""

    u8 = "u8"
    i8 = "i8"
    i16 = "i16"
    i32 = "i32"
    f32 = "f32"
    f64 = "f64"


byte_dtype_map: dict[bytes, DataType] = {
    b"\x08": DataType.u8,
    b"\x09": DataType.i8,
    b"\x0b": DataType.i16,
    b"\x0c": DataType.i32,
    b"\x0d": DataType.f32,
    b"\x0e": DataType.f64,
}


dtype_byte_length_map: dict[DataType, int] = {
    DataType.u8: 1,
    DataType.i8: 1,
    DataType.i16: 2,
    DataType.i32: 4,
    DataType.f32: 4,
    DataType.f64: 8,
}


dtype_ntype_map = {
    DataType.u8: np.dtype(np.uint8).newbyteorder(">"),
    DataType.i8: np.dtype(np.int8).newbyteorder(">"),
    DataType.i16: np.dtype(np.int16).newbyteorder(">"),
    DataType.i32: np.dtype(np.int32).newbyteorder(">"),
    DataType.f32: np.dtype(np.float32).newbyteorder(">"),
    DataType.f64: np.dtype(np.float64).newbyteorder(">"),
}


@dataclass(frozen=True)
class Header:
    data_type: DataType
    n_dim: int
    dim_sizes: list[int]
