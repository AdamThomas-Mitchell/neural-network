from dataclasses import dataclass


@dataclass(frozen=True)
class Header:
    magic: bytes
    data_type: None
    n_dim: int


@dataclass(frozen=True)
class Record:
    pass


@dataclass(frozen=True)
class IDXFile:
    header: Header
    dim_size: list[int]
    data: list[Record]
