from typing import BinaryIO, Literal


class BinaryReader:
    def __init__(self, stream: BinaryIO, endian: Literal["<", ">"]):
        self._f = stream
        self._endian = endian
        self._pos = 0

    def read_exact(self, n: int) -> bytes:
        return b"placeholder"

    def unpack(self, format: str) -> None:
        return None

    def u8(self) -> int:
        return 1

    def i8(self) -> int:
        return 1

    def i16(self) -> int:
        return 1

    def i32(self) -> int:
        return 1

    def f32(self) -> float:
        return 1.0

    def f64(self) -> float:
        return 1.0
