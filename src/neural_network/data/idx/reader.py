import struct
from typing import BinaryIO, Literal

from loguru import logger

from neural_network.errors import ParseError


class BinaryReader:
    def __init__(self, stream: BinaryIO, endian: Literal["<", ">"]):
        self._f = stream
        self._endian = endian
        self._pos = 0

    def read_exact(self, n: int) -> bytes:
        """Retrieve a given number of bytes from a binary stream.

        Args:
            n (int): Number of bytes to retrieve

        Raises:
            ValueError: If given negative number of bytes to read
            ParseError: If unable to read desired number of bytes

        Returns:
            bytes: The n raw bytes from the binary stream
        """
        if n < 0:
            raise ValueError(f"Cannot read a negative number of bytes: {n}")

        content: bytes = self._f.read(n)
        if len(content) != n:
            raise ParseError(f"Tried to read {n} bytes, only got {len(content)}")

        self._pos += n
        return content

    def unpack(self, format: str) -> tuple:
        """Read exactly one element from binary stream in the given format.

        Args:
            format (str): The format to unpack the next element in

        Returns:
            tuple: The next formatted element from the binary stream
        """
        # TODO: this will be inefficient to repeatedly create the same struct multiple times - look at caching
        try:
            s = struct.Struct(self._endian + format)
            buffer = self.read_exact(s.size)
            element = s.unpack(buffer)
            return element
        except struct.error as ex:
            logger.warning("Unable to unpack element from binary stream")
            raise ParseError("Unable to unpack element from binary stream") from ex

    def u8(self) -> int:
        return self.unpack("B")[0]

    def i8(self) -> int:
        return self.unpack("b")[0]

    def i16(self) -> int:
        return self.unpack("h")[0]

    def i32(self) -> int:
        return self.unpack("i")[0]

    def f32(self) -> float:
        return self.unpack("f")[0]

    def f64(self) -> float:
        return self.unpack("d")[0]
