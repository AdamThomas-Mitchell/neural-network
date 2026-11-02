import io

import pytest

from neural_network.data.idx.reader import BinaryReader
from neural_network.errors import ParseError


def make_reader(content: bytes, endian: str = ">") -> BinaryReader:
    return BinaryReader(io.BytesIO(content), endian=endian)


class TestReadExact:
    def test_reads_requested_bytes(self):
        reader = make_reader(b"abcdef")
        assert reader.read_exact(3) == b"abc"

    def test_successive_reads_advance_through_stream(self):
        reader = make_reader(b"abcdef")
        reader.read_exact(2)
        assert reader.read_exact(4) == b"cdef"

    def test_zero_bytes_returns_empty(self):
        reader = make_reader(b"abc")
        assert reader.read_exact(0) == b""

    def test_negative_bytes_raises_value_error(self):
        reader = make_reader(b"abc")
        with pytest.raises(ValueError, match="negative"):
            reader.read_exact(-1)

    def test_short_read_raises_parse_error(self):
        reader = make_reader(b"abc")
        with pytest.raises(ParseError, match="only got 3"):
            reader.read_exact(4)

    def test_empty_stream_raises_parse_error(self):
        reader = make_reader(b"")
        with pytest.raises(ParseError):
            reader.read_exact(1)


class TestUnpack:
    def test_unpacks_using_reader_endianness(self):
        assert make_reader(b"\x00\x01", endian=">").unpack("H") == (1,)
        assert make_reader(b"\x00\x01", endian="<").unpack("H") == (256,)

    def test_unpacks_multiple_elements(self):
        reader = make_reader(b"\x01\x02")
        assert reader.unpack("BB") == (1, 2)

    def test_invalid_format_raises_parse_error(self):
        reader = make_reader(b"\x00\x00\x00\x00")
        with pytest.raises(ParseError):
            reader.unpack("z")

    def test_insufficient_data_raises_parse_error(self):
        reader = make_reader(b"\x00\x00")
        with pytest.raises(ParseError):
            reader.unpack("i")


class TestU8:
    @pytest.mark.parametrize(
        ("content", "expected"), [(b"\x00", 0), (b"\x7f", 127), (b"\xff", 255)]
    )
    def test_reads_unsigned_byte(self, content: bytes, expected: int):
        assert make_reader(content).u8() == expected

    def test_empty_stream_raises_parse_error(self):
        with pytest.raises(ParseError):
            make_reader(b"").u8()


class TestI32:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (b"\x00\x00\xea\x60", 60000),
            (b"\x00\x00\x00\x00", 0),
            (b"\xff\xff\xff\xff", -1),
            (b"\x7f\xff\xff\xff", 2**31 - 1),
        ],
    )
    def test_reads_big_endian_signed_int(self, content: bytes, expected: int):
        assert make_reader(content, endian=">").i32() == expected

    def test_reads_little_endian_signed_int(self):
        assert make_reader(b"\x60\xea\x00\x00", endian="<").i32() == 60000

    def test_successive_reads(self):
        reader = make_reader(b"\x00\x00\x00\x01\x00\x00\x00\x02")
        assert [reader.i32(), reader.i32()] == [1, 2]

    def test_short_stream_raises_parse_error(self):
        with pytest.raises(ParseError):
            make_reader(b"\x00\x00\x01").i32()
