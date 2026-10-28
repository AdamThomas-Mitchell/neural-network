from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from neural_network.data.idx import parser
from neural_network.data.idx.models import DataType, Header
from neural_network.errors import InvalidFileError, ParseError


DTYPE_CODES: dict[type[np.generic], tuple[int, DataType]] = {
    np.uint8: (0x08, DataType.u8),
    np.int8: (0x09, DataType.i8),
    np.int16: (0x0B, DataType.i16),
    np.int32: (0x0C, DataType.i32),
    np.float32: (0x0D, DataType.f32),
    np.float64: (0x0E, DataType.f64),
}
NUMPY_DTYPES = list(DTYPE_CODES)
ARRAY = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
LABELS = np.array([5, 0, 4, 1, 9], dtype=np.uint8)


def make_idx_bytes(array: NDArray) -> bytes:
    """Encode an array in IDX format (big-endian header and data)."""
    code, _ = DTYPE_CODES[array.dtype.type]
    header = b"\x00\x00" + bytes([code, array.ndim])
    header += b"".join(d.to_bytes(4, "big") for d in array.shape)
    return header + array.astype(array.dtype.newbyteorder(">")).tobytes()


def make_idx_file(filepath: Path, array: NDArray = ARRAY) -> Path:
    filepath.write_bytes(make_idx_bytes(array))
    return filepath


class TestParseHeader:
    def test_parses_header(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        assert parser._parse_header(idx) == Header(
            data_type=DataType.u8, n_dim=3, dim_sizes=[4, 3, 2]
        )

    def test_parses_one_dimensional_header(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx", LABELS)
        assert parser._parse_header(idx) == Header(
            data_type=DataType.u8, n_dim=1, dim_sizes=[5]
        )

    @pytest.mark.parametrize("dtype", NUMPY_DTYPES)
    def test_parses_data_type(self, tmp_path: Path, dtype):
        idx = make_idx_file(tmp_path / "a.idx", ARRAY.astype(dtype))
        assert parser._parse_header(idx).data_type == DTYPE_CODES[dtype][1]

    def test_large_dimension_size(self, tmp_path: Path):
        idx = tmp_path / "a.idx"
        idx.write_bytes(b"\x00\x00\x08\x01" + (60000).to_bytes(4, "big"))
        assert parser._parse_header(idx).dim_sizes == [60000]

    @pytest.mark.parametrize("magic", [b"\x00\x01", b"\x1f\x8b", b"\xff\xff"])
    def test_incorrect_magic_number_raises_invalid_file_error(
        self, tmp_path: Path, magic: bytes
    ):
        idx = tmp_path / "a.idx"
        idx.write_bytes(magic + make_idx_bytes(ARRAY)[2:])
        with pytest.raises(InvalidFileError, match="incorrect magic number"):
            parser._parse_header(idx)

    def test_unknown_data_type_raises_invalid_file_error(self, tmp_path: Path):
        idx = tmp_path / "a.idx"
        idx.write_bytes(b"\x00\x00\x01\x01" + (5).to_bytes(4, "big"))
        with pytest.raises(InvalidFileError):
            parser._parse_header(idx)

    @pytest.mark.parametrize(
        "content",
        [
            b"",
            b"\x00",
            b"\x00\x00",
            b"\x00\x00\x08",
            b"\x00\x00\x08\x03" + (4).to_bytes(4, "big"),  # missing dim sizes
            b"\x00\x00\x08\x01\x00\x00",  # truncated dim size
        ],
    )
    def test_truncated_header_raises_parse_error(self, tmp_path: Path, content):
        idx = tmp_path / "a.idx"
        idx.write_bytes(content)
        with pytest.raises(ParseError):
            parser._parse_header(idx)

    def test_missing_file_raises_file_not_found_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parser._parse_header(tmp_path / "missing.idx")


class TestLoadData:
    def test_loads_multi_dimensional_data(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        data = parser.load_data(idx)
        assert data.shape == (4, 3, 2)
        np.testing.assert_array_equal(data, ARRAY)

    def test_loads_one_dimensional_data(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx", LABELS)
        data = parser.load_data(idx)
        np.testing.assert_array_equal(data, LABELS)

    def test_loads_empty_dataset(self, tmp_path: Path):
        empty = np.empty((0, 3, 2), dtype=np.uint8)
        idx = make_idx_file(tmp_path / "a.idx", empty)
        assert parser.load_data(idx).shape == (0, 3, 2)

    @pytest.mark.parametrize("dtype", NUMPY_DTYPES)
    def test_round_trips_every_data_type(self, tmp_path: Path, dtype):
        expected = (ARRAY.astype(np.int64) - 12).astype(dtype)
        idx = make_idx_file(tmp_path / "a.idx", expected)
        data = parser.load_data(idx)
        assert data.dtype == dtype
        np.testing.assert_array_equal(data, expected)

    def test_invalid_file_raises_invalid_file_error(self, tmp_path: Path):
        idx = tmp_path / "a.idx"
        idx.write_bytes(b"\xff\xff\x08\x01" + (1).to_bytes(4, "big") + b"\x00")
        with pytest.raises(InvalidFileError):
            parser.load_data(idx)

    def test_read_error_raises_parse_error(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        header = parser._parse_header(idx)
        with (
            patch.object(parser, "_parse_header", return_value=header),
            patch("builtins.open", side_effect=OSError("boom")),
        ):
            with pytest.raises(ParseError):
                parser.load_data(idx)


class TestParseSample:
    @pytest.mark.parametrize("sample_idx", [0, 1, 2, 3])
    def test_returns_sample_at_zero_based_index(self, tmp_path: Path, sample_idx):
        idx = make_idx_file(tmp_path / "a.idx")
        sample = parser.parse_sample(idx, sample_idx)
        np.testing.assert_array_equal(sample, ARRAY[sample_idx])

    def test_sample_has_shape_of_remaining_dimensions(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        assert parser.parse_sample(idx, 0).shape == (3, 2)

    def test_returns_sample_from_one_dimensional_file(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx", LABELS)
        np.testing.assert_array_equal(parser.parse_sample(idx, 2), [LABELS[2]])

    @pytest.mark.parametrize("dtype", NUMPY_DTYPES)
    def test_returns_sample_for_every_data_type(self, tmp_path: Path, dtype):
        expected = (ARRAY.astype(np.int64) - 12).astype(dtype)
        idx = make_idx_file(tmp_path / "a.idx", expected)
        sample = parser.parse_sample(idx, 1)
        assert sample.dtype == dtype
        np.testing.assert_array_equal(sample, expected[1])

    def test_index_beyond_last_sample_raises_value_error(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        with pytest.raises(ValueError, match="out of range"):
            parser.parse_sample(idx, 4)

    @pytest.mark.parametrize("sample_idx", [-1, -5])
    def test_negative_index_raises_value_error(self, tmp_path: Path, sample_idx):
        idx = make_idx_file(tmp_path / "a.idx")
        with pytest.raises(ValueError, match="out of range"):
            parser.parse_sample(idx, sample_idx)

    def test_invalid_file_raises_invalid_file_error(self, tmp_path: Path):
        idx = tmp_path / "a.idx"
        idx.write_bytes(b"\xff\xff\x08\x01" + (1).to_bytes(4, "big") + b"\x00")
        with pytest.raises(InvalidFileError):
            parser.parse_sample(idx, 0)

    def test_read_error_raises_parse_error(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        header = parser._parse_header(idx)
        with (
            patch.object(parser, "_parse_header", return_value=header),
            patch("builtins.open", side_effect=OSError("boom")),
        ):
            with pytest.raises(ParseError):
                parser.parse_sample(idx, 0)
