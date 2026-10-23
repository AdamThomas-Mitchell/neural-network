import gzip
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from neural_network.config.data import DatasetConfig, FileInfo
from neural_network.data import process
from neural_network.errors import (
    InvalidFileError,
    ReadError,
    WriteError,
)


CONTENT = b"hello world"
ARRAY = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)


def make_gzip_file(filepath: Path, content: bytes = CONTENT) -> Path:
    """Create a dummy gzip archive."""
    with gzip.open(filepath, "wb") as f:
        f.write(content)
    return filepath


def make_idx_bytes(array: NDArray[np.uint8] = ARRAY) -> bytes:
    """Encode a uint8 array in IDX format."""
    header = b"\x00\x00\x08" + bytes([array.ndim])
    header += b"".join(d.to_bytes(4, "big") for d in array.shape)
    return header + array.tobytes()


def make_idx_file(filepath: Path, array: NDArray[np.uint8] = ARRAY) -> Path:
    filepath.write_bytes(make_idx_bytes(array))
    return filepath


def make_config(
    raw_data_loc: Path, processed_data_loc: Path, labels: list[str]
) -> DatasetConfig:
    return DatasetConfig(
        name="test",
        mirrors=["https://mirror.example.com/"],
        files=[FileInfo(label=label, file_name=f"{label}.gz") for label in labels],
        raw_data_loc=raw_data_loc,
        processed_data_loc=processed_data_loc,
    )


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    path = tmp_path / "raw"
    path.mkdir()
    return path


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    path = tmp_path / "processed"
    path.mkdir()
    return path


class TestIsValidGzip:
    def test_gzip_file_is_valid(self, tmp_path: Path):
        f = make_gzip_file(tmp_path / "a.gz")
        assert process._is_valid_gzip(f) is True

    def test_plain_file_is_not_valid(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        assert process._is_valid_gzip(f) is False

    def test_missing_file_is_not_valid(self, tmp_path: Path):
        assert process._is_valid_gzip(tmp_path / "missing.gz") is False

    def test_directory_is_not_valid(self, tmp_path: Path):
        assert process._is_valid_gzip(tmp_path) is False

    def test_empty_file_is_not_valid(self, tmp_path: Path):
        f = tmp_path / "empty.gz"
        f.write_bytes(b"")
        assert process._is_valid_gzip(f) is False

    def test_truncated_magic_number_is_not_valid(self, tmp_path: Path):
        f = tmp_path / "a.gz"
        f.write_bytes(b"\x1f")
        assert process._is_valid_gzip(f) is False

    def test_magic_number_alone_is_valid(self, tmp_path: Path):
        f = tmp_path / "a.gz"
        f.write_bytes(b"\x1f\x8b")
        assert process._is_valid_gzip(f) is True

    def test_read_error_raises_read_error(self, tmp_path: Path):
        f = make_gzip_file(tmp_path / "a.gz")
        with patch("builtins.open", side_effect=OSError("boom")):
            with pytest.raises(ReadError):
                process._is_valid_gzip(f)


class TestUnzipGzipFile:
    def test_unzips_file(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz")
        out = tmp_path / "a.bin"
        process._unzip_gzip_file(zipped, out)
        assert out.read_bytes() == CONTENT

    def test_unzips_empty_file(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "empty.gz", b"")
        out = tmp_path / "empty.bin"
        process._unzip_gzip_file(zipped, out)
        assert out.read_bytes() == b""

    def test_does_not_overwrite_existing_output_file(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz")
        out = tmp_path / "a.bin"
        out.write_bytes(b"old content that is longer than the new content")
        process._unzip_gzip_file(zipped, out, overwrite=False)
        assert out.read_bytes() == b"old content that is longer than the new content"

    def test_overwrites_existing_output_file(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz")
        out = tmp_path / "a.bin"
        out.write_bytes(b"old content that is longer than the new content")
        process._unzip_gzip_file(zipped, out, overwrite=True)
        assert out.read_bytes() == CONTENT

    def test_non_gzip_file_raises_invalid_file_error(self, tmp_path: Path):
        zipped = tmp_path / "a.gz"
        zipped.write_bytes(CONTENT)
        out = tmp_path / "a.bin"
        with pytest.raises(InvalidFileError, match="not a valid gzip file"):
            process._unzip_gzip_file(zipped, out)
        assert not out.exists()

    def test_missing_file_raises_invalid_file_error(self, tmp_path: Path):
        with pytest.raises(InvalidFileError, match="not a valid gzip file"):
            process._unzip_gzip_file(tmp_path / "missing.gz", tmp_path / "a.bin")

    def test_bad_gzip_body_raises_read_error_and_cleans_up(self, tmp_path: Path):
        zipped = tmp_path / "a.gz"
        zipped.write_bytes(b"\x1f\x8b" + b"not really gzipped")
        out = tmp_path / "a.bin"
        with pytest.raises(ReadError):
            process._unzip_gzip_file(zipped, out)
        assert not out.exists()

    def test_unwritable_output_path_raises_write_error(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz")
        out = tmp_path / "missing_dir" / "a.bin"
        with pytest.raises(WriteError):
            process._unzip_gzip_file(zipped, out)

    def test_write_error_raises_write_error_and_cleans_up(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz")
        out = tmp_path / "a.bin"
        with patch.object(
            process.shutil, "copyfileobj", side_effect=OSError("disk full")
        ):
            with pytest.raises(WriteError):
                process._unzip_gzip_file(zipped, out)
        assert not out.exists()


class TestConvertIdxToNp:
    def test_converts_idx_to_numpy_file(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        out = tmp_path / "a.npy"
        process._convert_idx_to_np(idx, out)
        np.testing.assert_array_equal(np.load(out), ARRAY)

    def test_does_not_overwrite_existing_output_file(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        out = tmp_path / "a.npy"
        out.write_bytes(b"old")
        with patch.object(process, "load_data") as mock_load:
            process._convert_idx_to_np(idx, out, overwrite=False)
        mock_load.assert_not_called()
        assert out.read_bytes() == b"old"

    def test_overwrites_existing_output_file(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        out = tmp_path / "a.npy"
        out.write_bytes(b"old")
        process._convert_idx_to_np(idx, out, overwrite=True)
        np.testing.assert_array_equal(np.load(out), ARRAY)

    def test_invalid_idx_raises_invalid_file_error(self, tmp_path: Path):
        idx = tmp_path / "a.idx"
        idx.write_bytes(b"\xff\xff\x08\x01")
        out = tmp_path / "a.npy"
        with pytest.raises(InvalidFileError):
            process._convert_idx_to_np(idx, out)
        assert not out.exists()

    def test_write_error_raises_write_error_and_cleans_up(self, tmp_path: Path):
        idx = make_idx_file(tmp_path / "a.idx")
        out = tmp_path / "a.npy"
        with patch.object(process.np, "save", side_effect=OSError("disk full")):
            with pytest.raises(WriteError):
                process._convert_idx_to_np(idx, out)
        assert not out.exists()


class TestProcessMnistFile:
    def test_unzips_and_converts_file(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz", make_idx_bytes())
        out = tmp_path / "a.npy"
        process._process_mnist_file(zipped, out)
        np.testing.assert_array_equal(np.load(out), ARRAY)

    def test_does_not_leave_intermediate_files(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz", make_idx_bytes())
        out = tmp_path / "a.npy"
        process._process_mnist_file(zipped, out)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a.gz", "a.npy"]

    def test_invalid_gzip_raises_invalid_file_error(self, tmp_path: Path):
        not_zipped = make_idx_file(tmp_path / "a.gz")
        with pytest.raises(InvalidFileError, match="not a valid gzip file"):
            process._process_mnist_file(not_zipped, tmp_path / "a.npy")


class TestProcessMnistDataset:
    def test_processes_every_file(self, raw_dir: Path, processed_dir: Path):
        config = make_config(raw_dir, processed_dir, ["images", "labels"])
        for f in config.files:
            (raw_dir / f.file_name).touch()

        with patch.object(process, "_process_mnist_file") as mock:
            process.process_mnist_dataset(config)

        assert mock.call_args_list == [
            call(raw_dir / "images.gz", processed_dir / "images.npy"),
            call(raw_dir / "labels.gz", processed_dir / "labels.npy"),
        ]

    def test_finds_raw_files_regardless_of_directory_order(
        self, raw_dir: Path, processed_dir: Path
    ):
        config = make_config(raw_dir, processed_dir, ["images", "labels"])
        for f in config.files:
            (raw_dir / f.file_name).touch()
        reversed_listing = [raw_dir / "labels.gz", raw_dir / "images.gz"]

        with (
            patch.object(Path, "iterdir", return_value=iter(reversed_listing)),
            patch.object(process, "_process_mnist_file") as mock,
        ):
            process.process_mnist_dataset(config)

        assert mock.call_count == 2

    def test_missing_raw_files_raises(self, raw_dir: Path, processed_dir: Path):
        config = make_config(raw_dir, processed_dir, ["images", "labels"])
        (raw_dir / "images.gz").touch()

        with patch.object(process, "_process_mnist_file") as mock:
            with pytest.raises(FileNotFoundError, match="labels.gz") as exc_info:
                process.process_mnist_dataset(config)

        assert "images.gz" not in str(exc_info.value)
        mock.assert_not_called()

    def test_skips_existing_output_without_overwrite(
        self, raw_dir: Path, processed_dir: Path
    ):
        config = make_config(raw_dir, processed_dir, ["images", "labels"])
        for f in config.files:
            (raw_dir / f.file_name).touch()
        (processed_dir / "images.npy").touch()

        with patch.object(process, "_process_mnist_file") as mock:
            process.process_mnist_dataset(config)

        mock.assert_called_once_with(
            raw_dir / "labels.gz", processed_dir / "labels.npy"
        )

    def test_reprocesses_existing_output_with_overwrite(
        self, raw_dir: Path, processed_dir: Path
    ):
        config = make_config(raw_dir, processed_dir, ["images"])
        (raw_dir / "images.gz").touch()
        (processed_dir / "images.npy").touch()

        with patch.object(process, "_process_mnist_file") as mock:
            process.process_mnist_dataset(config, overwrite=True)

        mock.assert_called_once_with(
            raw_dir / "images.gz", processed_dir / "images.npy"
        )

    def test_end_to_end(self, raw_dir: Path, processed_dir: Path):
        config = make_config(raw_dir, processed_dir, ["images"])
        make_gzip_file(raw_dir / "images.gz", make_idx_bytes())

        process.process_mnist_dataset(config)

        np.testing.assert_array_equal(np.load(processed_dir / "images.npy"), ARRAY)

    def test_rerun_without_overwrite_skips_processed_files(
        self, raw_dir: Path, processed_dir: Path
    ):
        config = make_config(raw_dir, processed_dir, ["images"])
        make_gzip_file(raw_dir / "images.gz", make_idx_bytes())
        process.process_mnist_dataset(config)

        with patch.object(process, "_process_mnist_file") as mock:
            process.process_mnist_dataset(config)

        mock.assert_not_called()
