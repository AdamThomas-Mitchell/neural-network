import gzip
import hashlib
import io
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import pytest

from neural_network import data
from neural_network.config.data import DatasetConfig, FileInfo
from neural_network.errors import (
    DeleteError,
    DownloadError,
    ReadError,
    WriteError,
)


CONTENT = b"hello world"
CONTENT_SHA256 = hashlib.sha256(CONTENT).hexdigest()


class FakeResponse(io.BytesIO):
    """Minimal stand-in for the object returned by urlopen."""

    def __init__(self, content: bytes, content_length: int | None = None) -> None:
        super().__init__(content)
        self._content_length = content_length

    def getheader(self, name: str) -> str | None:
        if name == "Content-Length" and self._content_length is not None:
            return str(self._content_length)
        return None


def make_response(
    content: bytes = CONTENT, content_length: int | None = len(CONTENT)
) -> FakeResponse:
    return FakeResponse(content, content_length)


def make_gzip_file(filepath: Path, content: bytes = CONTENT) -> Path:
    """Creat a dummy gzip archive."""
    with gzip.open(filepath, "wb") as f:
        f.write(content)
    return filepath


def make_config(files: list[str], checksums: dict[str, str] | None = None):
    checksums = checksums or {}
    return DatasetConfig(
        name="test",
        mirrors=["https://mirror1.example.com/", "https://mirror2.example.com/"],
        files=[
            FileInfo(label=f, file_path=f, checksum=checksums.get(f)) for f in files
        ],
    )


class TestDeleteFile:
    def test_deletes_existing_file(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        data._delete_file(f)
        assert not f.exists()

    def test_missing_file_is_noop(self, tmp_path: Path):
        data._delete_file(tmp_path / "missing.bin")

    def test_os_error_raises_delete_error(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        with patch.object(Path, "unlink", side_effect=PermissionError("nope")):
            with pytest.raises(DeleteError):
                data._delete_file(f)


class TestVerifyFullFileDownload:
    def test_matching_size_passes(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        data._verify_full_file_download(f, len(CONTENT))
        assert f.exists()

    def test_mismatched_size_deletes_and_raises(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        with pytest.raises(DownloadError):
            data._verify_full_file_download(f, len(CONTENT) + 1)
        assert not f.exists()

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            data._verify_full_file_download(tmp_path / "missing.bin", 5)

    def test_stat_error_raises_read_error(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        with patch.object(Path, "stat", side_effect=OSError("boom")):
            with pytest.raises(ReadError):
                data._verify_full_file_download(f, len(CONTENT))


class TestCalculateFileChecksum:
    def test_returns_sha256(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        assert data._calculate_file_checksum(f) == CONTENT_SHA256

    def test_large_file_read_in_chunks(self, tmp_path: Path):
        big = b"x" * (8192 * 3 + 17)
        f = tmp_path / "big.bin"
        f.write_bytes(big)
        assert data._calculate_file_checksum(f) == hashlib.sha256(big).hexdigest()

    def test_missing_file_raises_filenotfound_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            data._calculate_file_checksum(tmp_path / "missing.bin")

    def test_read_error_raises_read_error(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        with patch("builtins.open", side_effect=OSError("boom")):
            with pytest.raises(ReadError):
                data._calculate_file_checksum(f)


class TestVerifyDownloadedFileIntegrity:
    def test_matching_checksum_passes(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        data._verify_downloaded_file_integrity(f, CONTENT_SHA256)
        assert f.exists()

    def test_missing_file_raises_file_not_found_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            data._verify_downloaded_file_integrity(
                tmp_path / "missing.bin", CONTENT_SHA256
            )

    def test_mismatched_checksum_deletes_and_raises_download_error(
        self, tmp_path: Path
    ):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        with pytest.raises(DownloadError):
            data._verify_downloaded_file_integrity(f, "0" * 64)
        assert not f.exists()


class TestDownloadFileSingleAttempt:
    def test_downloads_file(self, tmp_path: Path):
        out = tmp_path / "a.bin"
        with patch.object(data, "urlopen", return_value=make_response()):
            data._download_file_single_attempt("http://x/a.bin", out)
        assert out.read_bytes() == CONTENT

    def test_uses_timeout(self, tmp_path: Path):
        with patch.object(data, "urlopen", return_value=make_response()) as mock:
            data._download_file_single_attempt("http://x/a.bin", tmp_path / "a.bin")
        mock.assert_called_once_with("http://x/a.bin", timeout=10)

    def test_skips_existing_file_without_overwrite(self, tmp_path: Path):
        out = tmp_path / "a.bin"
        out.write_bytes(b"old")
        with patch.object(data, "urlopen") as mock:
            data._download_file_single_attempt("http://x/a.bin", out)
        mock.assert_not_called()
        assert out.read_bytes() == b"old"

    def test_overwrites_existing_file(self, tmp_path: Path):
        out = tmp_path / "a.bin"
        out.write_bytes(b"old")
        with patch.object(data, "urlopen", return_value=make_response()):
            data._download_file_single_attempt("http://x/a.bin", out, overwrite=True)
        assert out.read_bytes() == CONTENT

    def test_valid_checksum_passes(self, tmp_path: Path):
        out = tmp_path / "a.bin"
        with patch.object(data, "urlopen", return_value=make_response()):
            data._download_file_single_attempt(
                "http://x/a.bin", out, checksum=CONTENT_SHA256
            )
        assert out.exists()

    def test_bad_checksum_raises_and_deletes(self, tmp_path: Path):
        out = tmp_path / "a.bin"
        with patch.object(data, "urlopen", return_value=make_response()):
            with pytest.raises(DownloadError):
                data._download_file_single_attempt(
                    "http://x/a.bin", out, checksum="0" * 64
                )
        assert not out.exists()

    def test_missing_content_length_skips_size_check(self, tmp_path: Path):
        out = tmp_path / "a.bin"
        with patch.object(
            data, "urlopen", return_value=make_response(content_length=None)
        ):
            data._download_file_single_attempt("http://x/a.bin", out)
        assert out.read_bytes() == CONTENT

    def test_incomplete_download_raises_and_deletes(self, tmp_path: Path):
        out = tmp_path / "a.bin"
        with patch.object(
            data, "urlopen", return_value=make_response(content_length=100)
        ):
            with pytest.raises(DownloadError):
                data._download_file_single_attempt("http://x/a.bin", out)
        assert not out.exists()

    @pytest.mark.parametrize("exc", [URLError("down"), TimeoutError("slow")])
    def test_network_error_raises_download_error(self, tmp_path: Path, exc):
        out = tmp_path / "a.bin"
        with patch.object(data, "urlopen", side_effect=exc):
            with pytest.raises(DownloadError):
                data._download_file_single_attempt("http://x/a.bin", out)
        assert not out.exists()

    def test_os_error_raises_write_error(self, tmp_path: Path):
        out = tmp_path / "a.bin"
        with patch.object(data, "urlopen", side_effect=PermissionError("nope")):
            with pytest.raises(WriteError):
                data._download_file_single_attempt("http://x/a.bin", out)


class TestDownloadFile:
    def test_success_first_attempt(self, tmp_path: Path):
        with patch.object(data, "_download_file_single_attempt") as mock:
            data._download_file("http://x/a", tmp_path / "a")
        mock.assert_called_once()

    def test_passes_arguments_through(self, tmp_path: Path):
        out = tmp_path / "a"
        with patch.object(data, "_download_file_single_attempt") as mock:
            data._download_file("http://x/a", out, overwrite=True, checksum="abc")
        mock.assert_called_once_with("http://x/a", out, overwrite=True, checksum="abc")

    def test_retries_then_succeeds(self, tmp_path: Path):
        with patch.object(
            data,
            "_download_file_single_attempt",
            side_effect=[DownloadError("1"), DownloadError("2"), None],
        ) as mock:
            data._download_file("http://x/a", tmp_path / "a", retry_attempts=3)
        assert mock.call_count == 3

    def test_raises_after_exhausting_retries(self, tmp_path: Path):
        with patch.object(
            data, "_download_file_single_attempt", side_effect=DownloadError("fail")
        ) as mock:
            with pytest.raises(DownloadError):
                data._download_file("http://x/a", tmp_path / "a", retry_attempts=3)
        assert mock.call_count == 3

    def test_write_error_not_retried(self, tmp_path: Path):
        with patch.object(
            data, "_download_file_single_attempt", side_effect=WriteError()
        ) as mock:
            with pytest.raises(WriteError):
                data._download_file("http://x/a", tmp_path / "a")
        assert mock.call_count == 1


class TestDownloadDataset:
    def test_invalid_directory_raises(self, tmp_path: Path):
        with pytest.raises(NotADirectoryError):
            data.download_dataset(make_config(["a"]), tmp_path / "missing")

    def test_file_instead_of_directory_raises(self, tmp_path: Path):
        f = tmp_path / "file"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            data.download_dataset(make_config(["a"]), f)

    def test_downloads_all_files_from_first_mirror(self, tmp_path: Path):
        config = make_config(["a", "b"], {"a": "sum-a"})
        with patch.object(data, "_download_file") as mock:
            data.download_dataset(config, tmp_path, overwrite=True)

        assert mock.call_count == 2
        mock.assert_any_call(
            "https://mirror1.example.com/a",
            tmp_path / "a",
            overwrite=True,
            checksum="sum-a",
        )
        mock.assert_any_call(
            "https://mirror1.example.com/b",
            tmp_path / "b",
            overwrite=True,
            checksum=None,
        )

    def test_falls_back_to_next_mirror(self, tmp_path: Path):
        config = make_config(["a"])
        with patch.object(
            data, "_download_file", side_effect=[DownloadError("down"), None]
        ) as mock:
            data.download_dataset(config, tmp_path)

        urls = [c.args[0] for c in mock.call_args_list]
        assert urls == [
            "https://mirror1.example.com/a",
            "https://mirror2.example.com/a",
        ]

    def test_raises_when_all_mirrors_fail(self, tmp_path: Path):
        config = make_config(["a"])
        with patch.object(data, "_download_file", side_effect=DownloadError("down")):
            with pytest.raises(DownloadError, match="'a'"):
                data.download_dataset(config, tmp_path)

    def test_write_error_stops_trying_mirrors(self, tmp_path: Path):
        config = make_config(["a"])
        with patch.object(
            data, "_download_file", side_effect=WriteError("disk")
        ) as mock:
            with pytest.raises(DownloadError):
                data.download_dataset(config, tmp_path)
        assert mock.call_count == 1

    def test_reports_only_failed_files(self, tmp_path: Path):
        config = make_config(["good", "bad"])

        def fake(url, output, **kwargs):
            if url.endswith("bad"):
                raise DownloadError("nope")

        with patch.object(data, "_download_file", side_effect=fake):
            with pytest.raises(DownloadError) as exc_info:
                data.download_dataset(config, tmp_path)

        assert "bad" in str(exc_info.value)
        assert "good" not in str(exc_info.value)

    def test_end_to_end_with_mocked_network(self, tmp_path: Path):
        config = make_config(["a.bin"], {"a.bin": CONTENT_SHA256})
        with patch.object(data, "urlopen", return_value=make_response()) as mock:
            data.download_dataset(config, tmp_path)

        assert (tmp_path / "a.bin").read_bytes() == CONTENT
        assert mock.call_args.args[0] == "https://mirror1.example.com/a.bin"


class TestIsValidGzip:
    def test_gzip_file_is_valid(self, tmp_path: Path):
        f = make_gzip_file(tmp_path / "a.gz")
        assert data._is_valid_gzip(f) is True

    def test_plain_file_is_not_valid(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        assert data._is_valid_gzip(f) is False

    def test_missing_file_is_not_valid(self, tmp_path: Path):
        assert data._is_valid_gzip(tmp_path / "missing.gz") is False

    def test_directory_is_not_valid(self, tmp_path: Path):
        assert data._is_valid_gzip(tmp_path) is False

    def test_empty_file_is_not_valid(self, tmp_path: Path):
        f = tmp_path / "empty.gz"
        f.write_bytes(b"")
        assert data._is_valid_gzip(f) is False

    def test_truncated_magic_number_is_not_valid(self, tmp_path: Path):
        f = tmp_path / "a.gz"
        f.write_bytes(b"\x1f")
        assert data._is_valid_gzip(f) is False

    def test_magic_number_alone_is_valid(self, tmp_path: Path):
        f = tmp_path / "a.gz"
        f.write_bytes(b"\x1f\x8b")
        assert data._is_valid_gzip(f) is True

    def test_read_error_raises_read_error(self, tmp_path: Path):
        f = make_gzip_file(tmp_path / "a.gz")
        with patch("builtins.open", side_effect=OSError("boom")):
            with pytest.raises(ReadError):
                data._is_valid_gzip(f)


class TestUnzipGzipFile:
    def test_unzips_file(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz")
        out = tmp_path / "a.bin"
        data.unzip_gzip_file(zipped, out)
        assert out.read_bytes() == CONTENT

    def test_unzips_empty_file(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "empty.gz", b"")
        out = tmp_path / "empty.bin"
        data.unzip_gzip_file(zipped, out)
        assert out.read_bytes() == b""

    def test_does_not_overwrite_existing_output_file(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz")
        out = tmp_path / "a.bin"
        out.write_bytes(b"old content that is longer than the new content")
        data.unzip_gzip_file(zipped, out, overwrite=False)
        assert out.read_bytes() == b"old content that is longer than the new content"

    def test_non_gzip_file_raises_value_error(self, tmp_path: Path):
        zipped = tmp_path / "a.gz"
        zipped.write_bytes(CONTENT)
        out = tmp_path / "a.bin"
        with pytest.raises(ValueError, match="not a valid gzip file"):
            data.unzip_gzip_file(zipped, out)
        assert not out.exists()

    def test_missing_file_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not a valid gzip file"):
            data.unzip_gzip_file(tmp_path / "missing.gz", tmp_path / "a.bin")

    def test_bad_gzip_body_raises_read_error(self, tmp_path: Path):
        zipped = tmp_path / "a.gz"
        zipped.write_bytes(b"\x1f\x8b" + b"not really gzipped")
        with pytest.raises(ReadError):
            data.unzip_gzip_file(zipped, tmp_path / "a.bin")

    def test_unwritable_output_path_raises_write_error(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz")
        out = tmp_path / "missing_dir" / "a.bin"
        with pytest.raises(WriteError):
            data.unzip_gzip_file(zipped, out)

    def test_write_error_raises_write_error(self, tmp_path: Path):
        zipped = make_gzip_file(tmp_path / "a.gz")
        out = tmp_path / "a.bin"
        with patch.object(data.shutil, "copyfileobj", side_effect=OSError("disk full")):
            with pytest.raises(WriteError):
                data.unzip_gzip_file(zipped, out)
