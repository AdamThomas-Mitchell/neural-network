import hashlib
import shutil
from http.client import HTTPException
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from loguru import logger

from neural_network.config.data import DatasetConfig
from neural_network.data.utils import create_directory, delete_file
from neural_network.errors import (
    DeleteError,
    DownloadError,
    ReadError,
    WriteError,
)


def _verify_full_file_download(
    downloaded_filepath: Path, expected_file_size: int
) -> None:
    """Check downloaded file is the correct size.

    Args:
        downloaded_filepath (Path): Path to where the file was downloaded.
        expected_file_size (int): Expected size of the file.

    Raises:
        ReadError: If the file's size cannot be read, e.g. because it does not exist.
        DownloadError: If the downloaded file is not the correct size.
        DeleteError: If the incomplete file cannot be deleted.
    """
    logger.debug(f"Verifying downloaded file size for {downloaded_filepath}")

    try:
        actual_file_size: int = downloaded_filepath.stat().st_size
    except OSError as ex:
        logger.warning(f"Unable to read file size of {downloaded_filepath}: {ex}")
        raise ReadError(
            f"Unable to read file size of {downloaded_filepath}: {ex}"
        ) from ex

    if actual_file_size != expected_file_size:
        logger.warning(
            f"Incomplete file download to {downloaded_filepath}: expected "
            f"{expected_file_size} bytes, got {actual_file_size}"
        )
        delete_file(downloaded_filepath)
        raise DownloadError(
            f"Incomplete file download to {downloaded_filepath}: expected "
            f"{expected_file_size} bytes, got {actual_file_size}"
        )

    logger.debug(
        f"Successfully verified downloaded file size for {downloaded_filepath}"
    )


def _calculate_file_checksum(filepath: Path) -> str:
    """Calculate the SHA256 checksum of a file.

    Args:
        filepath (Path): Path to file to calculate checksum for.

    Returns:
        str: SHA256 checksum of the file.

    Raises:
        ReadError: If the file cannot be read, e.g. because it does not exist.
    """
    logger.debug(f"Calculating SHA256 checksum for {filepath}")

    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
    except OSError as ex:
        logger.warning(f"Unable to read {filepath} to calculate checksum: {ex}")
        raise ReadError(
            f"Unable to read {filepath} to calculate checksum: {ex}"
        ) from ex


def _verify_downloaded_file_integrity(
    downloaded_filepath: Path, expected_checksum: str
) -> None:
    """Verify the integrity of a downloaded file by checking its checksum.

    Args:
        downloaded_filepath (Path): Path to the downloaded file.
        expected_checksum (str): Expected SHA256 checksum of the file.

    Raises:
        ReadError: If the file cannot be read, e.g. because it does not exist.
        DownloadError: If the downloaded file has a different checksum than expected.
        DeleteError: If the corrupted file cannot be deleted.
    """
    logger.debug(f"Verifying downloaded file integrity for {downloaded_filepath}")

    actual_checksum: str = _calculate_file_checksum(downloaded_filepath)
    if actual_checksum != expected_checksum:
        logger.warning(f"Checksum mismatch for downloaded file {downloaded_filepath}")
        delete_file(downloaded_filepath)
        raise DownloadError(f"Corrupted file download for {downloaded_filepath}")

    logger.debug(
        f"Successfully verified downloaded file integrity for {downloaded_filepath}"
    )


def _parse_content_length(header_value: str | None) -> int | None:
    """Parse a ``Content-Length`` header into an expected file size.

    Args:
        header_value (str | None): Raw header value, or None if the header is absent.

    Returns:
        int | None: Expected size in bytes, or None if the header is absent or is
            not a non-negative integer, in which case the size check is skipped.
    """
    if header_value is None:
        return None

    try:
        content_length = int(header_value)
    except ValueError:
        logger.warning(f"Ignoring invalid Content-Length header: {header_value!r}")
        return None

    if content_length < 0:
        logger.warning(f"Ignoring negative Content-Length header: {header_value!r}")
        return None
    return content_length


def _download_file_single_attempt(
    url: str,
    output_filepath: Path,
    overwrite: bool = False,
    checksum: str | None = None,
) -> None:
    """Makes a single attempt to download a file from a given URL to a local directory path.

    Args:
        url (str): URL path of the file to download.
        output_filepath (Path): Path to downloaded file.
        overwrite (bool, optional): Whether existing file should be overwritten.
            Defaults to False.
        checksum (str | None, optional): Expected SHA256 checksum of the file.
            Defaults to None.

    Raises:
        DownloadError: If the download fails, is incomplete or is corrupted.
        WriteError: If there is an error when attempting to write the file to disk.
        ReadError: If the downloaded file cannot be read back for verification.
        DeleteError: If a failed download cannot be cleaned up.
    """
    logger.info(f"Downloading file from {url} to {output_filepath}")

    if output_filepath.exists() and not overwrite:
        logger.info(f"File already exists at {output_filepath}, skipping download")
        return

    try:
        with (
            urlopen(url, timeout=10) as response,
            open(output_filepath, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)
            expected_file_size = _parse_content_length(
                response.getheader("Content-Length")
            )

    # Network errors can surface mid-transfer, so they must be caught before the
    # broader OSError (ConnectionError, TimeoutError and URLError are subclasses).
    except (URLError, TimeoutError, ConnectionError, HTTPException) as ex:
        logger.warning(f"Error downloading file from {url}: {ex}")
        delete_file(output_filepath)
        raise DownloadError(f"Error downloading file from {url}: {ex}") from ex
    except OSError as ex:
        logger.warning(f"Error writing file to {output_filepath}: {ex}")
        delete_file(output_filepath)
        raise WriteError(f"Error writing file to {output_filepath}: {ex}") from ex

    if expected_file_size is not None:
        _verify_full_file_download(output_filepath, expected_file_size)
    if checksum:
        _verify_downloaded_file_integrity(output_filepath, checksum)
    logger.success(f"Successfully downloaded file from {url} to {output_filepath}")


def _download_file(
    url: str,
    output_filepath: Path,
    overwrite: bool = False,
    checksum: str | None = None,
    retry_attempts: int = 3,
) -> None:
    """Download a file from a given URL to a local directory path with retry logic.

    Args:
        url (str): URL path of the file to download.
        output_filepath (Path): Path to downloaded file.
        overwrite (bool, optional): Whether existing file should be overwritten.
            Defaults to False.
        checksum (str | None, optional): Expected SHA256 checksum of the file.
            Defaults to None.
        retry_attempts (int, optional): Number of times to retry the download.
            Defaults to 3.

    Raises:
        DownloadError: If every attempt to download the file fails.
        WriteError: If there is an error when attempting to write the file to disk.
        ReadError: If the downloaded file cannot be read back for verification.
        DeleteError: If a failed download cannot be cleaned up.
    """
    attempt: int = 0
    while True:
        try:
            _download_file_single_attempt(
                url, output_filepath, overwrite=overwrite, checksum=checksum
            )
            return
        except DownloadError as ex:
            attempt += 1
            logger.warning(
                f"Attempt {attempt} failed to download file from {url}: {ex}"
            )
            if attempt >= retry_attempts:
                logger.warning(
                    f"Failed to download file from {url} after {attempt} attempts"
                )
                raise


def download_dataset(dataset_config: DatasetConfig, overwrite: bool = False) -> None:
    """Download the data files for a given dataset to a local directory.

    Args:
        dataset_config (DatasetConfig): Configuration for the dataset to download.
            Its raw data directory is created if it does not exist.
        overwrite (bool, optional): Whether existing files should be overwritten.
            Defaults to False.

    Raises:
        WriteError: If the raw data directory cannot be created.
        DownloadError: If any file(s) are not successfully downloaded and saved.
    """
    output_dirpath: Path = dataset_config.raw_data_loc
    logger.info(f"Downloading {dataset_config.name} dataset files to {output_dirpath}")

    create_directory(output_dirpath)

    downloaded_files: set[str] = set()
    for file_info in dataset_config.files:
        file_name: str = file_info.file_name
        checksum: str | None = file_info.checksum

        for mirror in dataset_config.mirrors:
            try:
                file_url: str = mirror.encoded_string() + file_name
                output_filepath: Path = output_dirpath / file_name
                _download_file(
                    file_url, output_filepath, overwrite=overwrite, checksum=checksum
                )
                downloaded_files.add(file_name)
                break
            except DownloadError as ex:
                logger.warning(f"Failed to download {file_name} from {mirror}: {ex}")
            except (WriteError, ReadError, DeleteError) as ex:
                # A local disk problem will not be fixed by trying another mirror.
                logger.warning(f"Failed to save {file_name} to {output_dirpath}: {ex}")
                break

    files_to_download: set[str] = {f.file_name for f in dataset_config.files}
    if downloaded_files != files_to_download:
        failed = sorted(files_to_download - downloaded_files)
        logger.warning(f"Failed to download the following file(s): {failed}")
        raise DownloadError(f"Failed to download the following file(s): {failed}")

    logger.success(
        f"Successfully downloaded {dataset_config.name} dataset to {output_dirpath}"
    )
