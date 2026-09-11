import hashlib
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from loguru import logger

from neural_network.errors import DeleteError, DownloadError, ReadError, WriteError


def _delete_file(filepath: Path) -> None:
    """Delete a file at a given path.

    Args:
        filepath (Path): Path to file to delete.

    Raises:
        DeleteError: If there is an error when attempting to delete the file.
    """
    try:
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted file at {filepath}")
    except OSError as ex:
        logger.warning(f"Error deleting file at {filepath}: {ex}")
        raise DeleteError from ex


def _verify_full_file_download(
    downloaded_filepath: Path, expected_file_size: int
) -> None:
    """Check downloaded file is the correct size.

    Args:
        downloaded_filepath (Path): Path to where the file was downloaded.
        expected_file_size (int): Expected size of the file.

    Raises:
        DownloadError: If the downloaded file is not the correct size.
    """
    actual_file_size: int | None = None
    try:
        if downloaded_filepath.exists():
            actual_file_size = downloaded_filepath.stat().st_size
    except OSError as ex:
        logger.warning(
            f"Error occurred while fetching file size from {downloaded_filepath}: {ex}"
        )
        raise ReadError from ex

    if (
        expected_file_size is not None
        and actual_file_size is not None
        and actual_file_size != expected_file_size
    ):
        logger.warning(f"Incomplete file download from to {downloaded_filepath}")
        _delete_file(downloaded_filepath)
        raise DownloadError(f"Failed to download file to {downloaded_filepath}")


def _calculate_file_checksum(filepath: Path) -> str:
    """Calculate the SHA256 checksum of a file.

    Args:
        filepath (Path): Path to file to calculate checksum for.

    Returns:
        str: SHA256 checksum of the file.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
    except OSError as ex:
        logger.warning(
            f"Error occurred while calculating checksum for {filepath}: {ex}"
        )
        raise ReadError from ex


def _verify_downloaded_file_integrity(
    downloaded_filepath: Path, expected_checksum: str
) -> None:
    actual_checksum: str = _calculate_file_checksum(downloaded_filepath)
    if actual_checksum != expected_checksum:
        _delete_file(downloaded_filepath)
        raise DownloadError(f"Corrupted file download for {downloaded_filepath}")


def _download_file(url: str, output_filepath: Path, overwrite: bool = False) -> None:
    """Download a file from a given URL to a local directory path.

    Args:
        url (str): URL path of the file to download.
        output_filepath (Path): Path to downloaded file.
        overwrite (bool, optional): Whether existing file should be overwritten.
            Defaults to False.

    Raises:
        DownloadError: If there is an error when attempting file download.
        WriteError: If there is an error when attempting to write the file to disk.
    """
    if output_filepath.exists() and not overwrite:
        logger.info(f"File already exists at {output_filepath}, skipping download")
        return

    logger.info(f"Downloading file from {url} to {output_filepath}")
    try:
        with (
            urlopen(url, timeout=10) as response,
            open(output_filepath, "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)

            expected_file_size: int | None = None
            if url_file_size := response.getheader("Content-Length"):
                expected_file_size = int(url_file_size)

    except (URLError, TimeoutError) as ex:
        logger.warning(f"Error downloading file from {url}: {ex}")
        _delete_file(output_filepath)
        raise DownloadError from ex
    except OSError as ex:
        logger.warning(f"Error writing file to {output_filepath}: {ex}")
        _delete_file(output_filepath)
        raise WriteError from ex

    if expected_file_size:
        _verify_full_file_download(output_filepath, expected_file_size)
    logger.info(f"Successfully downloaded file from {url} to {output_filepath}")


def download_mnist_dataset(output_dirpath: Path, overwrite: bool = False) -> None:
    """Download the raw MNIST data files to a given local directory.

    Args:
        output_dirpath (Path): Path to local directory where files will be downloaded.
        overwrite (bool, optional): Whether existing MNIST files should be overwritten.
            Defaults to False.

    Raises:
        NotADirectoryError: If given output directory is not valid.
        FileNotFoundError: If any file(s) are not successfully downloaded.
    """
    mirrors: list[str] = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]
    files: set[str] = {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    }

    if not output_dirpath.exists() or not output_dirpath.is_dir():
        logger.error(f"{output_dirpath} is not a valid directory")
        raise NotADirectoryError(f"{output_dirpath} is not a valid directory")

    # Download files
    downloaded_files: set[str] = set()
    for file_name in files:
        for mirror in mirrors:
            try:
                file_url: str = mirror + file_name
                output_filepath: Path = output_dirpath / file_name
                _download_file(file_url, output_filepath, overwrite=overwrite)
                downloaded_files.add(file_name)
                break
            except DownloadError as ex:
                logger.warning(f"Failed to download {file_name} from {mirror}: {ex}")

    if downloaded_files != files:
        failed = files - downloaded_files
        logger.error(f"Failed to download the following file(s): {str(failed)}")
        raise FileNotFoundError(
            f"Failed to download the following file(s): {str(failed)}"
        )
