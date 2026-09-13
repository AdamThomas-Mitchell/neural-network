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
    logger.debug(f"Deleting file at {filepath}")
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
    logger.debug(f"Verifying downloaded file size for {downloaded_filepath}")
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

    logger.debug(
        f"Successfully verified downloaded file size for {downloaded_filepath}"
    )


def _calculate_file_checksum(filepath: Path) -> str:
    """Calculate the SHA256 checksum of a file.

    Args:
        filepath (Path): Path to file to calculate checksum for.

    Returns:
        str: SHA256 checksum of the file.
    """
    logger.debug(f"Calculating SHA256 checksum for {filepath}")

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
    """Verify the integrity of a downloaded file by checking its checksum.

    Args:
        downloaded_filepath (Path): Path to the downloaded file.
        expected_checksum (str): Expected SHA256 checksum of the file.

    Raises:
        DownloadError: If the downloaded file has a different checksum than expected.
    """
    logger.debug(f"Verifying downloaded file integrity for {downloaded_filepath}")

    actual_checksum: str = _calculate_file_checksum(downloaded_filepath)
    if actual_checksum != expected_checksum:
        _delete_file(downloaded_filepath)
        raise DownloadError(f"Corrupted file download for {downloaded_filepath}")

    logger.debug(
        f"Successfully verified downloaded file integrity for {downloaded_filepath}"
    )


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
        DownloadError: If there is an error when attempting file download.
        WriteError: If there is an error when attempting to write the file to disk.
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
        DownloadError: If there is an error when attempting file download.
        WriteError: If there is an error when attempting to write the file to disk.
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


def download_mnist_dataset(output_dirpath: Path, overwrite: bool = False) -> None:
    """Download the raw MNIST data files to a given local directory.

    Args:
        output_dirpath (Path): Path to local directory where files will be downloaded.
            Must be a valid existing directory.
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
    files: dict[str, dict[str, str]] = {
        "train_images": {
            "file_name": "train-images-idx3-ubyte.gz",
            "checksum": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        },
        "train_labels": {
            "file_name": "train-labels-idx1-ubyte.gz",
            "checksum": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        },
        "test_images": {
            "file_name": "t10k-images-idx3-ubyte.gz",
            "checksum": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        },
        "test_labels": {
            "file_name": "t10k-labels-idx1-ubyte.gz",
            "checksum": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        },
    }
    logger.info(f"Downloading MNIST dataset to {output_dirpath}")

    if not output_dirpath.exists() or not output_dirpath.is_dir():
        logger.error(f"{output_dirpath} is not a valid directory")
        raise NotADirectoryError(f"{output_dirpath} is not a valid directory")

    downloaded_files: set[str] = set()
    for _, value in files.items():
        file_name: str = value["file_name"]
        checksum: str = value["checksum"]

        for mirror in mirrors:
            try:
                file_url: str = mirror + file_name
                output_filepath: Path = output_dirpath / file_name
                _download_file(
                    file_url, output_filepath, overwrite=overwrite, checksum=checksum
                )
                downloaded_files.add(file_name)
                break
            except DownloadError as ex:
                logger.warning(f"Failed to download {file_name} from {mirror}: {ex}")

    files_to_download: set[str] = {f["file_name"] for _, f in files.items()}
    if downloaded_files != files_to_download:
        failed = files_to_download - downloaded_files
        logger.error(f"Failed to download the following file(s): {str(failed)}")
        raise FileNotFoundError(
            f"Failed to download the following file(s): {str(failed)}"
        )

    logger.success(f"Successfully downloaded MNIST dataset to {output_dirpath}")
