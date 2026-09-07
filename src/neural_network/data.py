import shutil
from pathlib import Path
from urllib.error import ContentTooShortError, HTTPError, URLError
from urllib.request import urlopen

from loguru import logger

from neural_network.errors import DownloadError


def _verify_full_file_download(url: str, downloaded_filepath: Path) -> None:
    """Check downloaded file is same size as file at the URL path.

    Args:
        url (str): URL path for file to download.
        downloaded_filepath (Path): Path to where the file was downloaded.

    Raises:
        DownloadError: If the downloaded file is not the same size as the file at the
            URL path.
    """
    expected_file_size: int | None = None
    with urlopen(url, timeout=10) as response:  # TODO: try-except block here maybe
        if response.has_header("Content-Length"):
            expected_file_size = int(response.get_header("Content-Length"))

    actual_file_size: int = downloaded_filepath.stat().st_size
    if expected_file_size and actual_file_size != expected_file_size:
        logger.warning(f"Incomplete file download from {url} to {downloaded_filepath}")
        raise DownloadError(f"Failed to download file at {url}")

    logger.info(f"Verified full download from {url} to {downloaded_filepath}")


def _verify_downloaded_file_integrity(url: str, downloaded_filepath: Path) -> None:
    # TODO
    pass


def download_file(url: str, output_filepath: Path) -> None:
    """Download a file from a given URL to a local directory path.

    Args:
        url (str): URL path of the file to download.
        output_path (Path): Path to downloaded file.
    """
    logger.info(f"Downloading file {url} to {output_filepath}...")
    try:
        with (
            urlopen(url, timeout=10) as response,
            open(str(output_filepath), "wb") as out_file,
        ):
            shutil.copyfileobj(response, out_file)
    except (
        HTTPError,
        ContentTooShortError,
        URLError,
        TimeoutError,
        FileNotFoundError,
    ) as ex:
        logger.warning("Failed to download file")
        if output_filepath.exists():
            output_filepath.unlink()
        raise DownloadError from ex
    logger.info("Download complete")


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

    # Validate output path is valid directory
    if not output_dirpath.exists() or not output_dirpath.is_dir():
        logger.error(f"{output_dirpath} is not a valid directory")
        raise NotADirectoryError(f"{output_dirpath} is not a valid directory")

    # Optionally, overwrite existing MNIST files
    gz_files = set(f.name for f in output_dirpath.glob("*.gz") if f.is_file())
    existing_files: set[str] = files & gz_files
    if existing_files and overwrite:
        logger.info(f"Overwriting the following existing files: {str(existing_files)}")
    else:
        files = files - existing_files

    # Download files
    downloaded_files: set[str] = set()
    for file_name in files:
        for mirror in mirrors:
            try:
                file_url: str = mirror + file_name
                output_filepath: Path = output_dirpath / file_name
                download_file(file_url, output_filepath)
                downloaded_files.add(file_name)
                logger.info(
                    f"Downloaded file from {file_url} and saved to {str(output_dirpath)}"
                )
                break
            # TODO: Raise custom error here
            except (HTTPError, ContentTooShortError, URLError, TimeoutError) as ex:
                logger.warning(ex)

    # Verify files downloaded
    if downloaded_files != files:
        failed = files - downloaded_files
        logger.error(f"Failed to download the following file(s): {str(failed)}")
        raise DownloadError(f"Failed to download the following files: {str(failed)}")
