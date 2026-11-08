import gzip
import shutil
import tempfile
import zlib
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from neural_network.config.data import DatasetConfig
from neural_network.data.idx import load_data
from neural_network.data.utils import create_directory, delete_file
from neural_network.errors import (
    InvalidFileError,
    MissingFileError,
    ReadError,
    WriteError,
)


def _is_valid_gzip(filepath: Path) -> bool:
    """Check whether a file starts with the gzip magic number.

    Args:
        filepath (Path): Path to the file to check.

    Returns:
        bool: True if the file exists and starts with the gzip magic number.

    Raises:
        ReadError: If the file exists but cannot be read.
    """
    magic: bytes = b"\x1f\x8b"

    if not filepath.is_file():
        return False

    try:
        with open(filepath, "rb") as f:
            return f.read(2) == magic
    except OSError as ex:
        logger.warning(f"Unable to read {filepath}: {ex}")
        raise ReadError(f"Unable to read {filepath}: {ex}") from ex


def _unzip_gzip_file(
    zipped_filepath: Path, output_filepath: Path, overwrite: bool = False
) -> None:
    """Decompress a gzip file.

    Args:
        zipped_filepath (Path): Path to the gzip file.
        output_filepath (Path): Path to write the decompressed contents to.
        overwrite (bool, optional): Whether an existing output file should be
            overwritten. Defaults to False.

    Raises:
        ReadError: If the gzip file cannot be read.
        InvalidFileError: If the file is not a valid, complete gzip file.
        WriteError: If the decompressed contents cannot be written.
        DeleteError: If a partially written output file cannot be cleaned up.
    """
    logger.debug(
        f"Unzipping file {zipped_filepath} and saving contents to {output_filepath}"
    )

    if not _is_valid_gzip(zipped_filepath):
        logger.warning(f"{zipped_filepath} is not a valid gzip file")
        raise InvalidFileError(f"{zipped_filepath} is not a valid gzip file")

    if output_filepath.is_file() and not overwrite:
        logger.info(
            f"Unable to unzip: {output_filepath} already exists and overwrite set to false"
        )
        return

    try:
        with gzip.open(zipped_filepath, "rb") as f_in:
            with open(output_filepath, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.success(f"Successfully unzipped {zipped_filepath} to {output_filepath}")
    # BadGzipFile is an OSError, so it must be caught before the write handler.
    # A truncated archive raises EOFError and corrupted data raises zlib.error.
    except (gzip.BadGzipFile, EOFError, zlib.error) as ex:
        logger.warning(f"{zipped_filepath} is not a valid gzip file: {ex}")
        delete_file(output_filepath)
        raise InvalidFileError(
            f"{zipped_filepath} is not a valid gzip file: {ex}"
        ) from ex
    except OSError as ex:
        logger.warning(f"Unable to write to {output_filepath}: {ex}")
        delete_file(output_filepath)
        raise WriteError(f"Unable to write to {output_filepath}: {ex}") from ex


def _convert_idx_to_np(
    idx_filepath: Path, output_filepath: Path, overwrite: bool = False
) -> None:
    """Load an IDX file and save its data as a NumPy ``.npy`` file.

    Args:
        idx_filepath (Path): Path to the IDX file.
        output_filepath (Path): Path to write the NumPy file to.
        overwrite (bool, optional): Whether an existing output file should be
            overwritten. Defaults to False.

    Raises:
        ReadError: If the IDX file cannot be read.
        ParseError: If the IDX header is truncated.
        InvalidFileError: If the file is not a valid IDX file.
        WriteError: If the NumPy file cannot be written.
        DeleteError: If a partially written output file cannot be cleaned up.
    """
    logger.debug(
        f"Parsing IDX file {idx_filepath} and saving results to NumPy file {output_filepath}"
    )

    if output_filepath.is_file() and not overwrite:
        logger.info(
            f"Unable to save to NumPy file: {output_filepath} already exists and overwrite set to false"
        )
        return

    try:
        data: NDArray = load_data(idx_filepath)
        np.save(output_filepath, data)
        logger.success(
            f"Successfully converted file {idx_filepath} in IDX format to NumPy array and saved to {output_filepath}"
        )
    except OSError as ex:
        logger.warning(f"Unable to write to {output_filepath}: {ex}")
        delete_file(output_filepath)
        raise WriteError(f"Unable to write to {output_filepath}: {ex}") from ex


def _process_mnist_file(
    input_file: Path, output_file: Path, overwrite: bool = False
) -> None:
    """Convert one gzipped IDX file to a NumPy file via a temporary directory.

    Args:
        input_file (Path): Path to the gzipped IDX file.
        output_file (Path): Path to write the NumPy file to.
        overwrite (bool, optional): Whether an existing output file should be
            overwritten. Defaults to False.

    Raises:
        ReadError: If the input file cannot be read.
        ParseError: If the IDX header is truncated.
        InvalidFileError: If the file is not a valid gzipped IDX file.
        WriteError: If the temporary directory or output file cannot be written.
        DeleteError: If a partially written file cannot be cleaned up.
    """
    try:
        with tempfile.TemporaryDirectory() as t_dir:
            t_file = Path(t_dir) / "unzipped.idx"
            _unzip_gzip_file(input_file, t_file)
            _convert_idx_to_np(t_file, output_file, overwrite)
    except OSError as ex:
        # The helpers convert their own OSErrors, so this can only come from
        # creating or cleaning up the temporary directory.
        logger.warning(f"Unable to use a temporary directory: {ex}")
        raise WriteError(f"Unable to use a temporary directory: {ex}") from ex
    logger.success(
        f"Successfully processed file {input_file} and saved output to {output_file}"
    )


def process_mnist_dataset(
    dataset_config: DatasetConfig, overwrite: bool = False
) -> None:
    """Convert a dataset's raw gzipped IDX files to NumPy files.

    Args:
        dataset_config (DatasetConfig): Configuration for the dataset to process.
            Its processed data directory is created if it does not exist.
        overwrite (bool, optional): Whether existing files should be overwritten.
            Defaults to False.

    Raises:
        ReadError: If the raw data directory or a raw file cannot be read.
        MissingFileError: If any raw data files are missing.
        ParseError: If a raw file's IDX header is truncated.
        InvalidFileError: If a raw file is not a valid gzipped IDX file.
        WriteError: If the processed data directory or a processed file cannot be
            written.
        DeleteError: If a partially written file cannot be cleaned up.
    """
    logger.info(f"Processing {dataset_config.name} dataset")

    output_dirpath: Path = dataset_config.processed_data_loc

    expected_raw_files: list[Path] = [
        dataset_config.raw_data_loc / f.file_name for f in dataset_config.files
    ]
    processed_files: list[Path] = [
        output_dirpath / f"{f.label}.npy" for f in dataset_config.files
    ]

    missing_files = []
    try:
        existing_raw_files = set(dataset_config.raw_data_loc.iterdir())
    except OSError as ex:
        logger.warning(f"Unable to read raw data directory: {ex}")
        raise ReadError(f"Unable to read raw data directory: {ex}") from ex
    for f in expected_raw_files:
        if f not in existing_raw_files:
            missing_files.append(f.name)
    if missing_files:
        logger.warning(f"Raw data files could not be found: {missing_files}")
        raise MissingFileError(f"Raw data files could not be found: {missing_files}")

    create_directory(output_dirpath)

    for in_file, out_file in zip(expected_raw_files, processed_files, strict=True):
        if out_file.is_file() and not overwrite:
            logger.info(f"File {out_file} already exists; skipping")
            continue
        _process_mnist_file(in_file, out_file, overwrite=overwrite)

    logger.success(f"Successfully processed all files in {dataset_config.name} dataset")
