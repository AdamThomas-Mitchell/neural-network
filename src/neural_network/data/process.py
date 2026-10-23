import gzip
import shutil
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from neural_network.config.data import DatasetConfig
from neural_network.data.idx import load_data
from neural_network.data.utils import delete_file
from neural_network.errors import (
    InvalidFileError,
    ReadError,
    WriteError,
)


def _is_valid_gzip(filepath: Path) -> bool:
    magic: bytes = b"\x1f\x8b"

    if not filepath.is_file():
        return False

    try:
        with open(filepath, "rb") as f:
            return f.read(2) == magic
    except OSError as ex:
        logger.warning(f"Unable to read {filepath}")
        raise ReadError(f"Unable to read {filepath}") from ex


def _unzip_gzip_file(
    zipped_filepath: Path, output_filepath: Path, overwrite: bool = False
) -> None:
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
    except gzip.BadGzipFile as ex:
        logger.warning(f"Unable to parse file: {zipped_filepath}")
        delete_file(output_filepath)
        raise ReadError(f"Unable to parse file: {zipped_filepath}") from ex
    except OSError as ex:
        logger.warning(f"Unable to write to {output_filepath}")
        delete_file(output_filepath)
        raise WriteError(f"Unable to write to {output_filepath}") from ex


def _convert_idx_to_np(
    idx_filepath: Path, output_filepath: Path, overwrite: bool = False
) -> None:
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
        logger.warning(f"Unable to write to {output_filepath}")
        delete_file(output_filepath)
        raise WriteError(f"Unable to write to {output_filepath}") from ex


def _process_mnist_file(input_file: Path, output_file: Path) -> None:
    with tempfile.TemporaryDirectory() as t_dir:
        t_file = Path(t_dir) / "unzipped.idx"
        _unzip_gzip_file(input_file, t_file)
        _convert_idx_to_np(t_file, output_file)
    logger.success(
        f"Successfully processed file {input_file} and saved output to {output_file}"
    )


def process_mnist_dataset(
    dataset_config: DatasetConfig, overwrite: bool = False
) -> None:
    logger.info(f"Processing {dataset_config.name} dataset")

    expected_raw_files: list[Path] = [
        dataset_config.raw_data_loc / f.file_name for f in dataset_config.files
    ]
    processed_files: list[Path] = [
        dataset_config.processed_data_loc / f"{f.label}.npy"
        for f in dataset_config.files
    ]

    missing_files = []
    existing_raw_files = set(dataset_config.raw_data_loc.iterdir())
    for f in expected_raw_files:
        if f not in existing_raw_files:
            missing_files.append(f.name)
    if missing_files:
        logger.warning(f"Raw data files could not be found: {missing_files}")
        raise FileNotFoundError(f"Raw data files could not be found: {missing_files}")

    for in_file, out_file in zip(expected_raw_files, processed_files, strict=True):
        if out_file.is_file() and not overwrite:
            logger.info(f"File {out_file} already exists; skipping")
            continue
        _process_mnist_file(in_file, out_file)

    logger.success(f"Successfully processed all files in {dataset_config.name} dataset")
