import gzip
import shutil
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from neural_network.data.idx import load_data
from neural_network.data.utils import delete_file
from neural_network.errors import (
    ReadError,
    WriteError,
)


def _is_valid_gzip(filepath: Path) -> bool:
    if not filepath.is_file():
        return False

    try:
        with open(filepath, "rb") as f:
            return (
                f.read(2) == b"\x1f\x8b"
            )  # NOTE: magic number for gzip compressed files is '1f 8b'
    except OSError as ex:
        logger.warning(f"Unable to read {filepath}")
        raise ReadError(f"Unable to read {filepath}") from ex


def unzip_gzip_file(
    zipped_filepath: Path, output_filepath: Path, overwrite: bool = False
) -> None:
    logger.debug(
        f"Unzipping file {zipped_filepath} and saving contents to {output_filepath}"
    )

    if not _is_valid_gzip(zipped_filepath):
        logger.warning(f"{zipped_filepath} is not a valid gzip file")
        raise ValueError(f"{zipped_filepath} is not a valid gzip file")

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


def convert_idx_to_np(
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

    data: NDArray = load_data(idx_filepath)
    np.save(output_filepath, data)
