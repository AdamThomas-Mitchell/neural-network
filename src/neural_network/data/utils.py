from pathlib import Path

from loguru import logger

from neural_network.errors import DeleteError


def delete_file(filepath: Path) -> None:
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
