from pathlib import Path

from loguru import logger

from neural_network.errors import DeleteError, WriteError


def create_directory(dirpath: Path) -> None:
    """Create a directory, along with any missing parents, if it does not exist.

    Args:
        dirpath (Path): Path to directory to create.

    Raises:
        WriteError: If the directory cannot be created, e.g. because a file already
            exists at the path.
    """
    try:
        dirpath.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        logger.warning(f"Error creating directory at {dirpath}: {ex}")
        raise WriteError(f"Error creating directory at {dirpath}: {ex}") from ex


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
