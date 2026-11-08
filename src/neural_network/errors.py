class DataError(Exception):
    """Base exception for errors regarding data."""


class DatasetConfigError(DataError):
    """Error for an invalid or unreadable dataset config."""


class DownloadError(DataError):
    """Error when downloading an external resource."""


class WriteError(DataError):
    """Error when writing to a local file."""


class DeleteError(DataError):
    """Error when deleting a local file."""


class ReadError(DataError):
    """Error when reading a local file."""


class MissingFileError(ReadError):
    """Error when expected local files do not exist."""


class ParseError(DataError):
    """Error when parsing binary data."""


class InvalidFileError(DataError):
    """Error when a file is invalid for a given format (e.g., invalid IDX or gzip file)."""
