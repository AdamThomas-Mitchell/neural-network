class DataError(Exception):
    """Base exception for errors regarding data."""


class DownloadError(DataError):
    """Error when downloading an external resource."""


class WriteError(DataError):
    """Error when writing to a local file."""


class DeleteError(DataError):
    """Error when deleting a local file."""


class ReadError(DataError):
    """Error when reading a local file."""
