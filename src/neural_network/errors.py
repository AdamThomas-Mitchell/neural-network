class DownloadError(Exception):
    """Error when downloading an external resource."""


class WriteError(Exception):
    """Error when writing to a local file."""


class DeleteError(Exception):
    """Error when deleting a local file."""


class ReadError(Exception):
    """Error when reading a local file."""
