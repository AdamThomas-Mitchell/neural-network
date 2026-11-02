from pathlib import Path
from unittest.mock import patch

import pytest

from neural_network.data.utils import delete_file
from neural_network.errors import DeleteError


CONTENT = b"hello world"


class TestDeleteFile:
    def test_deletes_existing_file(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        delete_file(f)
        assert not f.exists()

    def test_missing_file_is_noop(self, tmp_path: Path):
        delete_file(tmp_path / "missing.bin")

    def test_os_error_raises_delete_error(self, tmp_path: Path):
        f = tmp_path / "a.bin"
        f.write_bytes(CONTENT)
        with patch.object(Path, "unlink", side_effect=PermissionError("nope")):
            with pytest.raises(DeleteError):
                delete_file(f)
