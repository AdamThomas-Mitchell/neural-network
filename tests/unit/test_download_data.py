from pathlib import Path
from unittest.mock import patch

import pytest

from neural_network.config.data import DatasetConfig
from neural_network.errors import DownloadError
from neural_network.scripts import download_data


VALID_YAML = """
name: test
mirrors:
  - https://mirror.example.com/
files:
  - label: images
    file_path: images.gz
    checksum: abc123
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML)
    return path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    path = tmp_path / "out"
    path.mkdir()
    return path


def run_main(monkeypatch: pytest.MonkeyPatch, *args: object) -> None:
    monkeypatch.setattr("sys.argv", ["download-data", *map(str, args)])
    download_data.main()


class TestMain:
    def test_success_calls_download_dataset(
        self, monkeypatch, config_path: Path, output_dir: Path
    ):
        with patch.object(download_data, "download_dataset") as mock_download:
            run_main(monkeypatch, config_path, output_dir)

        mock_download.assert_called_once()
        kwargs = mock_download.call_args.kwargs
        assert isinstance(kwargs["dataset_config"], DatasetConfig)
        assert kwargs["dataset_config"].name == "test"
        assert kwargs["dataset_config"].files[0].checksum == "abc123"
        assert kwargs["output_dirpath"] == output_dir
        assert kwargs["overwrite"] is False

    def test_overwrite_flag_is_passed_through(
        self, monkeypatch, config_path: Path, output_dir: Path
    ):
        with patch.object(download_data, "download_dataset") as mock_download:
            run_main(monkeypatch, config_path, output_dir, "--overwrite")

        assert mock_download.call_args.kwargs["overwrite"] is True

    def test_missing_config_file_exits(
        self, monkeypatch, tmp_path: Path, output_dir: Path
    ):
        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, tmp_path / "missing.yaml", output_dir)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_config_path_that_is_a_directory_exits(
        self, monkeypatch, tmp_path: Path, output_dir: Path
    ):
        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, tmp_path, output_dir)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_missing_output_dir_exits(
        self, monkeypatch, config_path: Path, tmp_path: Path
    ):
        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, config_path, tmp_path / "nope")

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_output_path_that_is_a_file_exits(self, monkeypatch, config_path: Path):
        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, config_path, config_path)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_invalid_yaml_exits(self, monkeypatch, tmp_path: Path, output_dir: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: [unclosed")

        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, bad, output_dir)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_unreadable_config_exits(
        self, monkeypatch, config_path: Path, output_dir: Path
    ):
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with patch.object(download_data, "download_dataset") as mock_download:
                with pytest.raises(SystemExit) as exc:
                    run_main(monkeypatch, config_path, output_dir)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    @pytest.mark.parametrize(
        "contents",
        [
            "name: test\n",  # missing mirrors and files
            "name: test\nmirrors: []\nfiles: []\n",  # empty lists
            "name: test\nmirrors: [not-a-url]\nfiles:\n  - {label: a, file_path: a}\n",
            "- just\n- a\n- list\n",  # wrong top-level type
        ],
    )
    def test_invalid_config_schema_exits(
        self, monkeypatch, tmp_path: Path, output_dir: Path, contents: str
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text(contents)

        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, bad, output_dir)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_data_error_exits(self, monkeypatch, config_path: Path, output_dir: Path):
        with patch.object(
            download_data, "download_dataset", side_effect=DownloadError("boom")
        ):
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, config_path, output_dir)

        assert exc.value.code == 1

    def test_missing_required_arguments_exits_with_usage_error(self, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch)

        assert exc.value.code == 2
