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
    file_name: images.gz
    checksum: abc123
raw_data_loc: {raw_data_loc}
"""


@pytest.fixture
def raw_data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "raw"
    path.mkdir()
    return path


@pytest.fixture
def config_path(tmp_path: Path, raw_data_dir: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(VALID_YAML.format(raw_data_loc=raw_data_dir))
    return path


def run_main(monkeypatch: pytest.MonkeyPatch, *args: object) -> None:
    monkeypatch.setattr("sys.argv", ["download-data", *map(str, args)])
    download_data.main()


class TestMain:
    def test_success_calls_download_dataset(
        self, monkeypatch, config_path: Path, raw_data_dir: Path
    ):
        with patch.object(download_data, "download_dataset") as mock_download:
            run_main(monkeypatch, config_path)

        mock_download.assert_called_once()
        kwargs = mock_download.call_args.kwargs
        assert isinstance(kwargs["dataset_config"], DatasetConfig)
        assert kwargs["dataset_config"].name == "test"
        assert kwargs["dataset_config"].files[0].file_name == "images.gz"
        assert kwargs["dataset_config"].files[0].checksum == "abc123"
        assert kwargs["dataset_config"].raw_data_loc == raw_data_dir
        assert kwargs["overwrite"] is False

    def test_overwrite_flag_is_passed_through(self, monkeypatch, config_path: Path):
        with patch.object(download_data, "download_dataset") as mock_download:
            run_main(monkeypatch, config_path, "--overwrite")

        assert mock_download.call_args.kwargs["overwrite"] is True

    def test_missing_config_file_exits(self, monkeypatch, tmp_path: Path):
        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, tmp_path / "missing.yaml")

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_config_path_that_is_a_directory_exits(self, monkeypatch, tmp_path: Path):
        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, tmp_path)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_missing_raw_data_dir_exits(self, monkeypatch, tmp_path: Path):
        config = tmp_path / "config.yaml"
        config.write_text(VALID_YAML.format(raw_data_loc=tmp_path / "nope"))

        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, config)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_raw_data_loc_that_is_a_file_exits(self, monkeypatch, tmp_path: Path):
        not_a_dir = tmp_path / "file"
        not_a_dir.write_text("x")
        config = tmp_path / "config.yaml"
        config.write_text(VALID_YAML.format(raw_data_loc=not_a_dir))

        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, config)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_invalid_yaml_exits(self, monkeypatch, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: [unclosed")

        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, bad)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_unreadable_config_exits(self, monkeypatch, config_path: Path):
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with patch.object(download_data, "download_dataset") as mock_download:
                with pytest.raises(SystemExit) as exc:
                    run_main(monkeypatch, config_path)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    @pytest.mark.parametrize(
        "contents",
        [
            "name: test\n",  # missing mirrors and files
            "name: test\nmirrors: []\nfiles: []\n",  # empty lists
            "name: test\nmirrors: [not-a-url]\nfiles:\n  - {label: a, file_name: a}\n",
            "name: test\nmirrors: [https://x.com/]\nfiles:\n  - {label: a}\n",  # no file_name
            "- just\n- a\n- list\n",  # wrong top-level type
        ],
    )
    def test_invalid_config_schema_exits(
        self, monkeypatch, tmp_path: Path, contents: str
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text(contents)

        with patch.object(download_data, "download_dataset") as mock_download:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, bad)

        assert exc.value.code == 1
        mock_download.assert_not_called()

    def test_data_error_exits(self, monkeypatch, config_path: Path):
        with patch.object(
            download_data, "download_dataset", side_effect=DownloadError("boom")
        ):
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, config_path)

        assert exc.value.code == 1

    def test_missing_required_arguments_exits_with_usage_error(self, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch)

        assert exc.value.code == 2
