from pathlib import Path
from unittest.mock import patch

import pytest

from neural_network.config.data import DatasetConfig
from neural_network.errors import ParseError
from neural_network.scripts import process_data


VALID_YAML = """
name: test
mirrors:
  - https://mirror.example.com/
files:
  - label: images
    file_name: images.gz
raw_data_loc: {raw_data_loc}
processed_data_loc: {processed_data_loc}
"""


@pytest.fixture
def raw_data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "raw"
    path.mkdir()
    return path


@pytest.fixture
def processed_data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "processed"
    path.mkdir()
    return path


def write_config(tmp_path: Path, raw_data_loc: Path, processed_data_loc: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        VALID_YAML.format(
            raw_data_loc=raw_data_loc, processed_data_loc=processed_data_loc
        )
    )
    return path


@pytest.fixture
def config_path(tmp_path: Path, raw_data_dir: Path, processed_data_dir: Path) -> Path:
    return write_config(tmp_path, raw_data_dir, processed_data_dir)


def run_main(monkeypatch: pytest.MonkeyPatch, *args: object) -> None:
    monkeypatch.setattr("sys.argv", ["process_data", *map(str, args)])
    process_data.main()


class TestMain:
    def test_success_calls_process_mnist_dataset(
        self,
        monkeypatch,
        config_path: Path,
        raw_data_dir: Path,
        processed_data_dir: Path,
    ):
        with patch.object(process_data, "process_mnist_dataset") as mock_process:
            run_main(monkeypatch, config_path)

        mock_process.assert_called_once()
        kwargs = mock_process.call_args.kwargs
        assert isinstance(kwargs["dataset_config"], DatasetConfig)
        assert kwargs["dataset_config"].name == "test"
        assert kwargs["dataset_config"].raw_data_loc == raw_data_dir
        assert kwargs["dataset_config"].processed_data_loc == processed_data_dir
        assert kwargs["overwrite"] is False

    def test_overwrite_flag_is_passed_through(self, monkeypatch, config_path: Path):
        with patch.object(process_data, "process_mnist_dataset") as mock_process:
            run_main(monkeypatch, config_path, "--overwrite")

        assert mock_process.call_args.kwargs["overwrite"] is True

    def test_missing_config_file_exits(self, monkeypatch, tmp_path: Path):
        with patch.object(process_data, "process_mnist_dataset") as mock_process:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, tmp_path / "missing.yaml")

        assert exc.value.code == 1
        mock_process.assert_not_called()

    def test_config_path_that_is_a_directory_exits(self, monkeypatch, tmp_path: Path):
        with patch.object(process_data, "process_mnist_dataset") as mock_process:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, tmp_path)

        assert exc.value.code == 1
        mock_process.assert_not_called()

    def test_missing_raw_data_dir_exits(
        self, monkeypatch, tmp_path: Path, processed_data_dir: Path
    ):
        config = write_config(tmp_path, tmp_path / "nope", processed_data_dir)

        with patch.object(process_data, "process_mnist_dataset") as mock_process:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, config)

        assert exc.value.code == 1
        mock_process.assert_not_called()

    def test_missing_processed_data_dir_exits(
        self, monkeypatch, tmp_path: Path, raw_data_dir: Path
    ):
        config = write_config(tmp_path, raw_data_dir, tmp_path / "nope")

        with patch.object(process_data, "process_mnist_dataset") as mock_process:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, config)

        assert exc.value.code == 1
        mock_process.assert_not_called()

    def test_invalid_yaml_exits(self, monkeypatch, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: [unclosed")

        with patch.object(process_data, "process_mnist_dataset") as mock_process:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, bad)

        assert exc.value.code == 1
        mock_process.assert_not_called()

    def test_unreadable_config_exits(self, monkeypatch, config_path: Path):
        with patch("builtins.open", side_effect=PermissionError("denied")):
            with patch.object(process_data, "process_mnist_dataset") as mock_process:
                with pytest.raises(SystemExit) as exc:
                    run_main(monkeypatch, config_path)

        assert exc.value.code == 1
        mock_process.assert_not_called()

    @pytest.mark.parametrize(
        "contents",
        [
            "name: test\n",  # missing mirrors and files
            "name: test\nmirrors: []\nfiles: []\n",  # empty lists
            "name: test\nmirrors: [https://x.com/]\nfiles:\n  - {label: a}\n",
            "- just\n- a\n- list\n",  # wrong top-level type
        ],
    )
    def test_invalid_config_schema_exits(
        self, monkeypatch, tmp_path: Path, contents: str
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text(contents)

        with patch.object(process_data, "process_mnist_dataset") as mock_process:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, bad)

        assert exc.value.code == 1
        mock_process.assert_not_called()

    @pytest.mark.parametrize(
        "exc_to_raise", [ParseError("boom"), FileNotFoundError("missing raw files")]
    )
    def test_processing_error_exits(
        self, monkeypatch, config_path: Path, exc_to_raise: Exception
    ):
        with patch.object(
            process_data, "process_mnist_dataset", side_effect=exc_to_raise
        ):
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, config_path)

        assert exc.value.code == 1

    def test_missing_raw_files_exits(self, monkeypatch, config_path: Path):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, config_path)

        assert exc.value.code == 1

    def test_missing_required_arguments_exits_with_usage_error(self, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch)

        assert exc.value.code == 2
