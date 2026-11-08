from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neural_network.cli import process_data
from neural_network.config.data import DatasetConfig
from neural_network.data.registry import DATASET_REGISTRY
from neural_network.errors import (
    DatasetConfigError,
    InvalidFileError,
    ParseError,
    ReadError,
    WriteError,
)


TEST_DATASET_NAME = "test-dataset"
TEST_CONFIG_PATH = Path("configs/datasets/test.yaml")


@pytest.fixture
def dataset_config(tmp_path: Path) -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "name": "test",
            "mirrors": ["https://mirror.example.com/"],
            "files": [{"label": "images", "file_name": "images.gz"}],
            "raw_data_loc": tmp_path / "raw",
            "processed_data_loc": tmp_path / "processed",
        }
    )


@pytest.fixture
def mock_load(monkeypatch, dataset_config: DatasetConfig) -> MagicMock:
    monkeypatch.setitem(
        process_data.DATASET_REGISTRY, TEST_DATASET_NAME, TEST_CONFIG_PATH
    )
    mock = MagicMock(return_value=dataset_config)
    monkeypatch.setattr(process_data, "load_dataset_config", mock)
    return mock


@pytest.fixture
def mock_process(monkeypatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr(process_data, "process_mnist_dataset", mock)
    return mock


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr("sys.argv", ["process-data", *args])
    process_data.main()


class TestMain:
    def test_loads_config_registered_under_dataset_name(
        self, monkeypatch, mock_load: MagicMock, mock_process: MagicMock
    ):
        run_main(monkeypatch, TEST_DATASET_NAME)

        mock_load.assert_called_once_with(TEST_CONFIG_PATH)

    @pytest.mark.parametrize("dataset_name", list(DATASET_REGISTRY))
    def test_accepts_every_registered_dataset(
        self,
        monkeypatch,
        mock_load: MagicMock,
        mock_process: MagicMock,
        dataset_name: str,
    ):
        run_main(monkeypatch, dataset_name)

        mock_load.assert_called_once_with(DATASET_REGISTRY[dataset_name])

    def test_passes_loaded_config_to_process_mnist_dataset(
        self,
        monkeypatch,
        mock_load: MagicMock,
        mock_process: MagicMock,
        dataset_config: DatasetConfig,
    ):
        run_main(monkeypatch, TEST_DATASET_NAME)

        mock_process.assert_called_once_with(
            dataset_config=dataset_config, overwrite=False
        )

    def test_overwrite_flag_is_passed_through(
        self, monkeypatch, mock_load: MagicMock, mock_process: MagicMock
    ):
        run_main(monkeypatch, TEST_DATASET_NAME, "--overwrite")

        assert mock_process.call_args.kwargs["overwrite"] is True

    def test_config_error_exits_without_processing(
        self, monkeypatch, mock_load: MagicMock, mock_process: MagicMock
    ):
        mock_load.side_effect = DatasetConfigError("bad config")

        with patch.object(process_data, "logger") as mock_logger:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, TEST_DATASET_NAME)

        assert exc.value.code == 1
        mock_process.assert_not_called()
        message = mock_logger.error.call_args.args[0]
        assert "Failed to process dataset" in message
        assert "bad config" in message

    @pytest.mark.parametrize(
        "error",
        [
            ReadError("boom"),
            WriteError("boom"),
            ParseError("boom"),
            InvalidFileError("boom"),
        ],
    )
    def test_processing_data_error_exits(
        self,
        monkeypatch,
        mock_load: MagicMock,
        mock_process: MagicMock,
        error: Exception,
    ):
        mock_process.side_effect = error

        with patch.object(process_data, "logger") as mock_logger:
            with pytest.raises(SystemExit) as exc:
                run_main(monkeypatch, TEST_DATASET_NAME)

        assert exc.value.code == 1
        assert "Failed to process dataset" in mock_logger.error.call_args.args[0]

    def test_missing_raw_files_exits(
        self, monkeypatch, mock_load: MagicMock, mock_process: MagicMock
    ):
        mock_process.side_effect = FileNotFoundError("missing raw files")

        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, TEST_DATASET_NAME)

        assert exc.value.code == 1

    def test_unexpected_error_is_not_swallowed(
        self, monkeypatch, mock_load: MagicMock, mock_process: MagicMock
    ):
        mock_process.side_effect = ValueError("bug")

        with pytest.raises(ValueError, match="bug"):
            run_main(monkeypatch, TEST_DATASET_NAME)

    def test_missing_required_arguments_exits_with_usage_error(
        self, monkeypatch, mock_load: MagicMock
    ):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch)

        assert exc.value.code == 2
        mock_load.assert_not_called()

    def test_unknown_dataset_name_exits_with_usage_error(
        self, monkeypatch, mock_load: MagicMock, mock_process: MagicMock
    ):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, "not-a-dataset")

        assert exc.value.code == 2
        mock_load.assert_not_called()
        mock_process.assert_not_called()

    def test_help_describes_command(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            run_main(monkeypatch, "--help")

        help_text = capsys.readouterr().out
        assert exc.value.code == 0
        assert "processed_data_loc" in help_text
        assert "--overwrite" in help_text
        assert "Run download-data first" in help_text
        assert "Example: process-data mnist" in help_text
