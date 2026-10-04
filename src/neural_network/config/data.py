from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class FileInfo(BaseModel):
    label: str
    file_path: str
    checksum: str | None = None


# class DatasetConfig(BaseModel):
#     name: str
#     mirrors: list[HttpUrl] = Field(min_length=1)
#     files: list[FileInfo] = Field(min_length=1)


class DownloadConfig(BaseModel):
    mirrors: list[HttpUrl] = Field(min_length=1)
    files: list[FileInfo] = Field(min_length=1)


class DataFormat(StrEnum):
    gz = "gz"
    idx = "idx"
    npy = "npy"


class FormatTransition(BaseModel):
    initial: DataFormat
    final: DataFormat


ALLOWED_TRANSITIONS: list[FormatTransition] = [
    FormatTransition(
        initial=DataFormat.gz,
        final=DataFormat.idx,
    ),
    FormatTransition(
        initial=DataFormat.idx,
        final=DataFormat.npy,
    ),
]


class ProcessConfig(BaseModel):
    transitions: list[FormatTransition]

    # validate each element of transitions is in ALLOWED_TRANSITIONS


class DatasetConfig(BaseModel):
    name: str
    download_cfg: DownloadConfig
    process_cfg: ProcessConfig
