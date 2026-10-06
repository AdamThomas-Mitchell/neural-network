from pydantic import BaseModel, DirectoryPath, Field, HttpUrl


class FileInfo(BaseModel):
    label: str
    file_path: str
    checksum: str | None = None


class DatasetConfig(BaseModel):
    name: str
    mirrors: list[HttpUrl] = Field(min_length=1)
    files: list[FileInfo] = Field(min_length=1)
    raw_data_loc: DirectoryPath
    processed_data_loc: DirectoryPath
