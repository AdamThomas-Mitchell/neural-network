from pydantic import BaseModel, Field, HttpUrl


class FileInfo(BaseModel):
    label: str
    file_path: str
    checksum: str | None = None


class DatasetConfig(BaseModel):
    name: str
    mirrors: list[HttpUrl] = Field(min_length=1)
    files: list[FileInfo] = Field(min_length=1)
