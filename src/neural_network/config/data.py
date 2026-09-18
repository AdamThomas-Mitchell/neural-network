from pydantic import BaseModel


class FileInfo(BaseModel):
    label: str
    file_path: str
    checksum: str | None = None


class DatasetConfig(BaseModel):
    name: str
    mirrors: list[str]
    files: list[FileInfo]
