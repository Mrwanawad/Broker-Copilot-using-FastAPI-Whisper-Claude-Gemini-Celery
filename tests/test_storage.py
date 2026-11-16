"""Upload storage: local disk and S3-compatible (client stubbed, no network)."""

from pathlib import Path

from app import storage
from app.config import get_settings


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket, Key, Body):  # noqa: N803 - boto3 argument names
        self.objects[(Bucket, Key)] = Body

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[(bucket, key)])


def test_local_storage_roundtrip(tmp_path):
    ref = storage.put(b"RIFF", ".wav")
    assert not ref.startswith("s3://")
    assert storage.fetch(ref, tmp_path).read_bytes() == b"RIFF"


def test_s3_storage_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BC_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("BC_S3_BUCKET", "calls")
    get_settings.cache_clear()
    fake = FakeS3()
    monkeypatch.setattr(storage, "_s3", lambda: fake)
    ref = storage.put(b"OggS", ".ogg")
    assert ref.startswith("s3://calls/uploads/") and ref.endswith(".ogg")
    local = storage.fetch(ref, tmp_path / "work")
    assert local.read_bytes() == b"OggS" and local.parent == tmp_path / "work"
