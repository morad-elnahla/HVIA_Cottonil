import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from src import bootstrap as B
from package_cloud_data import build


def snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in B.FILES:
        if name.endswith(".parquet"):
            pd.DataFrame({"test": [1]}).to_parquet(source / name)
        else:
            (source / name).write_text("{}")
    archive = tmp_path / "bundle.zip"
    digest = build(source, archive)
    return archive.read_bytes(), {"BI_DATA_URL": "https://private.example/bundle", "BI_DATA_SHA256": digest}


def transport(monkeypatch, payload, status=200):
    calls = []
    class Response:
        status_code = status
        url = "https://private.example/bundle"
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield payload
    def get(*args, **kwargs):
        calls.append(kwargs)
        return Response()
    monkeypatch.setattr(B.requests, "get", get)
    return calls


def test_restore_then_reuse_and_version_switch(tmp_path, monkeypatch):
    payload, cfg = snapshot(tmp_path)
    calls = transport(monkeypatch, payload)
    local = tmp_path / "data" / "processed"
    restored = B.ensure(local, cfg)
    B.validate(restored)
    assert B.ensure(restored, cfg) == restored
    assert len(calls) == 1
    assert not local.exists()


def test_local_is_untouched_without_remote(tmp_path):
    assert B.ensure(tmp_path, {}) == tmp_path


@pytest.mark.parametrize("failure", ["checksum", "unauthorized", "manifest", "invalid_file"])
def test_failed_snapshot_never_publishes(tmp_path, monkeypatch, failure):
    payload, cfg = snapshot(tmp_path)
    if failure in ("manifest", "invalid_file"):
        out = io.BytesIO()
        with ZipFile(out, "w") as z:
            for name in (B.FILES if failure == "invalid_file" else ["../escape"]):
                z.writestr(name, "invalid")
        payload = out.getvalue()
        cfg["BI_DATA_SHA256"] = hashlib.sha256(payload).hexdigest()
    if failure == "checksum":
        cfg["BI_DATA_SHA256"] = "0" * 64
    cfg["BI_DATA_TOKEN"] = "secret-placeholder"
    transport(monkeypatch, payload, 403 if failure == "unauthorized" else 200)
    with pytest.raises(B.BundleError) as error:
        B.ensure(tmp_path / "data" / "processed", cfg)
    assert "secret-placeholder" not in str(error.value)
    assert "https" not in str(error.value)
    assert not list(tmp_path.rglob(".verified"))
    assert not (tmp_path / "escape").exists()


def test_missing_hash_rejected_before_network(tmp_path):
    with pytest.raises(B.BundleError, match="SHA256"):
        B.ensure(tmp_path, {"BI_DATA_URL": "https://example.com/data"})


def test_cloud_cold_start_all_pages_with_actual_saved_bundle(tmp_path, monkeypatch):
    """No local processed directory, mocked download, real unchanged data + pages."""
    from streamlit.testing.v1 import AppTest
    from src import data
    root = Path(__file__).resolve().parents[1]
    archive = tmp_path / "actual.zip"
    digest = build(root / "data" / "processed", archive)
    calls = transport(monkeypatch, archive.read_bytes())
    monkeypatch.setattr(data, "PROC", tmp_path / "cloud" / "processed")
    monkeypatch.setenv("BI_DATA_URL", "https://private.example/bundle")
    monkeypatch.setenv("BI_DATA_SHA256", digest)
    monkeypatch.setenv("LLM_ENABLED", "false")
    at = AppTest.from_file(str(root / "app.py"), default_timeout=180).run()
    assert not at.exception
    assert len(at.sidebar.radio[0].options) == 9
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    assert len(calls) == 1
    assert data.PROC != root / "data" / "processed"
    assert (data.PROC / "forecast.parquet").read_bytes() == (root / "data/processed/forecast.parquet").read_bytes()
