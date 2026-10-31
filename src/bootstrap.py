"""Restore a versioned, verified processed bundle without rebuilding models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import urlsplit
from zipfile import ZipFile, BadZipFile

import requests

FILES = tuple(n + ".parquet" for n in (
    "sales", "branches", "products", "customers", "stock", "returns", "costs",
    "deliveries", "forecast", "backtest", "churn", "fquality",
)) + ("meta.json", "health.json", "model_metrics.json")
MAX_DOWNLOAD = 512 * 1024 * 1024
MAX_EXPANDED = 2 * 1024 * 1024 * 1024
_LOCK = threading.Lock()


class BundleError(RuntimeError):
    """Safe diagnostic: never includes URLs, credentials, or server bodies."""


def settings(secrets=None):
    keys = ("BI_DATA_URL", "BI_DATA_SHA256", "BI_DATA_TOKEN")
    values = {k: os.environ.get(k, "") for k in keys}
    if secrets is not None:
        try:
            for k in keys:
                values[k] = values[k] or str(secrets.get(k, ""))
        except FileNotFoundError:
            pass
    return values


def validate(directory: Path):
    """Check the complete saved-model/data bundle before it becomes visible."""
    import pyarrow.parquet as pq
    for name in FILES:
        p = directory / name
        if not p.is_file():
            raise BundleError("bundle_missing_file: " + name)
        try:
            if p.suffix == ".parquet":
                pq.read_metadata(p)
            else:
                json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            raise BundleError("bundle_invalid_file: " + name) from None


def unpack(archive: Path, target: Path):
    with ZipFile(archive) as z:
        infos = z.infolist()
        # The deployment artifact contains exactly the approved processed files.
        if len(infos) != len(FILES) or {i.filename for i in infos} != set(FILES):
            raise BundleError("bundle_invalid_manifest")
        if sum(i.file_size for i in infos) > MAX_EXPANDED:
            raise BundleError("bundle_too_large")
        for info in infos:
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise BundleError("bundle_invalid_entry")
            with z.open(info) as source, (target / info.filename).open("wb") as out:
                while chunk := source.read(1024 * 1024):
                    out.write(chunk)
    validate(target)


def ensure(local: Path, config: dict) -> Path:
    """Use local data unless a remote snapshot is explicitly configured.

    Each hash has its own immutable directory; partial downloads never replace
    a working snapshot. No automatic retries or model building happen here.
    """
    url = config.get("BI_DATA_URL", "").strip()
    digest = config.get("BI_DATA_SHA256", "").strip().lower()
    if not url and not digest:
        return local
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise BundleError("data_config: BI_DATA_URL must be an HTTPS download URL")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise BundleError("data_config: BI_DATA_SHA256 must contain 64 hex characters")
    base = local.parent.parent if local.parent.name == "snapshots" else local.parent
    target = base / "snapshots" / digest
    with _LOCK:
        if (target / ".verified").is_file() and all((target / n).is_file() for n in FILES):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise BundleError("bundle_cache_incomplete: remove the incomplete snapshot and restart")
        with tempfile.TemporaryDirectory(prefix="download-", dir=target.parent) as work:
            temp = Path(work)
            archive = temp / "bundle.zip"
            headers = {}
            if config.get("BI_DATA_TOKEN"):
                headers["Authorization"] = "Bearer " + config["BI_DATA_TOKEN"]
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
                    if response.status_code != 200:
                        raise BundleError(f"data_download_http_{response.status_code}")
                    if urlsplit(response.url).scheme != "https":
                        raise BundleError("data_download_insecure_redirect")
                    sha = hashlib.sha256()
                    size = 0
                    with archive.open("wb") as out:
                        for chunk in response.iter_content(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_DOWNLOAD:
                                raise BundleError("data_download_too_large")
                            sha.update(chunk)
                            out.write(chunk)
                if sha.hexdigest() != digest:
                    raise BundleError("data_checksum_mismatch")
                staged = temp / "processed"
                staged.mkdir()
                unpack(archive, staged)
                (staged / ".verified").write_text(digest, encoding="ascii")
                staged.rename(target)
            except requests.Timeout:
                raise BundleError("data_download_timeout") from None
            except requests.RequestException:
                raise BundleError("data_download_connection_failed") from None
            except (BadZipFile, OSError):
                raise BundleError("data_bundle_or_storage_error") from None
    return target
