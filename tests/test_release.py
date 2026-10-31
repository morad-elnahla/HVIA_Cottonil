from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import package_delivery
import release_check


def test_delivery_contains_all_runtime_modules_and_no_private_files():
    names = {name for _, name in package_delivery.delivery_files()}
    root = Path(__file__).resolve().parents[1]
    assert {str(p.relative_to(root)).replace("\\", "/") for p in (root / "src").glob("*.py")} <= names
    assert {"src/health.py", "requirements.lock", "DEPLOYMENT.md", "release_check.py"} <= names
    assert not any(n.startswith(("data/", "logs/", ".env/")) or n == ".env" for n in names)


def test_source_archive_integrity(tmp_path):
    subprocess.run([sys.executable, str(package_delivery.ROOT / "package_delivery.py"), "--output-dir", str(tmp_path)], check=True, capture_output=True)
    archive = next(tmp_path.glob("*.zip"))
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert "Cottonil_BI/src/health.py" in z.namelist()
        assert not any(n.endswith("/.env") or "/data/" in n or "/logs/" in n for n in z.namelist())


def test_preflight_reports_missing_data_without_building_it(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "app.py").write_text("pass", encoding="utf-8")
    report = release_check.check(tmp_path)
    assert not report["ready_for_local_start"]
    assert "data/processed/sales.parquet" in report["missing_data_files"]
    assert not (tmp_path / "data").exists()
