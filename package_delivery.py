"""Create a source-only delivery ZIP; approved processed data is explicit opt-in.

No API requests, model training, data rebuild, or third-party dependencies.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "app.py", "requirements.txt", "requirements.lock", "README.md", ".env.example", ".gitignore",
    "Dockerfile", ".dockerignore", "DEPLOYMENT.md", "RELEASE_REPORT.md", "release_check.py", "live_acceptance.py",
    "package_delivery.py", ".streamlit/config.toml", "photo/1.png",
    "src/__init__.py", "src/assistant.py", "src/data.py", "src/insights.py",
    "src/inventory.py", "src/llm.py", "src/metrics.py", "src/models.py", "src/ui.py", "src/health.py",
    "tests/conftest.py", "tests/test_core.py", "tests/test_assistant_regression.py", "tests/test_release.py", "tests/README.md",
)
PROCESSED_FILES = tuple("data/processed/" + name for name in (
    "sales.parquet", "branches.parquet", "products.parquet", "customers.parquet",
    "stock.parquet", "returns.parquet", "costs.parquet", "deliveries.parquet",
    "forecast.parquet", "backtest.parquet", "churn.parquet", "fquality.parquet",
    "meta.json", "health.json", "model_metrics.json",
))
SECRET = re.compile(r"(?:gsk_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def desktop() -> Path:
    """Use the Windows Desktop known-folder setting, including OneDrive redirection."""
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
            value, _ = winreg.QueryValueEx(key, "Desktop")
        return Path(os.path.expandvars(value))
    return Path.home() / "Desktop"


def delivery_files(include_processed: bool = False) -> list[tuple[Path, str]]:
    names = SOURCE_FILES + (PROCESSED_FILES if include_processed else ())
    selected = []
    for name in names:
        source = ROOT / name
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(ROOT) or not source.is_file():
            raise ValueError(f"Unsafe or missing delivery file: {name}")
        # Reject symlinks/junctions even if they currently point inside the project.
        if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
               for p in (source, *source.parents) if p != ROOT and p.is_relative_to(ROOT)):
            raise ValueError(f"Linked paths are not permitted: {name}")
        if source.suffix in (".py", ".md", ".toml", ".json", ".example"):
            content = source.read_text(encoding="utf-8")
            if SECRET.search(content):
                raise ValueError(f"Possible credential detected in {name}; archive not created.")
            if name == ".env.example":
                for line in content.splitlines():
                    key, sep, value = line.partition("=")
                    if sep and ("KEY" in key.upper() or "TOKEN" in key.upper() or "SECRET" in key.upper()) and value.strip():
                        raise ValueError("Secret fields in .env.example must be empty.")
        selected.append((source, name))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-processed", action="store_true", help="Include the complete existing processed dataset for authorized private delivery only.")
    parser.add_argument("--confirm-data-sharing", action="store_true", help="Confirm that every processed data file is approved for this recipient.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list files without creating an archive.")
    parser.add_argument("--output-dir", type=Path, help="Optional destination override; default is your Desktop.")
    args = parser.parse_args()
    if args.include_processed and not args.confirm_data_sharing:
        parser.error("Processed data needs explicit recipient approval: add --confirm-data-sharing only when authorized.")
    selected = delivery_files(args.include_processed)
    mode = "with-approved-data" if args.include_processed else "source-only"
    if args.dry_run:
        print(f"Mode: {mode}")
        for _, name in selected:
            print(name)
        return
    destination = args.output_dir or desktop()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"Cottonil_BI_{mode}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.zip"
    delivery_note = (
        "Cottonil BI - private deployment candidate.\n"
        "See DEPLOYMENT.md and RELEASE_REPORT.md for setup and verified limits.\n"
        "No .env, API keys, raw data, logs, caches, or virtual environments are included.\n"
        + ("Approved processed data is included. Keep this archive private and share only with the authorized recipient.\n"
           if args.include_processed else
           "SOURCE ONLY: this archive has no business data. The dashboard needs a complete authorized data/processed bundle supplied separately.\n")
    )
    try:
        with ZipFile(target, "x", compression=ZIP_DEFLATED) as archive:
            for source, name in selected:
                archive.write(source, f"Cottonil_BI/{name}")
            archive.writestr("Cottonil_BI/DELIVERY.txt", delivery_note)
        with ZipFile(target) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Archive integrity failed: {bad}")
    except Exception:
        if target.exists():
            target.unlink()
        raise
    print(f"Created: {target}")
    print(f"Mode: {mode}; {len(selected)} files plus DELIVERY.txt")


if __name__ == "__main__":
    main()
