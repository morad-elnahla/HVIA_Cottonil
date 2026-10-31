"""Create a PRIVATE deployment snapshot. Never uploads or changes source data."""
from pathlib import Path
import hashlib
from zipfile import ZipFile, ZIP_DEFLATED

from src.bootstrap import FILES, validate

ROOT = Path(__file__).resolve().parent


def build(source: Path, output: Path):
    validate(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(source / name, name)
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


if __name__ == "__main__":
    output = ROOT / "dist" / "cottonil-cloud-data.zip"
    print("Private bundle:", output)
    print("BI_DATA_SHA256=" + build(ROOT / "data" / "processed", output))
