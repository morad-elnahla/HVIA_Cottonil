"""Read-only local deployment preflight. No network, rebuild or training."""
from pathlib import Path
from importlib import import_module
import json
from package_delivery import delivery_files, PROCESSED_FILES


def check(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    missing = [name for name in PROCESSED_FILES if not (root / name).is_file()]
    errors = []
    for module in ("streamlit", "src.assistant", "src.health", "src.llm", "src.inventory", "src.models"):
        try:
            import_module(module)
        except Exception as error:
            errors.append({"module": module, "error_type": type(error).__name__})
    for name in ("app.py", *[str(p.relative_to(root)) for p in (root / "src").glob("*.py")]):
        try:
            compile((root / name).read_text(encoding="utf-8"), name, "exec")
        except (SyntaxError, OSError) as error:
            errors.append({"file": name, "error_type": type(error).__name__})
    return {"ready_for_local_start": not missing and not errors, "missing_data_files": missing,
            "errors": errors, "network_tested": False, "authentication": "requires private host or authenticated gateway"}


if __name__ == "__main__":
    delivery_files()  # Verify the source allowlist and secret exclusions, too.
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready_for_local_start"] else 1)
