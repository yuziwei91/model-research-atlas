from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist" / "supernode"
OUT = DIST / "research-bundle.zip"
FILES = [
    "report.md", "models.json", "sources.csv", "sources.json", "hardware.csv", "hardware.json",
    "precision-support.csv", "precision-support.json", "workloads.csv", "workloads.json",
    "formulas.csv", "ascend-priorities.csv", "ascend-priorities.json",
    "model-coverage.csv", "model-coverage.json", "scenario-matrix.csv", "scenario-matrix.json",
]
README = """Model Research Atlas: Supernode supplement

This is a supplemental research and budgeting package. It is not an execution guarantee,
hardware benchmark, or independently audited certification. Scenario values are editable
assumptions; inspect report.md and source registers before relying on them. The package
contains 156 model-scope supplements. Only Kimi-K3 has a scenario baseline in this study;
other models are scope/gap records and do not inherit K3 cache geometry.

CSV files use UTF-8 with BOM. Scenario capacity snapshots are a bounded matrix across
three representative platforms and do not claim backend support. The interactive calculator
and full hardware catalog remain available on the hosted site.
"""

paths = [DIST / name for name in FILES]
paths += sorted((DIST / "models").glob("*-supernode.csv"))
if any(not p.is_file() for p in paths):
    missing = [str(p) for p in paths if not p.is_file()]
    raise SystemExit("Missing bundle inputs: " + ", ".join(missing))
with ZipFile(OUT, "w", ZIP_DEFLATED, compresslevel=9) as archive:
    entries = [("README.txt", README.encode("utf-8"))]
    entries += [(p.relative_to(DIST).as_posix(), p.read_bytes()) for p in paths]
    for name, data in entries:
        info = ZipInfo(name, (2026, 10, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)
print(f"Created {OUT} with {len(paths) + 1} files")
