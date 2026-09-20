

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "onevoice-dolphin-kaggle-bundle.zip"

CODE_FILES = [
    "pyproject.toml",
    "bench/__init__.py",
    "bench/_common.py",
    "bench/_dolphin.py",
    "bench/_dolphin_mouth.py",
    "bench/bench_dolphin_offline.py",
    "scripts/vendor_dolphin.py",
    "scripts/export_dolphin_separation.py",
]

def main() -> None:
    tt = ROOT / "data" / "dolphin_tier_a" / "tt"
    if not (tt / "mix.json").is_file():
        raise SystemExit(f"missing {tt / 'mix.json'}")

    if OUT.exists():
        OUT.unlink()

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in CODE_FILES:
            p = ROOT / rel
            if not p.is_file():
                raise SystemExit(f"missing code file: {p}")
            zf.write(p, f"onevoice/{rel}")

        pkg = ROOT / "src" / "onevoice"
        for f in pkg.rglob("*.py"):
            arc = f"onevoice/{f.relative_to(ROOT).as_posix()}"
            zf.write(f, arc)

        for f in tt.rglob("*"):
            if f.is_file():
                arc = f"data/dolphin_tier_a/tt/{f.relative_to(tt).as_posix()}"
                zf.write(f, arc)

        nb = ROOT / "notebooks" / "dolphin_gpu_probe.ipynb"
        if nb.is_file():
            zf.write(nb, "notebooks/dolphin_gpu_probe.ipynb")

    names: list[str] = []
    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()

    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(names)} entries)")
    assert any(n.endswith("tt/mix.json") for n in names)
    assert any("bench_dolphin_offline.py" in n for n in names)
    assert any(n.startswith("onevoice/src/onevoice/") for n in names)

if __name__ == "__main__":
    main()
