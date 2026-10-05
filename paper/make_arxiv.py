"""make_arxiv.py - pack paper/latex into an arXiv source zip and prove it compiles the way arXiv does.

    python paper/build.py && python paper/make_arxiv.py      # writes paper/arxiv_source.zip

The zip holds only files LaTeX actually reads (found with pdflatex -recorder): main.tex with figure paths
flattened to figures/, the generated main.bbl (arXiv does not run BibTeX), the local style files and the
figures. It is test-compiled in a clean directory with pdflatex only before it is written.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "latex"
FIG = HERE / "figures"
OUT = HERE / "arxiv_source.zip"


def pdflatex(cwd: Path, *extra: str) -> str:
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", *extra, "main"], cwd=cwd,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        sys.stdout.write(r.stdout[-3000:])
        raise SystemExit(f"pdflatex failed in {cwd}")
    return r.stdout


def used_files() -> tuple[list[Path], list[Path]]:
    """Local inputs (style files) and figures that pdflatex opened, from its -recorder log."""
    if not (TEX / "main.bbl").exists():
        raise SystemExit("main.bbl missing: run python paper/build.py first")
    pdflatex(TEX, "-recorder")
    local, figs = set(), set()
    for line in (TEX / "main.fls").read_text().splitlines():
        if not line.startswith("INPUT "):
            continue
        p = (TEX / line[6:]).resolve()
        if p.parent == TEX and p.suffix in {".sty", ".bst", ".cls", ".tex", ".bbl"} and p.name != "main.tex":
            local.add(p)
        elif p.parent == FIG.resolve() and p.suffix in {".pdf", ".png", ".jpg"}:
            figs.add(p)
    return sorted(local), sorted(figs)


def main() -> None:
    local, figs = used_files()
    src = (TEX / "main.tex").read_text()
    src, n = re.subn(r"\\graphicspath\{\{\.\./figures/\}\}", r"\\graphicspath{{figures/}}", src)
    if n != 1:
        raise SystemExit("could not rewrite \\graphicspath")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "src"
        (stage / "figures").mkdir(parents=True)
        (stage / "main.tex").write_text(src)
        for p in local:
            shutil.copy(p, stage / p.name)
        for p in figs:
            shutil.copy(p, stage / "figures" / p.name)
        names = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
        # compile exactly as arXiv would: pdflatex only, using the shipped .bbl
        test = Path(tmp) / "test"
        shutil.copytree(stage, test)
        for _ in range(3):
            log = pdflatex(test)
        if re.search(r"undefined (references|citations)|Citation .* undefined|Reference .* undefined", log):
            raise SystemExit("undefined references in the arXiv test build")
        pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
            for name in names:
                z.write(stage / name, name)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, test build: {pages.group(1) if pages else '?'} pages)")
    for name in names:
        print("  ", name)


if __name__ == "__main__":
    main()
