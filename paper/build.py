"""build.py - figures and paper.pdf from paper/latex/main.tex (pdflatex + bibtex).

    python paper/build.py                # writes paper/figures/* and paper/paper.pdf
    python paper/build.py --pages DIR    # also rasterize every page to DIR for a layout check
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEX = HERE / "latex"


def latex() -> Path:
    def run(*cmd: str) -> None:
        r = subprocess.run(cmd, cwd=TEX, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0:
            sys.stdout.write(r.stdout[-4000:])
            raise SystemExit(f"{cmd[0]} failed")

    for tool in ("pdflatex", "bibtex"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found; install TeX Live (e.g. apt install texlive-latex-extra texlive-fonts-recommended)")
    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main")
    run("bibtex", "main")
    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main")
    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main")
    log = (TEX / "main.log").read_text(errors="replace")
    for needle in ("undefined references", "Citation `", "Reference `"):
        if needle in log:
            print("warning:", next(l for l in log.splitlines() if needle in l))
    pdf = HERE / "paper.pdf"
    shutil.copyfile(TEX / "main.pdf", pdf)
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="", help="directory for PNG renders of every page (layout check)")
    ap.add_argument("--skip_figures", action="store_true")
    args = ap.parse_args()
    if not args.skip_figures:
        subprocess.run([sys.executable, str(HERE / "figures.py")], check=True, cwd=ROOT)
    pdf = latex()
    print("PDF:", pdf)
    if args.pages:
        os.makedirs(args.pages, exist_ok=True)
        subprocess.run(["pdftoppm", "-r", "80", "-png", str(pdf), os.path.join(args.pages, "page")], check=True)
        print("pages:", sorted(os.listdir(args.pages)))


if __name__ == "__main__":
    main()
