"""build.py - figures + references + paper.pdf from paper.html with headless Chromium (no TeX needed).

    python paper/build.py            # writes paper/figures/* and paper/paper.pdf
    python paper/build.py --pages DIR  # also rasterize every page to DIR for a layout check
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def chromium() -> str:
    for c in [os.environ.get("CHROME", ""), shutil.which("chromium") or "", shutil.which("google-chrome") or "",
              *sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))]:
        if c and os.path.exists(c):
            return c
    raise SystemExit("no Chromium found; set CHROME=/path/to/chrome")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="", help="directory for PNG renders of every page (layout check)")
    ap.add_argument("--skip_figures", action="store_true")
    args = ap.parse_args()
    if not args.skip_figures:
        subprocess.run([sys.executable, str(HERE / "figures.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "refs.py")], check=True)
    pdf = HERE / "paper.pdf"
    subprocess.run([chromium(), "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={pdf}", (HERE / "paper.html").as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("PDF:", pdf)
    if args.pages:
        os.makedirs(args.pages, exist_ok=True)
        subprocess.run(["pdftoppm", "-r", "80", "-png", str(pdf), os.path.join(args.pages, "page")], check=True)
        print("pages:", sorted(os.listdir(args.pages)))


if __name__ == "__main__":
    main()
