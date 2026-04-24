#!/usr/bin/env python3
"""
Build a SSRN-ready PDF from the working-paper markdown.

Usage
-----
    python scripts/build_pdf.py

Inputs  (auto-discovered):
    paper/*.md            - single markdown file (the working paper)
    figures/*.png         - all figures referenced from the paper markdown

Output:
    paper/Kull_2026_Cramer_Holds_vs_Recommends.pdf

Requirements
------------
    - pandoc  >= 2.9
    - xelatex (TeX Live; Ubuntu: `sudo apt install texlive-xetex texlive-fonts-recommended texlive-fonts-extra`)

Fonts: the script uses ``fontconfig`` (``fc-list``) to prefer TeX Gyre Termes / Heros /
Cursor when registered; on macOS without those OTFs in the system catalog (e.g. BasicTeX),
it falls back to Times New Roman, Helvetica, Menlo. Override with environment variables
``CRAMER_PDF_MAINFONT``, ``CRAMER_PDF_SANSFONT``, ``CRAMER_PDF_MONOFONT`` if needed.

Source-of-truth policy
----------------------
Everything in the PDF comes from the markdown. The script parses these fields
from the paper's top matter and passes them to pandoc unchanged:

    # **Title**              -> document title
    **Author:** X            -> author name
    **Affiliation:** Y       -> \\thanks{} footnote (first line)
    **Email:** z@...         -> \\thanks{} footnote (second line)
    *Working Paper, MMM YYYY* (or legacy *This version: MMM YYYY*) -> pandoc ``date:``
    (rendered as ``Working Paper, MMM YYYY``; no separate “This version” line)
    **JEL classification:** ..  -> JEL block under abstract
    ## **Abstract**  ...     -> abstract block
    **Keywords:** ...        -> keywords block under abstract

If any field is missing from the markdown, the script aborts with a clear
error rather than silently inventing content.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import indent
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO / "paper"
FIGURES_DIR = REPO / "figures"
OUTPUT_PDF = PAPER_DIR / "Kull_2026_Cramer_Holds_vs_Recommends.pdf"

# Unicode characters that Latin Modern / TeX Gyre don't ship glyphs for.
# These are mapped to their LaTeX math-mode equivalents via `newunicodechar`.
UNICODE_MAP: dict[str, str] = {
    "≥": r"\ensuremath{\geq}",
    "≤": r"\ensuremath{\leq}",
    "≈": r"\ensuremath{\approx}",
    "≠": r"\ensuremath{\neq}",
    "±": r"\ensuremath{\pm}",
    "∞": r"\ensuremath{\infty}",
    "⁻": r"\ensuremath{{}^{-}}",
    "⁺": r"\ensuremath{{}^{+}}",
    "₀": r"\ensuremath{{}_{0}}",
    "₁": r"\ensuremath{{}_{1}}",
    "α": r"\ensuremath{\alpha}",
    "β": r"\ensuremath{\beta}",
    "ρ": r"\ensuremath{\rho}",
    "μ": r"\ensuremath{\mu}",
    "σ": r"\ensuremath{\sigma}",
    "Δ": r"\ensuremath{\Delta}",
    "×": r"\ensuremath{\times}",
    "·": r"\ensuremath{\cdot}",
    "…": r"\ldots",
    "−": r"\ensuremath{-}",
    "✓": r"\ensuremath{\checkmark}",
}


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------


def _check_binary(name: str) -> None:
    if shutil.which(name) is None:
        sys.exit(
            f"error: required binary '{name}' not found on PATH. "
            f"Install it and try again (see the header of this script)."
        )


def check_dependencies() -> None:
    for b in ("pandoc", "xelatex"):
        _check_binary(b)


def _fc_list_lower() -> str:
    """Lowercased ``fc-list`` output, or empty if fontconfig is unavailable."""
    try:
        r = subprocess.run(
            ["fc-list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _platform_font_trio() -> tuple[str, str, str]:
    """Last-resort fonts when ``fc-list`` is empty or no candidate matches."""
    plat = sys.platform
    if plat == "darwin":
        return ("Times New Roman", "Helvetica", "Menlo")
    if plat == "win32":
        return ("Times New Roman", "Segoe UI", "Consolas")
    return ("Liberation Serif", "Liberation Sans", "Liberation Mono")


def _first_font_in_fc(candidates: list[str], fc_text: str) -> Optional[str]:
    if not fc_text:
        return None
    for name in candidates:
        if name.lower() in fc_text:
            return name
    return None


def resolve_xelatex_fonts() -> tuple[str, str, str]:
    """Return (mainfont, sansfont, monofont) for ``fontspec`` / XeLaTeX.

    Prefer TeX Gyre when registered with fontconfig; otherwise fall back to
    common system fonts so the PDF builds on macOS without TeX Gyre OTFs.
    """
    fc = _fc_list_lower()
    p_main, p_sans, p_mono = _platform_font_trio()

    main = _first_font_in_fc(
        [
            "TeX Gyre Termes",
            "TeX Gyre Pagella",
            "TeX Gyre Schola",
            "Times New Roman",
            "Liberation Serif",
            "DejaVu Serif",
        ],
        fc,
    ) or p_main
    sans = _first_font_in_fc(
        [
            "TeX Gyre Heros",
            "TeX Gyre Adventor",
            "Arial",
            "Helvetica",
            "Segoe UI",
            "Liberation Sans",
        ],
        fc,
    ) or p_sans
    mono = _first_font_in_fc(
        [
            "TeX Gyre Cursor",
            "Menlo",
            "Consolas",
            "Courier New",
            "Liberation Mono",
            "DejaVu Sans Mono",
        ],
        fc,
    ) or p_mono

    if (e := os.environ.get("CRAMER_PDF_MAINFONT", "").strip()):
        main = e
    if (e := os.environ.get("CRAMER_PDF_SANSFONT", "").strip()):
        sans = e
    if (e := os.environ.get("CRAMER_PDF_MONOFONT", "").strip()):
        mono = e

    return (main, sans, mono)


# ---------------------------------------------------------------------------
# Source discovery + parsing
# ---------------------------------------------------------------------------


def find_paper_markdown() -> Path:
    """Return the single .md file under paper/. Error if zero or >1 found."""
    candidates = sorted(PAPER_DIR.glob("*.md"))
    # Ignore obvious ancillary files.
    candidates = [p for p in candidates if p.name.lower() != "ssrn_submission.md"]
    if not candidates:
        sys.exit(f"error: no *.md file in {PAPER_DIR}")
    if len(candidates) > 1:
        sys.exit(
            f"error: multiple *.md files in {PAPER_DIR}; specify which one to build:\n"
            + "\n".join(f"  - {p.name}" for p in candidates)
        )
    return candidates[0]


def _extract_field(src: str, pattern: str, label: str) -> str:
    m = re.search(pattern, src)
    if not m:
        sys.exit(
            f"error: could not find '{label}' in the paper markdown. "
            f"Expected pattern: {pattern!r}"
        )
    return m.group(1).strip()


def _extract_paper_date_line(src: str) -> str:
    """Return the pandoc ``date:`` value (always ``Working Paper, MMM YYYY``)."""
    m = re.search(r"\*Working Paper,\s*(.+?)\*", src)
    if m:
        return f"Working Paper, {m.group(1).strip()}"
    m = re.search(r"\*This version:\s*(.+?)\.?\*", src)
    if m:
        return f"Working Paper, {m.group(1).strip()}"
    sys.exit(
        "error: could not find a top-matter date line in the paper markdown. "
        "Expected *Working Paper, April 2026* or *This version: April 2026.*"
    )


def parse_paper(md_path: Path) -> dict:
    src = md_path.read_text(encoding="utf-8")

    # Title: first-level heading with surrounding ** (the paper uses # **...**)
    title_match = re.search(r"^#\s*\*\*(.+?)\*\*\s*$", src, re.M)
    if not title_match:
        sys.exit("error: expected the paper to start with '# **Title**'")
    title = title_match.group(1).strip()

    author = _extract_field(src, r"\*\*Author:\*\*\s*(.+)", "Author")
    affiliation = _extract_field(src, r"\*\*Affiliation:\*\*\s*(.+)", "Affiliation")
    email = _extract_field(src, r"\*\*Email:\*\*\s*(.+)", "Email")
    paper_date = _extract_paper_date_line(src)
    jel = _extract_field(
        src, r"\*\*JEL classification:\*\*\s*(.+)", "JEL classification"
    ).rstrip(".")

    # Abstract: between "## **Abstract**" and the next "---" separator.
    abs_match = re.search(
        r"##\s*\*\*Abstract\*\*\s*\n\n(.+?)\n---\n", src, re.S
    )
    if not abs_match:
        sys.exit("error: could not find the Abstract section")
    abstract_block = abs_match.group(1).strip()

    # Keywords are the last bold line inside the abstract block.
    kw_match = re.search(
        r"\*\*Keywords:\*\*\s*(.+)", abstract_block
    )
    if not kw_match:
        sys.exit("error: could not find **Keywords:** inside the abstract block")
    keywords = kw_match.group(1).strip().rstrip(".")

    # The abstract proper is everything before the keywords line.
    abstract = re.sub(
        r"\n\s*\*\*Keywords:\*\*.*$", "", abstract_block, flags=re.S
    ).strip()

    # Body: everything after the abstract's closing '---'.
    body_start = abs_match.end()
    body = src[body_start:].lstrip()

    return {
        "title": title,
        "author": author,
        "affiliation": affiliation,
        "email": email,
        "paper_date": paper_date,
        "jel": jel,
        "abstract": abstract,
        "keywords": keywords,
        "body": body,
    }


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def rewrite_image_paths(body: str, paper_dir: Path) -> str:
    """Rewrite markdown image paths to absolute paths for the build file."""

    def repl(match: re.Match) -> str:
        alt, path = match.group(1), match.group(2)
        if path.startswith(("http://", "https://", "/")):
            return match.group(0)

        if path.startswith(("./", "../")):
            resolved = (paper_dir / path).resolve()
        else:
            # Backward-compatible support for older paper markdown that used
            # bare figure filenames while the PNGs lived in figures/.
            resolved = (FIGURES_DIR / path).resolve()
        return f"![{alt}]({resolved})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, body)


def build_yaml_header(meta: dict) -> str:
    """Render pandoc YAML front-matter from parsed fields.

    Note: the author block is NOT set here — it is passed to pandoc via
    ``-V author=...`` on the command line so the raw LaTeX linebreaks
    (``\\\\``) survive without being escaped by pandoc's markdown reader.
    """
    # Title: insert a LaTeX line break after the colon, if the title has one.
    raw_title = meta["title"].replace("–", "--").replace("—", "---")
    if ":" in raw_title:
        left, _, right = raw_title.partition(":")
        title_yaml = f"{left.strip()}:\\\n  {right.strip()}"
    else:
        title_yaml = raw_title

    lines = [
        "---",
        "title: |",
        f"  {title_yaml}",
        f'date: "{meta["paper_date"]}"',
        "abstract: |",
    ]
    lines.append(indent(meta["abstract"], "  "))
    lines.append(f'keywords: "{meta["keywords"]}"')
    lines.append(f'jel: "{meta["jel"]}"')
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def build_author_tex(meta: dict) -> str:
    """Return a raw-LaTeX author block: name / affiliation / email on 3 lines.

    This string is passed to pandoc via ``-V author=...`` so the
    ``\\\\`` line breaks go straight into the LaTeX ``\\author{...}`` command
    without being processed by pandoc's markdown reader.
    """
    affiliation_escaped = meta["affiliation"].replace("&", r"\&")
    return (
        f"{meta['author']} \\\\ "
        f"\\textit{{{affiliation_escaped}}} \\\\ "
        f"\\href{{mailto:{meta['email']}}}{{{meta['email']}}}"
    )


def build_latex_includes() -> str:
    """LaTeX preamble: Unicode character map and caption / float tweaks."""
    lines = [r"\usepackage{newunicodechar}"]
    for ch, repl in UNICODE_MAP.items():
        # Use curly braces around the char to avoid TeX weirdness.
        lines.append(rf"\newunicodechar{{{ch}}}{{{repl}}}")
    lines.extend(
        [
            r"\usepackage{float}",
            r"\makeatletter\renewcommand*{\fps@figure}{H}\makeatother",
            r"\usepackage[font=small,labelfont=bf,skip=6pt]{caption}",
            r"\usepackage{xurl}",
            r"\renewenvironment{abstract}%",
            r"  {\begin{center}\textbf{Abstract}\end{center}%",
            r"   \begin{quote}\small\noindent\ignorespaces}%",
            r"  {\end{quote}}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_keywords_block(meta: dict) -> str:
    # One blank line after the abstract, then JEL/keywords; introduction starts
    # on a new page.
    return (
        r"\vspace{\baselineskip}" + "\n"
        r"\begin{center}" + "\n"
        r"\small" + "\n"
        rf"\noindent\textbf{{Keywords:}} {meta['keywords']}.\\[0.6\baselineskip]" + "\n"
        rf"\noindent\textbf{{JEL classification:}} {meta['jel']}." + "\n"
        r"\end{center}" + "\n"
        r"\clearpage" + "\n"
    )


def compile_pdf(
    build_md: Path,
    header_tex: Path,
    keywords_tex: Path,
    out_pdf: Path,
    author_tex: str,
) -> None:
    main_f, sans_f, mono_f = resolve_xelatex_fonts()
    print(f"[build_pdf] fonts:   {main_f} / {sans_f} / {mono_f}")
    cmd = [
        "pandoc",
        str(build_md),
        "-o",
        str(out_pdf),
        "--pdf-engine=xelatex",
        "-V",
        f"author={author_tex}",
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=11pt",
        "-V",
        "documentclass=article",
        "-V",
        "linestretch=1.15",
        "-V",
        f"mainfont={main_f}",
        "-V",
        f"sansfont={sans_f}",
        "-V",
        f"monofont={mono_f}",
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=NavyBlue",
        "-V",
        "urlcolor=NavyBlue",
        "-V",
        "citecolor=NavyBlue",
        "-H",
        str(header_tex),
        f"--include-before-body={keywords_tex}",
    ]
    print("[build_pdf] running: pandoc (via xelatex)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(f"error: pandoc failed with exit code {result.returncode}")
    # Surface only non-"Missing character" warnings (which are noise if the
    # UNICODE_MAP is complete for this paper's glyph set).
    warnings = [
        ln
        for ln in (result.stderr or "").splitlines()
        if ln.strip() and "Missing character" not in ln
    ]
    if warnings:
        print("[build_pdf] pandoc warnings:")
        for w in warnings[:20]:
            print(f"  {w}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    check_dependencies()
    md = find_paper_markdown()
    print(f"[build_pdf] source: {md}")
    meta = parse_paper(md)

    print(f"[build_pdf] title:   {meta['title']}")
    print(f"[build_pdf] author:  {meta['author']}")
    print(f"[build_pdf] date:    {meta['paper_date']}")

    body = rewrite_image_paths(meta["body"], md.parent)
    yaml_header = build_yaml_header(meta)
    build_md_text = yaml_header + body

    build_dir = REPO / "build"
    build_dir.mkdir(exist_ok=True)
    build_md = build_dir / "paper_build.md"
    header_tex = build_dir / "header.tex"
    keywords_tex = build_dir / "keywords_block.tex"

    build_md.write_text(build_md_text, encoding="utf-8")
    header_tex.write_text(build_latex_includes(), encoding="utf-8")
    keywords_tex.write_text(build_keywords_block(meta), encoding="utf-8")

    author_tex = build_author_tex(meta)
    compile_pdf(build_md, header_tex, keywords_tex, OUTPUT_PDF, author_tex)

    size_mb = OUTPUT_PDF.stat().st_size / 1024 / 1024
    print(f"[build_pdf] wrote {OUTPUT_PDF}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
