#!/usr/bin/env python3
"""Build a version of the paper with the changes highlighted in yellow.

The script diffs the working copy of a .tex file against its committed version
with latexdiff, converts latexdiff's markup into a yellow highlight, and
compiles the result.

    ./highlight-changes.py                       # working tree vs. the default base commit
    ./highlight-changes.py -r da707a2            # vs. an explicit revision
    ./highlight-changes.py -r da707a2 -n HEAD    # two committed revisions
    ./highlight-changes.py --show-deleted        # also print the removed text in grey
    ./highlight-changes.py --keep                # keep the generated .tex next to the pdf

What ends up highlighted:

  * changed words and sentences get a yellow background;
  * blocks that latexdiff cannot mark inline -- whole new equations, figures,
    theorem environments, table rows -- get a thick yellow bar in the margin,
    because a background box around a display equation would break the layout.

Deleted text is dropped by default: this revision rewrote most of the paper, and
interleaving the old wording makes the result unreadable. Use --show-deleted if
the reviewer asks to see it.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# The commit the manuscript was in when it was submitted to SN Computer Science.
# Everything after it belongs to the current revision round.
DEFAULT_BASE_REV = "da707a2"
DEFAULT_TEX = "paper/jmlda-template-eng.tex"

# Chunks containing any of these are emitted without a \colorbox: they are
# alignment or structural markup that must not be swallowed by a box.
UNSAFE = ("\\\\", "&", "\\begin", "\\end", "\\item", "\\label", "\\caption")

PREAMBLE = r"""%DIF PREAMBLE EXTENSION ADDED BY highlight-changes.py
\RequirePackage{xcolor}
\RequirePackage[outerbars,color]{changebar}
\definecolor{diffhlcolor}{rgb}{1,0.94,0.35}
\definecolor{diffdelcolor}{gray}{0.55}
\setlength{\changebarwidth}{3pt}
\setlength{\changebarsep}{6pt}
\cbcolor{diffhlcolor}
% One highlighted chunk. \fboxsep is small so that neighbouring chunks read as
% one continuous marker stroke rather than as separate boxes.
\providecommand{\DIFhlw}[1]{{\setlength{\fboxsep}{1.1pt}\colorbox{diffhlcolor}{#1}}}
% The remaining \DIF* markers are resolved in the source by highlight-changes.py;
% these definitions only catch the ones it deliberately leaves alone.
\providecommand{\DIFadd}[1]{#1}
\providecommand{\DIFdel}[1]{DELETEDTEXTPLACEHOLDER}
\providecommand{\DIFaddbegin}{}
\providecommand{\DIFaddend}{}
\providecommand{\DIFdelbegin}{}
\providecommand{\DIFdelend}{}
\providecommand{\DIFaddFL}[1]{\DIFadd{#1}}
\providecommand{\DIFdelFL}[1]{\DIFdel{#1}}
\providecommand{\DIFaddbeginFL}{}
\providecommand{\DIFaddendFL}{}
\providecommand{\DIFdelbeginFL}{}
\providecommand{\DIFdelendFL}{}
"""


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        sys.exit(f"{cmd[0]} failed:\n{proc.stderr.strip() or proc.stdout.strip()}")
    return proc


def find_group(text: str, open_brace: int) -> int:
    """Index just past the '}' matching the '{' at open_brace."""
    depth = 0
    i = open_brace
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def split_chunks(body: str) -> list[str]:
    """Split on spaces that sit at brace depth 0 and outside math.

    Splitting blindly on every space cuts '$\\mathbf{q} \\in \\mathbb{R}^r$' in
    half and the result does not compile, so math and groups are kept whole.
    """
    chunks: list[str] = []
    cur: list[str] = []
    depth = 0      # {} groups
    bracket = 0    # [] optional arguments, e.g. \cite[\S 2]{...}
    math = False
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in "()[]":  # \( \) \[ \] toggle math
                math = nxt in "(["
            cur.append(body[i:i + 2])
            i += 2
            continue
        if c == "$":
            math = not math
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "[":
            bracket += 1
        elif c == "]":
            bracket = max(0, bracket - 1)
        elif c in " \n\t" and depth == 0 and bracket == 0 and not math:
            if cur:
                chunks.append("".join(cur))
                cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    if cur:
        chunks.append("".join(cur))
    return chunks


def wrap(chunk: str) -> str:
    unbalanced = (
        chunk.count("{") != chunk.count("}")
        or chunk.count("[") != chunk.count("]")
        or chunk.count("$") % 2
    )
    if unbalanced:
        return chunk
    if any(tok in chunk for tok in UNSAFE):
        return chunk
    return "\\DIFhlw{" + chunk + "}"


MATH_ENVS = {
    "equation", "equation*", "displaymath", "align", "align*", "alignat", "alignat*",
    "eqnarray", "eqnarray*", "gather", "gather*", "multline", "multline*", "split",
    "array", "cases", "aligned", "xy", "xymatrix",
}

_ENV = re.compile(r"\\(begin|end)\s*\{([^}]*)\}")


def math_spans(tex: str) -> list[tuple[int, int]]:
    """Character ranges that are in math mode.

    latexdiff happily writes \\DIFadd{...} inside \\begin{equation}, and
    \\colorbox is not allowed there, so those occurrences have to be left alone
    and marked with a margin bar instead.
    """
    spans: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []
    for m in _ENV.finditer(tex):
        kind, name = m.group(1), m.group(2)
        if kind == "begin":
            if name in MATH_ENVS:
                stack.append((name, m.start()))
        elif stack and stack[-1][0] == name:
            spans.append((stack.pop()[1], m.end()))
    # \[ ... \] and $ ... $
    depth_open = None
    i = 0
    while i < len(tex) - 1:
        two = tex[i : i + 2]
        if two == "\\[":
            depth_open = i
        elif two == "\\]" and depth_open is not None:
            spans.append((depth_open, i + 2))
            depth_open = None
        elif two == "\\$":
            i += 2
            continue
        elif tex[i] == "$":
            j = tex.find("$", i + 1)
            if j == -1:
                break
            spans.append((i, j + 1))
            i = j + 1
            continue
        i += 1
    return spans


def in_math(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def drop_command(tex: str, name: str) -> str:
    """Remove every `\\name{...}` together with its argument."""
    marker = "\\" + name + "{"
    out: list[str] = []
    i = 0
    while True:
        j = tex.find(marker, i)
        if j == -1:
            out.append(tex[i:])
            return "".join(out)
        end = find_group(tex, j + len(marker) - 1)
        if end == -1:
            out.append(tex[i:])
            return "".join(out)
        out.append(tex[i:j])
        i = end


def drop_spans(tex: str, begin: str, end: str) -> tuple[str, int]:
    """Remove every `begin ... end` region, markers included."""
    out: list[str] = []
    i = 0
    n = 0
    while True:
        a = tex.find(begin, i)
        if a == -1:
            out.append(tex[i:])
            return "".join(out), n
        b = tex.find(end, a + len(begin))
        if b == -1:
            out.append(tex[i:])
            return "".join(out), n
        out.append(tex[i:a])
        i = b + len(end)
        n += 1


def mark_blocks(tex: str) -> tuple[str, int]:
    """Turn \\DIFaddbegin ... \\DIFaddend into a margin bar, or drop it.

    latexdiff emits this pair hundreds of times in the middle of paragraphs.
    Mapping all of them to changebar commands wrecks the layout, so the bar is
    kept only where the pair encloses a display environment -- a new equation or
    a new theorem, which cannot carry a background highlight of its own.
    """
    begin, end = "\\DIFaddbegin", "\\DIFaddend"
    out: list[str] = []
    i = 0
    bars = 0
    while True:
        a = tex.find(begin, i)
        if a == -1:
            out.append(tex[i:])
            return "".join(out), bars
        b = tex.find(end, a + len(begin))
        if b == -1:
            out.append(tex[i:])
            return "".join(out), bars
        inner = tex[a + len(begin):b]
        out.append(tex[i:a])
        envs = {m.group(2) for m in _ENV.finditer(inner) if m.group(1) == "begin"}
        if envs & (MATH_ENVS | {"theorem", "lemma", "remark", "proof", "table", "figure"}):
            out.append("\\cbstart{}" + inner + "\\cbend{}")
            bars += 1
        else:
            out.append(inner)
        i = b + len(end)


def clean_markup(tex: str, show_deleted: bool) -> tuple[str, int]:
    """Remove the latexdiff markers that must not survive into the output.

    Two reasons. Inside a tabular a cell has to start with \\multicolumn, and
    even a macro expanding to nothing is a token in front of it, which makes TeX
    report a misplaced \\omit. And a deleted display equation is left behind by
    latexdiff as an empty displaymath, which prints as a blank vertical gap.
    """
    head, sep, rest = tex.partition(r"\begin{document}")
    if not sep:
        return tex, 0
    for name in ("DIFaddbeginFL", "DIFaddendFL", "DIFdelbeginFL", "DIFdelendFL"):
        rest = rest.replace("\\" + name, "")
    if show_deleted:
        rest = rest.replace("\\DIFdelbegin", "").replace("\\DIFdelend", "")
    else:
        rest = drop_command(rest, "DIFdelFL")
        rest, _ = drop_spans(rest, "\\DIFdelbegin", "\\DIFdelend")
    rest, bars = mark_blocks(rest)
    return head + sep + rest, bars


def highlight_adds(tex: str) -> tuple[str, int, int]:
    """Replace the body of every \\DIFadd / \\DIFaddFL by per-chunk \\DIFhlw."""
    spans = math_spans(tex)
    count = skipped = 0
    for marker in ("\\DIFaddFL{", "\\DIFadd{"):
        out: list[str] = []
        i = 0
        while True:
            j = tex.find(marker, i)
            if j == -1:
                out.append(tex[i:])
                break
            end = find_group(tex, j + len(marker) - 1)
            if end == -1:
                out.append(tex[i:])
                break
            body = tex[j + len(marker):end - 1]
            out.append(tex[i:j])
            # Do not rewrite the \providecommand lines of the preamble, and
            # never put a \colorbox into math mode.
            line_start = tex.rfind("\n", 0, j) + 1
            head = tex[line_start:j]
            if "providecommand" in head or "newcommand" in head:
                out.append(tex[j:end])
            elif in_math(j, spans):
                out.append(body)
                skipped += 1
            else:
                out.append(" ".join(wrap(c) for c in split_chunks(body)))
                count += 1
            i = end
        tex = "".join(out)
        spans = math_spans(tex)
    return tex, count, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tex", nargs="?", default=None, help=f"path to the .tex file (default: {DEFAULT_TEX})")
    ap.add_argument("-r", "--base", default=DEFAULT_BASE_REV, help=f"revision to compare against (default: {DEFAULT_BASE_REV})")
    ap.add_argument("-n", "--new", default=None, help="revision to compare (default: the working tree)")
    ap.add_argument("-o", "--output", default=None, help="output pdf (default: <name>-highlighted.pdf next to the source)")
    ap.add_argument("--show-deleted", action="store_true", help="print the removed text in grey instead of dropping it")
    ap.add_argument("--keep", action="store_true", help="keep the generated .tex next to the pdf")
    args = ap.parse_args()

    for tool in ("latexdiff", "pdflatex", "bibtex", "git"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found in PATH")

    repo = Path(run(["git", "rev-parse", "--show-toplevel"]).stdout.strip())
    tex = Path(args.tex).resolve() if args.tex else repo / DEFAULT_TEX
    if not tex.exists():
        sys.exit(f"{tex} does not exist")
    rel = tex.relative_to(repo).as_posix()
    texdir, stem = tex.parent, tex.stem

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "old.tex").write_text(
            run(["git", "show", f"{args.base}:{rel}"], cwd=repo).stdout, encoding="utf8"
        )
        if args.new:
            (tmp / "new.tex").write_text(
                run(["git", "show", f"{args.new}:{rel}"], cwd=repo).stdout, encoding="utf8"
            )
        else:
            shutil.copy(tex, tmp / "new.tex")

        preamble = PREAMBLE.replace(
            "\\providecommand{\\DIFdel}[1]{DELETEDTEXTPLACEHOLDER}",
            "\\providecommand{\\DIFdel}[1]{{\\color{diffdelcolor}[#1]}}"
            if args.show_deleted
            else "\\providecommand{\\DIFdel}[1]{}",
        )
        (tmp / "preamble.tex").write_text(preamble, encoding="utf8")

        # --math-markup=whole keeps formulas as single units, so a changed
        # formula is marked as a whole rather than shredded token by token.
        diff = run(
            [
                "latexdiff",
                "--encoding=utf8",
                "-p", str(tmp / "preamble.tex"),
                "--math-markup=whole",
                str(tmp / "old.tex"),
                str(tmp / "new.tex"),
            ]
        ).stdout

        diff, bars = clean_markup(diff, args.show_deleted)
        diff, n, skipped = highlight_adds(diff)
        print(f"{n} fragments highlighted, {skipped} skipped inside math, "
              f"{bars} blocks marked with a margin bar")

        # Compile inside the source directory so that the class, the bst files,
        # the bibliography and figures/ all resolve.
        work = texdir / f"{stem}-highlighted.tex"
        work.write_text(diff, encoding="utf8")
        aux_stem = work.stem
        try:
            for step in (
                ["pdflatex", "-interaction=nonstopmode", work.name],
                ["bibtex", aux_stem],
                ["pdflatex", "-interaction=nonstopmode", work.name],
                ["pdflatex", "-interaction=nonstopmode", work.name],
                # changebar needs one more pass: the bar positions come from the aux file.
                ["pdflatex", "-interaction=nonstopmode", work.name],
            ):
                run(step, cwd=texdir, check=False)

            pdf = texdir / f"{aux_stem}.pdf"
            if not pdf.exists():
                sys.exit(f"compilation produced no pdf, see {texdir / (aux_stem + '.log')}")
            out = Path(args.output).resolve() if args.output else pdf
            if out != pdf:
                shutil.move(pdf, out)

            log = (texdir / f"{aux_stem}.log").read_text(errors="replace")
            errors = [ln for ln in log.split("\n") if ln.startswith("! ")]
            pages = re.findall(r"\((\d+) pages", log)
            print(f"{out}: {pages[-1] if pages else '?'} pages, "
                  f"{len(errors)} TeX errors")
            for e in errors[:5]:
                print("   ", e[:120])
            if "There were undefined references" in log:
                print("    warning: undefined references, rerun or check the bibliography")
            if "There were undefined citations" in log:
                print("    warning: undefined citations, bibtex did not resolve them")
        finally:
            for ext in ("aux", "log", "out", "bbl", "blg", "synctex.gz", "cb", "cb2"):
                (texdir / f"{aux_stem}.{ext}").unlink(missing_ok=True)
            if not args.keep:
                work.unlink(missing_ok=True)
            else:
                print(f"kept {work}")


if __name__ == "__main__":
    main()
