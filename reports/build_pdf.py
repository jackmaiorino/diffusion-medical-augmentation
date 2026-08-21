"""Render a report markdown file to PDF via headless Chrome (mermaid blocks included)."""

import argparse
import base64
import mimetypes
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
]

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs"

CSS = """
@page { size: letter; margin: 0.85in 0.9in; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.5pt; line-height: 1.45;
       color: #111; max-width: 100%; }
h1 { font-size: 17pt; margin: 0 0 0.3em; line-height: 1.25; }
h2 { font-size: 13pt; margin: 1.5em 0 0.4em; border-bottom: 1px solid #ccc;
     padding-bottom: 0.15em; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 1.2em 0 0.3em; page-break-after: avoid; }
p { margin: 0.5em 0; }
code { font-family: "Consolas", monospace; font-size: 9pt; background: #f4f4f4;
       padding: 0.1em 0.15em; border-radius: 2px; hyphens: none; }
pre { background: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 3px;
      padding: 0.6em 0.8em; font-size: 8.5pt; line-height: 1.35; overflow-x: auto;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 0.28em 0.5em; }
th { background: #f0f0f0; text-align: left; }
blockquote { margin: 0.9em 0; padding: 0.5em 1em; border-left: 3px solid #999;
             background: #fafafa; font-style: italic; page-break-inside: avoid; }
img { max-width: 100%; page-break-inside: avoid; }
em { color: #444; }
ul { list-style: none; margin: 0.5em 0; padding-left: 1.4em; }
ol { list-style-type: decimal; margin: 0.5em 0; padding-left: 1.4em; }
li { margin: 0.25em 0; }
/* keep the bibliography compact enough to avoid a nearly empty trailing page */
h2:last-of-type + ol { font-size: 9.25pt; line-height: 1.28; }
h2:last-of-type + ol > li { margin: 0.12em 0; }
/* the serif stack renders a broken disc glyph, so draw the bullet ourselves */
ul > li { position: relative; }
ul > li::before { content: "\\2022"; font-family: Arial, sans-serif; position: absolute;
                  left: -0.9em; top: -0.02em; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.2em 0; }
.mermaid { text-align: center; margin: 1em 0; page-break-inside: avoid; }
/* cap the height so a tall flowchart flows with the text instead of taking a whole page */
.mermaid svg { max-width: 100%; max-height: 4in; width: auto; height: auto; }
"""

MERMAID_INIT = f"""
<script type="module">
  import mermaid from "{MERMAID_CDN}";
  mermaid.initialize({{ startOnLoad: true, theme: "neutral",
                        themeVariables: {{ fontSize: "13px" }} }});
</script>
"""


def find_chrome() -> Path:
    """Return the first Chrome executable that exists, or exit with a message."""
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    sys.exit("Chrome not found. Edit CHROME_CANDIDATES in build_pdf.py.")


def break_inline_paths(html: str) -> str:
    """Let inline code paths wrap after slashes so justified lines don't stretch."""

    def add_breaks(match: re.Match) -> str:
        return "<code>" + match.group(1).replace("/", "/<wbr>") + "</code>"

    # only inline <code>; fenced blocks render as <pre><code> and wrap on their own
    return re.sub(r"(?<!<pre>)<code>([^<]*)</code>", add_breaks, html)


def embed_images(html: str, base_dir: Path) -> str:
    """Inline local <img> sources as data URIs so the PDF is self-contained."""

    def replace(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (base_dir / src).resolve()
        if not path.exists():
            print(f"  warning: image not found, skipping: {src}")
            return match.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        payload = base64.b64encode(path.read_bytes()).decode()
        return match.group(0).replace(src, f"data:{mime};base64,{payload}")

    return re.sub(r'<img[^>]*src="([^"]+)"', replace, html)


MERMAID_TOKEN = "xxMERMAIDBLOCK{}xx"


def extract_mermaid(md_text: str) -> tuple[str, list[str]]:
    """Swap ```mermaid fences for placeholders so markdown cannot mangle the diagram source."""
    blocks: list[str] = []

    def stash(match: re.Match) -> str:
        blocks.append(match.group(1))
        return MERMAID_TOKEN.format(len(blocks) - 1)

    # the closing fence match must not eat the blank line that ends the paragraph,
    # or the placeholder merges into the caption below it
    pattern = re.compile(r"^```mermaid[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)
    return pattern.sub(stash, md_text), blocks


def restore_mermaid(html: str, blocks: list[str]) -> str:
    """Put the untouched diagram source back as <div class="mermaid"> for mermaid.js."""
    for i, block in enumerate(blocks):
        token = MERMAID_TOKEN.format(i)
        html = html.replace(f"<p>{token}</p>", f'<div class="mermaid">\n{block}</div>')
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", type=Path, help="source .md file")
    parser.add_argument("-o", "--output", type=Path, help="output .pdf (default: alongside source)")
    args = parser.parse_args()

    src = args.markdown.resolve()
    out = (args.output or src.with_suffix(".pdf")).resolve()

    md_text = src.read_text(encoding="utf-8")
    md_text, diagrams = extract_mermaid(md_text)
    n_diagrams = len(diagrams)

    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
    )
    body = restore_mermaid(body, diagrams)
    body = embed_images(body, src.parent)
    body = break_inline_paths(body)

    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{src.stem}</title><style>{CSS}</style>"
        f"{MERMAID_INIT if n_diagrams else ''}"
        f"</head><body>{body}</body></html>"
    )

    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        # virtual-time-budget lets the mermaid module load and draw before the print snapshot
        subprocess.run(
            [
                str(find_chrome()),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--virtual-time-budget=15000",
                f"--print-to-pdf={out}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )

    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {n_diagrams} mermaid diagram(s))")


if __name__ == "__main__":
    main()
