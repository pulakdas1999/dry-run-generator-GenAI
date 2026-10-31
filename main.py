import os
import re
import sys
import time
import argparse
from datetime import date
from pathlib import Path

import markdown as md
from dotenv import load_dotenv
from google import genai
from google.genai import types
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Make emoji print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load variables from .env before anything else runs
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Create a .env file with: GEMINI_API_KEY=your_key_here"
    )

MODEL = "gemini-2.5-flash"
COOLDOWN_SECONDS = 60  # wait this long after any API error, then move on
MAX_LINES = 300  # reject very large files to control cost and keep traces accurate

LANGUAGES = {
    ".java": "Java",
    ".py": "Python",
    ".cpp": "C++",
    ".c": "C",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".cs": "C#",
    ".go": "Go",
}

SYSTEM_PROMPT = """
You are a programming teacher who explains code to students by doing a DRY RUN.
You will receive source code (often Java, but it can be any language). Produce an exact,
step-by-step dry run that a student can follow on paper.

## Colour-coded sections (use these markers exactly, and always in this order)
🟦 CODE        - the code segment being traced (quote only the relevant lines, with line numbers)
🟨 DRY RUN     - the step-by-step trace (tables and diagrams)
🟩 OUTPUT      - what is printed or returned, exactly
🟥 NOTE        - only if there is a bug, exception, infinite loop, or edge case (otherwise omit)

Repeat the 🟦 → 🟨 → 🟩 group for each logical part of the program (e.g. initialisation,
each loop, each method call). Finish with one final "🟩 FINAL OUTPUT" block.

## Rules for the dry run
1. Start with a one-line "🟦 INPUT" section: state each input, its data type and its initial value.
2. Use a trace table for every loop or sequence of changing variables:
   | Step | Line | Variables after this step | What happened |
   Keep the "What happened" column to a few words, e.g. "c = a + b = 1 + 1 = 2".
3. Use simple ASCII diagrams whenever they make things clearer:
   - Arrays/lists: show cells and mark pointers, e.g.
       index:  0   1   2   3
              [5] [2] [8] [1]
                   ^i
   - Recursion: show the call stack or a call tree, and show values returning back up.
   - Objects/linked lists: show boxes and arrows (A -> B -> C -> null).
4. Be concise, with no filler and no long paragraphs. But NEVER skip any value that
   affects the output. Every printed line must appear in the trace.
5. For long loops (more than 8 iterations): show the first 3 iterations and the last
   iteration in full, then write "... iterations 4 to N-1 follow the same pattern ..." and
   give the pattern as a one-line formula. Still state the final variable values and the
   complete output.
6. Compute every value carefully and double-check arithmetic. Never guess a value.
7. If the code has no fixed input (e.g. it uses Scanner), assume a small sample input,
   and state clearly at the top what you assumed.
8. Do not rewrite or improve the code, and do not add theory. Only trace it.
9. Output plain Markdown only. No intro like "Sure!" and no closing remarks.
10. Put each section marker on its own line, followed by a blank line, then the content.
    Use fenced code blocks only for code and printed output, and tables only for traces.
"""

client = genai.Client(api_key=API_KEY)


# ---------- Reading code ----------

def read_code(path_str: str) -> tuple[str, str]:
    """Read the source file and return (code, language)."""
    path = Path(path_str.strip().strip('"').strip("'")).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    code = path.read_text(encoding="utf-8", errors="replace")
    if len(code.splitlines()) > MAX_LINES:
        raise ValueError(f"File has more than {MAX_LINES} lines. Please trim it.")

    language = LANGUAGES.get(path.suffix.lower(), "Java")
    return code, language


# ---------- Gemini call ----------

def dry_run(code: str, language: str = "Java") -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Language: {language}\n\nDry run this code:\n\n```{language.lower()}\n{code}\n```",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text


# ---------- Formatting ----------

MARKERS = "🟦🟨🟩🟥"
COLOURS = {"🟦": "blue", "🟨": "yellow", "🟩": "green", "🟥": "red"}

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:30px auto;background:#f6f7f9;color:#222}
.sec{border-left:6px solid;border-radius:6px;padding:4px 16px;margin:14px 0;background:#fff}
.sec h4{margin:10px 0 6px}
.blue{border-color:#3b82f6;background:#eff6ff}
.yellow{border-color:#eab308;background:#fefce8}
.green{border-color:#22c55e;background:#f0fdf4}
.red{border-color:#ef4444;background:#fef2f2}
table{border-collapse:collapse;margin:8px 0;width:100%}
th,td{border:1px solid #cbd5e1;padding:6px 10px;text-align:left}
th{background:#e2e8f0}
pre{background:#1e1e2e;color:#cdd6f4;padding:10px;border-radius:6px;overflow-x:auto}
code{font-family:Consolas,monospace}
"""


def normalize(text: str) -> str:
    """Ensure a blank line around each section marker so tables and code fences render."""
    text = re.sub(rf"(?m)^([{MARKERS}][^\n]*)\n(?!\n)", r"\1\n\n", text)
    text = re.sub(rf"(?m)([^\n])\n([{MARKERS}])", r"\1\n\n\2", text)
    return text


PDF_CSS = """
@page{size:A4;margin:14mm}
body{max-width:none;margin:0;background:#fff}
h1{font-family:Segoe UI,Arial,sans-serif;margin:0 0 4px}
.date{color:#555;margin-bottom:18px}
.program{break-before:page}
.program:first-of-type{break-before:auto}
.program h2{margin:0 0 10px;padding-bottom:6px;border-bottom:2px solid #cbd5e1}
.sec{break-inside:auto}
tr{break-inside:avoid}
pre{white-space:pre-wrap;word-break:break-word}
"""


def to_body(text: str) -> str:
    """Convert one dry-run Markdown result into coloured section blocks."""
    chunks = re.split(rf"(?m)^(?=[{MARKERS}])", normalize(text))
    body = ""
    for chunk in chunks:
        if not chunk.strip():
            continue
        cls = COLOURS.get(chunk.strip()[0], "")
        first, _, rest = chunk.partition("\n")
        html = md.markdown(rest, extensions=["tables", "fenced_code"])
        body += f'<div class="sec {cls}"><h4>{first.strip()}</h4>{html}</div>\n'
    return body


def build_document(programs: list[tuple[str, str]], today: str) -> str:
    """programs = [(file name, section html), ...] -> one HTML document."""
    parts = "".join(
        f'<section class="program"><h2>{name}</h2>{body}</section>\n'
        for name, body in programs
    )
    return (
        f"<html><head><meta charset='utf-8'><style>{CSS}{PDF_CSS}</style></head><body>"
        f"<h1>Dry Run Report</h1><div class='date'>{today}</div>{parts}</body></html>"
    )


# ---------- Processing ----------

def collect_files(target: Path, recursive: bool) -> list[Path]:
    if target.is_dir():
        pattern = "**/*.java" if recursive else "*.java"
        return sorted(target.glob(pattern))
    if target.is_file():
        return [target]
    raise FileNotFoundError(f"Path not found: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry run Java files using Gemini and save one PDF.")
    parser.add_argument("path", nargs="?", help="A Java file OR a folder containing Java files")
    parser.add_argument("-r", "--recursive", action="store_true", help="Also search sub-folders")
    args = parser.parse_args()

    target = Path((args.path or input("Enter path to a file or folder: "))
                  .strip().strip('"').strip("'")).expanduser()

    files = collect_files(target, args.recursive)
    if not files:
        print("No .java files found.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")
    pdf_path = OUTPUT_DIR / f"DryRun_{today}.pdf"
    print(f"Found {len(files)} file(s). Building: {pdf_path}\n")

    programs, failed = [], []
    for n, src in enumerate(files, start=1):
        print(f"[{n}/{len(files)}] {src.name}")

        # Problems with the file itself (too long, unreadable): no need to wait
        try:
            code, language = read_code(str(src))
        except Exception as e:
            failed.append((src.name, str(e)))
            print(f"   SKIPPED: {e}")
            continue

        # API call: on any error, cool down for a minute, then go to the next file
        try:
            result = dry_run(code, language)
            programs.append((src.name, to_body(result)))
            print("   done")
            time.sleep(1)  # small pause between successful calls
        except Exception as e:
            failed.append((src.name, str(e)))
            print(f"   API ERROR: {e}")
            if n < len(files):
                print(f"   Waiting {COOLDOWN_SECONDS}s before the next file...")
                time.sleep(COOLDOWN_SECONDS)

    if not programs:
        print("\nNothing to write: every file failed.")
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(build_document(programs, today))
        page.pdf(path=str(pdf_path), format="A4", print_background=True)
        browser.close()

    print(f"\nSaved: {pdf_path.resolve()}")
    print(f"{len(programs)} included, {len(failed)} failed.")
    for name, err in failed:
        print(f"  - {name}: {err}")