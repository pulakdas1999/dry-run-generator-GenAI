import os
import re
import sys
import time
import json
import hashlib
import argparse
from collections import deque
from datetime import date
from pathlib import Path

import markdown as md
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, AuthenticationError
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CACHE_FILE = OUTPUT_DIR / ".dryrun_cache.json"   # one hidden file holding finished dry runs, so reruns skip them

# Make emoji print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load variables from .env before anything else runs
load_dotenv()

# ---------- Settings (all overridable from .env) ----------
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
REASONING_EFFORT = os.getenv("LLM_REASONING", "low")      # low | medium | high (gpt-oss models only)
TPM_BUDGET = int(os.getenv("LLM_TPM_BUDGET", "7000"))     # stay under Groq's 8K tokens/minute free limit
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4000"))

COOLDOWN_SECONDS = 60   # wait after an unexpected API error, then move on to the next file
MAX_RETRIES = 3         # per-minute rate-limit retries for the same file
MAX_LINES = 300         # reject very large files to control cost and keep traces accurate

if not API_KEY:
    raise RuntimeError(
        "No API key found. Create a .env file with: GROQ_API_KEY=your_key_here"
    )

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

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


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


# ---------- LLM call + rate-limit handling ----------

class TokenPacer:
    """Keeps tokens-per-minute under budget by waiting before a request if needed."""

    def __init__(self, budget: int, window: int = 60):
        self.budget = budget
        self.window = window
        self.events = deque()      # (timestamp, tokens used)
        self.estimate = 4000       # guess for the next request; updated after each call

    def wait(self) -> None:
        while True:
            now = time.time()
            while self.events and now - self.events[0][0] >= self.window:
                self.events.popleft()
            used = sum(t for _, t in self.events)
            if not self.events or used + self.estimate <= self.budget:
                return
            pause = self.window - (now - self.events[0][0]) + 1
            print(f"   Pacing: waiting {pause:.0f}s to stay under the token limit...")
            time.sleep(max(pause, 1))

    def record(self, tokens: int) -> None:
        self.events.append((time.time(), tokens))
        self.estimate = max(1500, tokens)


pacer = TokenPacer(TPM_BUDGET)


def dry_run(code: str, language: str = "Java") -> tuple[str, int]:
    """Returns (markdown text, total tokens used)."""
    kwargs = dict(
        model=MODEL,
        temperature=0.2,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Language: {language}\n\nDry run this code:\n\n```{language.lower()}\n{code}\n```"},
        ],
    )
    if "gpt-oss" in MODEL:
        kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    text = choice.message.content or ""

    if choice.finish_reason == "length":
        text += ("\n\n🟥 NOTE\n\nThe model's answer was cut off because it hit the output limit. "
                 "Increase LLM_MAX_OUTPUT_TOKENS in .env to get the full trace.")

    tokens = response.usage.total_tokens if response.usage else pacer.estimate
    return text, tokens


def is_daily_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return "per day" in msg or "(tpd)" in msg or "(rpd)" in msg


def retry_after_seconds(err: Exception, default: float = 60) -> float:
    try:
        value = err.response.headers.get("retry-after")
        return max(float(value), 5) if value else default
    except Exception:
        return default


# ---------- Saved results (one hidden JSON file, no .md files) ----------

def load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


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


def normalize(text: str) -> str:
    """Ensure a blank line around each section marker so tables and code fences render."""
    text = re.sub(rf"(?m)^([{MARKERS}][^\n]*)\n(?!\n)", r"\1\n\n", text)
    text = re.sub(rf"(?m)([^\n])\n([{MARKERS}])", r"\1\n\n\2", text)
    return text


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
    parser = argparse.ArgumentParser(description="Dry run Java files with an LLM and save one PDF.")
    parser.add_argument("path", nargs="?", help="A Java file OR a folder containing Java files")
    parser.add_argument("-r", "--recursive", action="store_true", help="Also search sub-folders")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore saved results and regenerate every file")
    args = parser.parse_args()

    target = Path((args.path or input("Enter path to a file or folder: "))
                  .strip().strip('"').strip("'")).expanduser()

    files = collect_files(target, args.recursive)
    if not files:
        print("No .java files found.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    cache = {} if args.fresh else load_cache()
    today = date.today().strftime("%Y-%m-%d")
    pdf_path = OUTPUT_DIR / f"DryRun_{today}.pdf"
    print(f"Model: {MODEL}")
    print(f"Found {len(files)} file(s). Building: {pdf_path}\n")

    programs, failed, not_attempted = [], [], []
    daily_limit_hit = False

    for n, src in enumerate(files, start=1):
        print(f"[{n}/{len(files)}] {src.name}")

        # Problems with the file itself (too long, unreadable): no need to wait
        try:
            code, language = read_code(str(src))
        except Exception as e:
            failed.append((src.name, str(e)))
            print(f"   SKIPPED: {e}")
            continue

        # Already done earlier (e.g. yesterday)? Reuse it, no API call needed
        cache_key = f"{src.stem}_{hashlib.sha1(code.encode('utf-8')).hexdigest()[:8]}"
        if cache_key in cache:
            programs.append((src.name, to_body(cache[cache_key])))
            print("   loaded from saved results")
            continue

        result = None
        for attempt in range(1, MAX_RETRIES + 1):
            pacer.wait()
            try:
                result, tokens = dry_run(code, language)
                pacer.record(tokens)
                break
            except AuthenticationError:
                print("   Your API key was rejected. Check GROQ_API_KEY in .env.")
                sys.exit(1)
            except RateLimitError as e:
                if is_daily_limit(e):
                    daily_limit_hit = True
                    print("   DAILY LIMIT reached.")
                    break
                wait = retry_after_seconds(e)
                print(f"   Per-minute limit hit (attempt {attempt}/{MAX_RETRIES}). Waiting {wait:.0f}s...")
                time.sleep(wait)
            except Exception as e:
                failed.append((src.name, str(e)))
                print(f"   API ERROR: {e}")
                if n < len(files):
                    print(f"   Waiting {COOLDOWN_SECONDS}s before the next file...")
                    time.sleep(COOLDOWN_SECONDS)
                break
        else:
            failed.append((src.name, "still rate-limited after retries"))

        if daily_limit_hit:
            not_attempted = files[n - 1:]   # this file and everything after it
            break

        if result is not None:
            cache[cache_key] = result
            save_cache(cache)
            programs.append((src.name, to_body(result)))
            print("   done")

    if not programs:
        print("\nNothing to write: no file succeeded.")
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(build_document(programs, today))
        page.pdf(path=str(pdf_path), format="A4", print_background=True)
        browser.close()

    print(f"\nSaved: {pdf_path.resolve()}")
    print(f"{len(programs)} included, {len(failed)} failed, {len(not_attempted)} not attempted.")
    for name, err in failed:
        print(f"  - {name}: {err}")
    if daily_limit_hit:
        print("\nThe free daily limit was reached. Run the same command again after it resets;")
        print("finished files are remembered and will be reused, so only the remaining ones are sent.")