import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

from datetime import datetime
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
"""

client = genai.Client(api_key=API_KEY)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry run a source code file using Gemini.")
    parser.add_argument("path", nargs="?", help="Path to the code file (e.g. Main.java)")
    parser.add_argument("-o", "--output", help="Optional: custom path for the output .md file")
    args = parser.parse_args()

    path = args.path or input("Enter path to your code file: ")
    path = path.strip().strip('"').strip("'")

    code, language = read_code(path)
    result = dry_run(code, language)
    print(result)

    # Save into the outputs/ folder (created if missing)
    if args.output:
        out_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{Path(path).stem}_dryrun_{stamp}.md"

    out_path.write_text(result, encoding="utf-8")
    print(f"\nSaved to: {out_path.resolve()}")

# python .\main.py C:\Users\user\Downloads\JAVA_PRACTICE\practiceProgramsBlueJ\armstrongNumber.java