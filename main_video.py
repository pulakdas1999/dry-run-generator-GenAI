r"""
Dry-run VIDEO generator.

For every Java file: the LLM returns a small JSON "storyboard" (one entry per executed step),
Python validates it with Pydantic, then draws one picture per step with Pillow and stitches them
into a short MP4. No AI is used to make the pictures or the video.

Install once:
    pip install openai python-dotenv pydantic pillow numpy imageio imageio-ffmpeg

Run:
    python .\main_video.py "C:\Users\user\Downloads\JAVA_PRACTICE\practiceProgramsBlueJ\armstrongNumber.java"
    python .\main_video.py "C:\Users\user\Downloads\JAVA_PRACTICE\practiceProgramsBlueJ" -r
"""
import os
import re
import sys
import time
import argparse
from collections import deque
from pathlib import Path
from typing import Annotated

try:
    import numpy as np
    import imageio.v2 as imageio
    import imageio_ffmpeg  # noqa: F401  (bundles the ffmpeg program, nothing else to install)
    from PIL import Image, ImageDraw, ImageFont
    from dotenv import load_dotenv
    from openai import OpenAI, RateLimitError, AuthenticationError, BadRequestError
    from pydantic import BaseModel, BeforeValidator, Field
except ImportError as e:
    raise SystemExit(
        f"Missing package: {e.name}\nInstall everything with:\n"
        "  pip install openai python-dotenv pydantic pillow numpy imageio imageio-ffmpeg"
    )

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Make text print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ---------- Settings (all overridable from .env) ----------
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
REASONING_EFFORT = os.getenv("LLM_REASONING", "low")
TPM_BUDGET = int(os.getenv("LLM_TPM_BUDGET", "7000"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "5000"))
VIDEO_MAX_STEPS = int(os.getenv("VIDEO_MAX_STEPS", "60"))            # longer traces get loops shortened
VIDEO_TARGET_SECONDS = float(os.getenv("VIDEO_TARGET_SECONDS", "75"))  # videos are sped up to fit this

COOLDOWN_SECONDS = 60
MAX_RETRIES = 3
MAX_LINES = 300

if not API_KEY:
    raise RuntimeError("No API key found. Create a .env file with: GROQ_API_KEY=your_key_here")

# ---------- Video look and timing ----------
W, H, FPS = 1280, 720, 15
TITLE_SECONDS = 3.5
FINAL_SECONDS = 5.0
PRE_SECONDS = 0.5          # "about to run this line" frame, before the variables change
MIN_STEP_SECONDS = 1.3     # shortest time a finished step stays on screen (before speed-up)
MAX_STEP_SECONDS = 4.0
MIN_POST_SECONDS = 0.7     # floor after speed-up
MIN_SCALE = 0.45           # never speed up by more than this

BG = (17, 17, 27)
PANEL = (30, 30, 46)
LINE_BG = (55, 58, 80)
TEXT = (205, 214, 244)
WHITE = (255, 255, 255)
DIM = (127, 132, 156)
DARK = (17, 17, 27)
BLUE = (59, 130, 246)       # code
YELLOW = (234, 179, 8)      # variables / the line being run
GREEN = (34, 197, 94)       # output
GREEN_TEXT = (134, 239, 172)
ORANGE = (251, 146, 60)     # a value that just changed
CHANGED_BG = (70, 52, 30)
BAR_BG = (49, 50, 68)

MAX_VARS = 8
CODE_ROWS = 17
OUT_ROWS = 3

SYSTEM_PROMPT = """
You are an expert programming teacher. You receive source code with line numbers and produce an exact
DRY RUN as JSON. A program will turn your JSON into a short step-by-step video, so follow the
format exactly.

Return ONLY one JSON object. No markdown fences, no text before or after it.

{
  "title": "what the program does, max 6 words",
  "summary": "one sentence, max 25 words, saying what the program does",
  "assumed_input": "inputs you assumed, e.g. n = 153 (assumed); empty string if there is no input",
  "steps": [
    {
      "line": 12,
      "explain": "detailed step explaining the math/logic calculation using real values, e.g. 'Compute 153 % 10 = 3'",
      "changes": {"sum": "27"},
      "removed": [],
      "printed": "",
      "skipped": false
    }
  ],
  "final_output": "the exact full text the program prints"
}

Rules:
1. "line" is the line number (as shown in the numbered source) of the statement being executed
   in that step. Use only line numbers that exist.
2. One step per executed statement that does something: assignments, declarations with a value,
   printing, method calls and returns, and every condition check of an if/while/for (use the
   line of the condition). Skip lines that do nothing (braces, comments, blank lines,
   declarations without a value).
3. Never merge steps. Every loop iteration is traced in first-class detail, including the final failing
   condition check that ends the loop.
4. "changes" holds ONLY the variables created or changed by this step, each with its new value
   as a text string written the way Java shows it: "text" for Strings, 'a' for chars,
   true/false, [1, 2, 3] for arrays. For a method call, put the parameters in "changes" on the
   step that enters the method.
5. "removed" lists variables that go out of scope after this step (for example when a method
   returns or a loop block ends). Usually empty.
6. "printed" is the exact text this step prints. println adds a newline at the end (write it as
   \\n inside the JSON string); print does not. Empty string when nothing is printed.
7. If the program reads input (Scanner etc.), assume a small sample input, state it in
   "assumed_input", and set the variable in the step that reads it.
8. **CRITICAL EXPLANATION RULE:** In "explain", explicitly show the mathematical breakdown, formula, or evaluation steps using real live values (e.g., "Add 10 + 15 to get 25" or "Check if 5 < 10, which is true"), max 22 words. Never be vague.
9. Keep the trace at or under {MAX_STEPS} steps. Only if the full trace would be longer: trace the
   first 3 iterations of each long loop and its last iteration, and between them add ONE step on
   the loop's line with "skipped": true and an "explain" such as "Iterations 4 to 9 repeat the
   same way (skipped to keep the video short)". Otherwise never skip anything and use
   "skipped": false.
10. "final_output" must be exactly everything printed by all steps, joined together.
""".replace("{MAX_STEPS}", str(VIDEO_MAX_STEPS))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

LANGUAGES = {".java": "Java"}


# ---------- Storyboard schema (validated by Pydantic) ----------

def _to_text(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _to_dict(v):
    return {} if v is None else v


def _to_list(v):
    return [] if v is None else v


def _to_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


Text = Annotated[str, BeforeValidator(_to_text)]


class Step(BaseModel):
    line: int
    explain: Text = ""
    changes: Annotated[dict[str, Text], BeforeValidator(_to_dict)] = Field(default_factory=dict)
    removed: Annotated[list[Text], BeforeValidator(_to_list)] = Field(default_factory=list)
    printed: Text = ""
    skipped: Annotated[bool, BeforeValidator(_to_bool)] = False


class Storyboard(BaseModel):
    title: Text = "Dry run"
    summary: Text = ""
    assumed_input: Text = ""
    steps: list[Step]
    final_output: Text = ""


class StoryboardError(Exception):
    """The model could not produce a usable storyboard."""


def parse_storyboard(raw: str) -> Storyboard:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the answer")
    return Storyboard.model_validate_json(text[start:end + 1])


def check_storyboard(sb: Storyboard, n_lines: int) -> list[str]:
    problems = []
    if not sb.steps:
        problems.append("steps is empty")
    bad = sorted({s.line for s in sb.steps if not 1 <= s.line <= n_lines})
    if bad:
        problems.append(f"line numbers outside 1-{n_lines}: {bad[:5]}")
    if len(sb.steps) > VIDEO_MAX_STEPS * 2:
        problems.append(f"too many steps ({len(sb.steps)}); the limit is {VIDEO_MAX_STEPS}")
    return problems


# ---------- Reading code ----------

def read_code(path: Path) -> str:
    code = path.read_text(encoding="utf-8", errors="replace")
    if len(code.splitlines()) > MAX_LINES:
        raise ValueError(f"File has more than {MAX_LINES} lines. Please trim it.")
    return code


def numbered(code: str) -> str:
    return "\n".join(f"{i}: {line}" for i, line in enumerate(code.splitlines(), start=1))


# ---------- LLM call + rate-limit handling ----------

class TokenPacer:
    """Keeps tokens-per-minute under budget by waiting before a request if needed."""

    def __init__(self, budget: int, window: int = 60):
        self.budget = budget
        self.window = window
        self.events = deque()
        self.estimate = 4000

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


def make_storyboard(code: str, language: str) -> tuple[Storyboard, int]:
    n_lines = len(code.splitlines())
    user_msg = (f"Language: {language}\n\nSource code ({n_lines} lines):\n\n{numbered(code)}")
    extra = ""
    problems: list[str] = []

    for attempt in range(2):
        kwargs = dict(
            model=MODEL,
            temperature=0.2,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg + extra},
            ],
        )
        if "gpt-oss" in MODEL:
            kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}

        try:
            response = client.chat.completions.create(**kwargs)
        except BadRequestError:
            kwargs.pop("response_format")
            response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        text = choice.message.content or ""
        used = response.usage.total_tokens if response.usage else pacer.estimate

        problems = []
        sb = None
        if choice.finish_reason == "length":
            problems.append("the JSON was cut off because it was too long")
        else:
            try:
                sb = parse_storyboard(text)
            except ValueError as e:
                problems.append(f"invalid JSON ({str(e)[:160]})")
        if sb is not None:
            problems += check_storyboard(sb, n_lines)

        if not problems:
            return sb, used

        if attempt == 0:
            print(f"   Storyboard problem: {'; '.join(problems)}. Asking again...")
            pacer.record(used)
            pacer.wait()
            extra = ("\n\nYour previous answer was rejected: " + "; ".join(problems) +
                     f". Return the complete JSON again with at most {max(10, VIDEO_MAX_STEPS // 2)} "
                     "steps and explanations under 15 words with math breakdowns.")

    raise StoryboardError("; ".join(problems))


def is_daily_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return "per day" in msg or "(tpd)" in msg or "(rpd)" in msg


def retry_after_seconds(err: Exception, default: float = 60) -> float:
    try:
        value = err.response.headers.get("retry-after")
        return max(float(value), 5) if value else default
    except Exception:
        return default


# ---------- Drawing helpers ----------

MONO_NAMES = ["consola.ttf", "Consolas.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "cour.ttf"]
MONO_B_NAMES = ["consolab.ttf", "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf", "courbd.ttf"]
SANS_NAMES = ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
SANS_B_NAMES = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]

_font_warning_shown = False


def load_font(names: list[str], size: int):
    global _font_warning_shown
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    if not _font_warning_shown:
        print("   Warning: no TrueType font found, text will look small. Install Consolas or DejaVu fonts.")
        _font_warning_shown = True
    return ImageFont.load_default()


class Fonts:
    def __init__(self):
        self.mono = load_font(MONO_NAMES, 18)
        self.mono_b = load_font(MONO_B_NAMES, 18)
        self.mono_l = load_font(MONO_NAMES, 24)
        self.sans = load_font(SANS_NAMES, 20)
        self.sans_b = load_font(SANS_B_NAMES, 20)
        self.small_b = load_font(SANS_B_NAMES, 15)
        self.cap = load_font(SANS_NAMES, 20)
        self.sub = load_font(SANS_NAMES, 28)
        self.title = load_font(SANS_B_NAMES, 50)


FONTS = None


def fit(text: str, font, max_w: float) -> str:
    if font.getlength(text) <= max_w:
        return text
    while text and font.getlength(text + "...") > max_w:
        text = text[:-1]
    return text + "..."


def wrap(text: str, font, max_w: float, max_lines: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if font.getlength(trial) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1] + " ..."
    return [fit(line, font, max_w) for line in lines]


def panel(d, x, y, w, h, title, color, tag=""):
    d.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=PANEL, outline=color, width=2)
    d.rounded_rectangle([x, y, x + w, y + 32], radius=12, fill=color)
    d.rectangle([x + 1, y + 16, x + w - 1, y + 32], fill=color)
    d.text((x + 14, y + 4), title, font=FONTS.sans_b, fill=DARK)
    if tag:
        tw = FONTS.small_b.getlength(tag)
        d.text((x + w - 14 - tw, y + 7), tag, font=FONTS.small_b, fill=DARK)


def draw_code(d, src_lines, cur):
    x, y, w, h = 24, 20, 700, 532
    panel(d, x, y, w, h, "CODE", BLUE)
    n = len(src_lines)
    start = 0 if n <= CODE_ROWS else max(0, min(cur - 1 - CODE_ROWS // 2, n - CODE_ROWS))
    top = y + 44
    for k in range(CODE_ROWS):
        idx = start + k
        if idx >= n:
            break
        ly = top + k * 27
        is_cur = (idx + 1 == cur)
        if is_cur:
            d.rectangle([x + 6, ly - 2, x + w - 6, ly + 24], fill=LINE_BG)
            d.rectangle([x + 6, ly - 2, x + 10, ly + 24], fill=YELLOW)
            d.polygon([(x + 18, ly + 3), (x + 18, ly + 19), (x + 30, ly + 11)], fill=YELLOW)
        num = str(idx + 1)
        d.text((x + 66 - FONTS.mono.getlength(num), ly), num, font=FONTS.mono, fill=DIM)
        font = FONTS.mono_b if is_cur else FONTS.mono
        d.text((x + 80, ly), fit(src_lines[idx], font, w - 96), font=font,
               fill=WHITE if is_cur else TEXT)


def draw_vars(d, state, changed, old):
    x, y, w, h = 744, 20, 512, 310
    panel(d, x, y, w, h, "VARIABLES", YELLOW)
    names = list(state.keys())
    hidden = 0
    if len(names) > MAX_VARS:
        chosen = [n for n in names if n in changed]
        for n in reversed(names):
            if len(chosen) >= MAX_VARS:
                break
            if n not in chosen:
                chosen.append(n)
        keep = set(chosen[:MAX_VARS])
        hidden = len(names) - len(keep)
        names = [n for n in names if n in keep]
    if not names:
        d.text((x + 16, y + 48), "(no variables yet)", font=FONTS.mono, fill=DIM)
        return
    vx, vw = x + 170, w - 170 - 16
    for k, name in enumerate(names):
        ry = y + 44 + k * 28
        is_changed = name in changed
        if is_changed:
            d.rectangle([x + 6, ry - 3, x + w - 6, ry + 24], fill=CHANGED_BG)
        d.text((x + 16, ry), fit(name, FONTS.mono_b, 140), font=FONTS.mono_b, fill=YELLOW)
        value = state[name]
        if is_changed and name in old:
            o = fit(old[name], FONTS.mono, vw * 0.42)
            d.text((vx, ry), o, font=FONTS.mono, fill=DIM)
            ax = vx + FONTS.mono.getlength(o)
            d.text((ax, ry), " -> ", font=FONTS.mono, fill=DIM)
            ax += FONTS.mono.getlength(" -> ")
            d.text((ax, ry), fit(value, FONTS.mono_b, vx + vw - ax), font=FONTS.mono_b, fill=ORANGE)
        else:
            font = FONTS.mono_b if is_changed else FONTS.mono
            d.text((vx, ry), fit(value, font, vw), font=font, fill=ORANGE if is_changed else TEXT)
    if hidden:
        d.text((x + 16, y + h - 26), f"+ {hidden} more variable(s)", font=FONTS.small_b, fill=DIM)


def draw_caption(d, idx, total, text, skipped):
    x, y, w, h = 744, 342, 512, 210
    panel(d, x, y, w, h, "EXPLANATION & MATH", BLUE, tag=f"STEP {idx} / {total}")
    if skipped:
        d.text((x + 16, y + 42), "SKIPPED TO KEEP VIDEO SHORT", font=FONTS.small_b, fill=ORANGE)
        offset = 24
    else:
        offset = 0
    for k, line in enumerate(wrap(text, FONTS.cap, w - 32, 5)):
        d.text((x + 16, y + 46 + offset + k * 28), line, font=FONTS.cap, fill=WHITE)


def draw_output(d, console, printed_now, assumed_input):
    x, y, w, h = 24, 564, 1232, 128
    tag = "+ printed" if printed_now else ("Input: " + assumed_input if assumed_input else "")
    panel(d, x, y, w, h, "OUTPUT & INPUT CONSOLE", GREEN, tag=tag)
    if not console:
        display_text = f"(no output yet | Assumed Input: {assumed_input})" if assumed_input else "(nothing printed yet)"
        d.text((x + 16, y + 44), fit(display_text, FONTS.mono, w - 32), font=FONTS.mono, fill=DIM)
        return
    for k, line in enumerate(console.split("\n")[-OUT_ROWS:]):
        d.text((x + 16, y + 44 + k * 26), fit(line, FONTS.mono, w - 32), font=FONTS.mono, fill=GREEN_TEXT)


def step_frame(src_lines, cur, state, changed, old, console, printed_now, caption, idx, total, skipped, assumed_input):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    draw_code(d, src_lines, cur)
    draw_vars(d, state, changed, old)
    draw_caption(d, idx, total, caption, skipped)
    draw_output(d, console, printed_now, assumed_input)
    # Progress bar at the very bottom
    d.rectangle([24, 704, 24 + 1232, 712], fill=BAR_BG)
    d.rectangle([24, 704, 24 + int(1232 * idx / max(total, 1)), 712], fill=BLUE)
    return img


def title_frame(sb: Storyboard, total_steps: int):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    third = W // 3
    d.rectangle([0, 0, third, 10], fill=BLUE)
    d.rectangle([third, 0, 2 * third, 10], fill=YELLOW)
    d.rectangle([2 * third, 0, W, 10], fill=GREEN)

    d.text((80, 80), "DRY RUN", font=FONTS.sans_b, fill=DIM)
    y = 125
    for line in wrap(sb.title, FONTS.title, W - 160, 2):
        d.text((80, y), line, font=FONTS.title, fill=WHITE)
        y += 64
    y += 12
    for line in wrap(sb.summary, FONTS.sub, W - 160, 3):
        d.text((80, y), line, font=FONTS.sub, fill=TEXT)
        y += 40
    if sb.assumed_input.strip():
        y += 18
        d.rounded_rectangle([80, y, W - 80, y + 54], radius=10, outline=YELLOW, width=2)
        d.text((100, y + 12), fit("Input: " + sb.assumed_input, FONTS.mono, W - 220),
               font=FONTS.mono, fill=YELLOW)

    ly = H - 170
    legend = [(BLUE, "Code & Math: highlighted line and explicit calculation breakdown"),
              (YELLOW, "Variables: orange means the value just changed (old -> new)"),
              (GREEN, "Output & Input: terminal console and inputs fed to program")]
    for color, label in legend:
        d.rounded_rectangle([80, ly + 4, 106, ly + 30], radius=5, fill=color)
        d.text((122, ly), label, font=FONTS.sans, fill=TEXT)
        ly += 42
    d.text((W - 80 - FONTS.small_b.getlength(f"{total_steps} steps"), H - 60),
           f"{total_steps} steps", font=FONTS.small_b, fill=DIM)
    return img


def final_frame(final_text: str, assumed_input: str):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x, y, w, h = 80, 50, W - 160, 530
    panel(d, x, y, w, h, "FINAL OUTPUT & INPUT SUMMARY", GREEN)
    
    current_y = y + 46
    if assumed_input.strip():
        d.text((x + 24, current_y), fit("Assumed Input: " + assumed_input, FONTS.mono_b, w - 48), font=FONTS.mono_b, fill=YELLOW)
        current_y += 36
        d.line([(x + 24, current_y), (x + w - 24, current_y)], fill=LINE_BG, width=2)
        current_y += 18

    lines = final_text.rstrip("\n").split("\n") if final_text.strip() else ["(the program prints nothing)"]
    if len(lines) > 9:
        lines = lines[:8] + ["..."]
    for k, line in enumerate(lines):
        d.text((x + 24, current_y + k * 36), fit(line, FONTS.mono_l, w - 48), font=FONTS.mono_l, fill=GREEN_TEXT)
        
    d.text((80, H - 70), "Dry run complete", font=FONTS.sans_b, fill=DIM)
    return img


# ---------- Video ----------

def post_seconds(explain: str) -> float:
    return min(MAX_STEP_SECONDS, max(MIN_STEP_SECONDS, 1.0 + 0.045 * len(explain)))


def render_video(code: str, sb: Storyboard, out_path: Path) -> float:
    global FONTS
    if FONTS is None:
        FONTS = Fonts()

    src_lines = code.expandtabs(4).splitlines() or [""]
    steps = sb.steps
    total = len(steps)

    printed_all = "".join(s.printed.replace("\\n", "\n") for s in steps)
    final_text = printed_all if printed_all.strip() else sb.final_output

    raw_total = sum(post_seconds(s.explain) + PRE_SECONDS for s in steps)
    budget = max(5.0, VIDEO_TARGET_SECONDS - TITLE_SECONDS - FINAL_SECONDS)
    scale = min(1.0, max(MIN_SCALE, budget / raw_total))
    pre_s = max(0.25, PRE_SECONDS * scale)

    out_path.parent.mkdir(exist_ok=True)
    seconds = TITLE_SECONDS + FINAL_SECONDS

    with imageio.get_writer(str(out_path), fps=FPS, codec="libx264", quality=7,
                            pixelformat="yuv420p", macro_block_size=16) as writer:

        def add(img, secs):
            arr = np.asarray(img)
            for _ in range(max(1, round(secs * FPS))):
                writer.append_data(arr)

        add(title_frame(sb, total), TITLE_SECONDS)

        state, console = {}, ""
        for i, st in enumerate(steps, start=1):
            before = dict(state)
            printed = st.printed.replace("\\n", "\n")
            for name, value in st.changes.items():
                state[name] = value
            removed_any = any(name in before for name in st.removed)
            for name in st.removed:
                state.pop(name, None)
            changed = {k for k, v in st.changes.items() if before.get(k) != v}
            console_after = console + printed
            caption = st.explain or "Run this line."
            post_s = max(MIN_POST_SECONDS, post_seconds(st.explain) * scale)

            first = step_frame(src_lines, st.line, before, set(), {}, console, False,
                               caption, i, total, st.skipped, sb.assumed_input)
            if not changed and not printed and not removed_any:
                add(first, pre_s + post_s)
            else:
                add(first, pre_s)
                add(step_frame(src_lines, st.line, state, changed, before, console_after,
                               bool(printed), caption, i, total, st.skipped, sb.assumed_input), post_s)
            seconds += pre_s + post_s
            console = console_after

        add(final_frame(final_text, sb.assumed_input), FINAL_SECONDS)

    return seconds


# ---------- Processing ----------

def collect_files(target: Path, recursive: bool) -> list[Path]:
    if target.is_dir():
        pattern = "**/*.java" if recursive else "*.java"
        return sorted(target.glob(pattern))
    if target.is_file():
        return [target]
    raise FileNotFoundError(f"Path not found: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn Java files into short dry-run videos.")
    parser.add_argument("path", nargs="?", help="A Java file OR a folder containing Java files")
    parser.add_argument("-r", "--recursive", action="store_true", help="Also search sub-folders")
    args = parser.parse_args()

    target = Path((args.path or input("Enter path to a file or folder: "))
                  .strip().strip('"'.strip("'"))).expanduser()

    files = collect_files(target, args.recursive)
    if not files:
        print("No .java files found.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Model: {MODEL}")
    print(f"Found {len(files)} file(s). Videos go to: {OUTPUT_DIR}\n")

    made, failed, not_attempted = [], [], []
    used_names = set()
    daily_limit_hit = False

    for n, src in enumerate(files, start=1):
        print(f"[{n}/{len(files)}] {src.name}")

        try:
            code = read_code(src)
        except Exception as e:
            failed.append((src.name, str(e)))
            print(f"   SKIPPED: {e}")
            continue

        sb = None
        for attempt in range(1, MAX_RETRIES + 1):
            pacer.wait()
            try:
                sb, tokens = make_storyboard(code, "Java")
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
            except StoryboardError as e:
                failed.append((src.name, f"no valid storyboard: {e}"))
                print(f"   FAILED: no valid storyboard ({e})")
                break
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
            not_attempted = files[n - 1:]
            break

        if sb is not None:
            name = src.stem
            if name in used_names:
                name = f"{src.stem}_{src.parent.name}"
            used_names.add(name)
            video_path = OUTPUT_DIR / f"{name}.mp4"
            try:
                seconds = render_video(code, sb, video_path)
                made.append((video_path, seconds, len(sb.steps)))
                print(f"   video saved: {video_path.name} ({seconds:.0f}s, {len(sb.steps)} steps)")
            except Exception as e:
                failed.append((src.name, f"video rendering failed: {e}"))
                print(f"   FAILED to render video: {e}")

    print(f"\nFinished: {len(made)} video(s), {len(failed)} failed, {len(not_attempted)} not attempted.")
    for path, seconds, n_steps in made:
        print(f"   + {path}  ({seconds:.0f}s, {n_steps} steps)")
    for name, err in failed:
        print(f"   - {name}: {err}")
    if daily_limit_hit:
        print("\nThe free daily limit was reached. Run these files again after it resets:")
        for f in not_attempted:
            print(f"   - {f}")