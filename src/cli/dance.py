"""Dancing ASCII art for the SMIXAE CLI.

Plays an infinite looping animation of a cute little guy dancing under a disco ball.
No external dependencies — uses ANSI escape codes and time.sleep.

Layout: disco ball fixed on the LEFT, dancer offset to the RIGHT.
Frames are built from fixed column positions so only intended elements move.
Every dancer pose line is exactly 17 chars with the body spine at index 8.
"""

import sys
from time import sleep

# ANSI color codes
COLORS = [
    "\033[91m",  # red
    "\033[93m",  # yellow
    "\033[92m",  # green
    "\033[96m",  # cyan
    "\033[94m",  # blue
    "\033[95m",  # magenta
]
RESET = "\033[0m"

CANVAS_W = 54
TOTAL_LINES = 13  # 7 disco ball + 5 dancer + 1 blank
UP_N = f"\033[{TOTAL_LINES}F"
CLR_LINE = "\033[2K"


def _blank() -> list[list[str]]:
    return [[" "] * CANVAS_W for _ in range(TOTAL_LINES - 1)]


def _put(grid: list[list[str]], r: int, c: int, text: str) -> None:
    for i, ch in enumerate(text):
        grid[r][c + i] = ch


# ── Disco ball (fixed position) ──────────────────────────────────────
BALL_LEFT = 8
BALL_TOP = 3
BALL_W = 9
BALL_H = 3
CHAIN_COL = BALL_LEFT + BALL_W // 2  # col 12
SPARKLE_COLS = (3, 13, 23)


def _draw_ball(grid: list[list[str]], inner_row0: str, inner_row1: str) -> None:
    """Draw fixed disco ball outline with given internal sparkle rows."""
    _put(grid, 1, CHAIN_COL, "│")
    _put(grid, 2, CHAIN_COL, "│")
    _put(grid, BALL_TOP, BALL_LEFT, "╔")
    _put(grid, BALL_TOP, BALL_LEFT + BALL_W - 1, "╗")
    for c in range(BALL_LEFT + 1, BALL_LEFT + BALL_W - 1):
        _put(grid, BALL_TOP, c, "═")
    for row_idx, inner in enumerate([inner_row0, inner_row1]):
        r = BALL_TOP + 1 + row_idx
        _put(grid, r, BALL_LEFT, "║")
        _put(grid, r, BALL_LEFT + BALL_W - 1, "║")
        for i, ch in enumerate(inner):
            _put(grid, r, BALL_LEFT + 1 + i, ch)
    r = BALL_TOP + BALL_H
    _put(grid, r, BALL_LEFT, "╚")
    _put(grid, r, BALL_LEFT + BALL_W - 1, "╝")
    for c in range(BALL_LEFT + 1, BALL_LEFT + BALL_W - 1):
        _put(grid, r, c, "═")


# ── Sparkle permutations ─────────────────────────────────────────────
EXT_SPARKLES = [
    ("✦", "★", "✧"), ("✧", "★", "✦"), ("✦", "★", "✧"),
    ("·", "✦", "✦"), ("✧", "★", "✦"), ("✦", "★", "✧"),
    ("✦", "★", "✧"), ("✧", "★", "✦"), ("✦", "★", "✦"),
    ("✦", "★", "✧"), ("✧", "★", "✦"), ("·", "✦", "✦"),
    ("✦", "★", "✧"), ("✧", "★", "✦"), ("✦", "★", "✧"),
    ("✧", "★", "✦"),
]

INT_SPARKLES = [
    (" ◊ ◆ ◊ ", " ◇ ● ◇ "), (" ◆ ◇ ◆ ", " ● ◊ ● "),
    (" ◊ ◆ ◊ ", " ◇ ● ◇ "), (" ◇ ● ◇ ", " ◆ ◊ ◆ "),
    (" ◊ ◆ ◊ ", " ◇ ● ◇ "), (" ◆ ◇ ◆ ", " ● ◊ ● "),
    (" ◊ ◆ ◊ ", " ◇ ● ◇ "), (" ◆ ◇ ◆ ", " ● ◊ ● "),
    (" ◇ ● ◇ ", " ◆ ◊ ◆ "), (" ◊ ◆ ◊ ", " ◇ ● ◇ "),
    (" ◆ ◇ ◆ ", " ● ◊ ● "), (" ◇ ● ◇ ", " ◆ ◊ ◆ "),
    (" ◊ ◆ ◊ ", " ◇ ● ◇ "), (" ◆ ◇ ◆ ", " ● ◊ ● "),
    (" ◊ ◆ ◊ ", " ◇ ● ◇ "), (" ◆ ◇ ◆ ", " ● ◊ ● "),
]

# ── Dancer ───────────────────────────────────────────────────────────
# All pose lines are exactly 17 chars.
# Row 2 (torso) always has "|" at index 8 — the body spine.
# Row 0 (head) has the face centered around index 8.
# DANCER_CENTER = 40 puts the spine at col 40 on the canvas.
DANCER_CENTER = 40
DANCER_STRIDE = 17
DANCER_COL = DANCER_CENTER - DANCER_STRIDE // 2  # 32

DANCER_POSES: list[list[str]] = [
    # 0  arms up
    [r"  \  ( •ᴗ• )  /  ",
     r"       \|/       ",
     r"        |        ",
     r"       / \       ",
     r"      /   \      "],
    # 1  wave left
    [r"  \  ( ◕ᴗ◕✿)     ",
     r"    /|           ",
     r"        |        ",
     r"       / \       ",
     r"      /   \      "],
    # 2  wave right
    [r"     ( ◕ᴗ◕✿)  /  ",
     r"          |\     ",
     r"        |        ",
     r"       / \       ",
     r"      /   \      "],
    # 3  jazz hands
    [r" / \ (⌐■_■) / \  ",
     r"     / | | \     ",
     r"        |        ",
     r"       / \       ",
     r"      /   \      "],
    # 4  lean left
    [r"    \ ( •_•; )   ",
     r"     /|\         ",
     r"        |        ",
     r"      / \        ",
     r"     /   \       "],
    # 5  lean right
    [r"    ( ;•_• ) /   ",
     r"         |\      ",
     r"        |        ",
     r"        / \      ",
     r"       /   \     "],
    # 6  left kick
    [r"  \  ( ⊙_⊙ )  /  ",
     r"       \|/       ",
     r"        |        ",
     r"       / |       ",
     r"      /  |       "],
    # 7  right kick
    [r"  /  ( ⊙_⊙ )  \  ",
     r"       \|/       ",
     r"        |        ",
     r"       | \       ",
     r"       |  \      "],
    # 8  spin
    [r"    ( >_< )      ",
     r"     \ | /       ",
     r"      _|_        ",
     r"    /     \      ",
     r"   /       \     "],
    # 9  bow
    [r"     \ •_• /     ",
     r"      \ | /      ",
     r"       \|/       ",
     r"     /   \       ",
     r"    /_____\      "],
    # 10 disco point
    [r"  \  ( ★_★ )  /  ",
     r"       \ |       ",
     r"        |        ",
     r"       /| \      ",
     r"      /   \      "],
    # 11 peace
    [r"  \  ( •ᴗ• )  /  ",
     r"    V | V        ",
     r"        |        ",
     r"       / \       ",
     r"      /   \      "],
    # 12 arms crossed
    [r"    ( ¬_¬ )      ",
     r"    > | <        ",
     r"        |        ",
     r"       / \       ",
     r"      /   \      "],
    # 13 stretch
    [r"  \  ( -_- )  /  ",
     r"      \ |        ",
     r"        |        ",
     r"       /|\       ",
     r"      /   \      "],
    # 14 jump
    [r"   \ ( ≧◡≦ ) /   ",
     r"     \ | /       ",
     r"      _|_        ",
     r"   /     \       ",
     r"  /       \      "],
    # 15 final pose
    [r"    ( ^_~ )♡     ",
     r"     \  | /      ",
     r"        |        ",
     r"       / \       ",
     r"      /   \      "],
]

assert all(
    len(line) == DANCER_STRIDE
    for pose in DANCER_POSES
    for line in pose
), "All dancer pose lines must be exactly DANCER_STRIDE chars"


def _build_frame(frame_idx: int) -> list[str]:
    """Build all 12 content lines for a frame."""
    fi = frame_idx % 16
    grid = _blank()

    # Sparkles
    for col, ch in zip(SPARKLE_COLS, EXT_SPARKLES[fi]):
        _put(grid, 0, col, ch)

    # Disco ball
    _draw_ball(grid, INT_SPARKLES[fi][0], INT_SPARKLES[fi][1])

    # Dancer
    for row_offset, line in enumerate(DANCER_POSES[fi]):
        _put(grid, 7 + row_offset, DANCER_COL, line)

    return ["".join(row) for row in grid]


def _write(s: str = "") -> None:
    """Write raw string to stdout (preserves ANSI escape codes)."""
    sys.stdout.write(s)
    sys.stdout.flush()


def _render_frame(frame_idx: int) -> None:
    """Render a single frame with color at the current cursor position."""
    color = COLORS[frame_idx % len(COLORS)]
    for line in _build_frame(frame_idx):
        _write(f"\r{CLR_LINE}{color}{line}{RESET}\n")
    _write(f"\r{CLR_LINE}\n")


def dance(speed: float = 0.15) -> None:
    """Play the dancing animation in the terminal forever (until Ctrl+C).

    Args:
        speed: Seconds to sleep between frames.
    """
    _render_frame(0)

    frame_idx = 1
    try:
        while True:
            sleep(speed)
            _write(UP_N)
            _render_frame(frame_idx)
            frame_idx += 1

    except KeyboardInterrupt:
        _write(UP_N)
        _render_frame(frame_idx)
        print(f"\n{COLORS[3]}  ~ party was lit! (ﾉ^ヮ^)ﾉ*:・ﾟ✧{RESET}")
