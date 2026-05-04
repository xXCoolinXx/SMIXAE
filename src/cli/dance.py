"""Dancing ASCII art for the SMIXAE CLI.

Plays a short looping animation of a cute little guy doing a little dance.
No external dependencies — uses ANSI escape codes and time.sleep.
"""

import sys
from time import sleep

# 4-frame dance cycle (3 lines per frame)
FRAMES = [
    # Frame 1 — arms up
    [
        "  (•ᴗ•)♡",
        "   /|\\",
        "   / \\",
    ],
    # Frame 2 — arms down
    [
        "  (•ᴗ•)♡",
        "   \\|/",
        "   | |",
    ],
    # Frame 3 — arms left
    [
        "  (•ᴗ•)♡",
        "   |\\",
        "    \\",
    ],
    # Frame 4 — arms right
    [
        "  (•ᴗ•)♡",
        "    /|",
        "   /",
    ],
]

# ANSI escape helpers
UP_4 = "\033[4F"    # move cursor up 4 lines (3 for frame + 1 blank line)
CLR_LINE = "\033[2K"  # clear entire current line


def _write(s: str = "") -> None:
    """Write raw string to stdout (preserves ANSI escape codes)."""
    sys.stdout.write(s)
    sys.stdout.flush()


def dance(loops: int = 5, speed: float = 0.2) -> None:
    """Play the dancing animation in the terminal.

    Args:
        loops: Number of times to cycle through all frames.
        speed: Seconds to sleep between frames.
    """
    try:
        # Print initial frame + blank line so we have room to overwrite
        for line in FRAMES[0]:
            print(line)
        print()

        for _ in range(loops):
            for frame in FRAMES[1:]:          # skip frame 0 (already printed)
                sleep(speed)
                # Move up to start of frame, clear each line, then redraw
                _write(UP_4)
                for line in frame:
                    _write(f"\r{CLR_LINE}{line}\n")
                _write(f"\r{CLR_LINE}\n")  # clear the blank line too

        # Return to frame 0 for final pose
        sleep(speed)
        _write(UP_4)
        for line in FRAMES[0]:
            _write(f"\r{CLR_LINE}{line}\n")
        _write(f"\r{CLR_LINE}\n")
        print("  ~ nice moves ~")

    except KeyboardInterrupt:
        # Graceful exit — move cursor down past the animation area
        print("\n  dance interrupted!  (｡•́︿•̀｡)")
