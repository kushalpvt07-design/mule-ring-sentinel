"""
console.py
──────────
Console-output helpers shared by the three CLI entry points
(data.generator, data.extractor, models.train).

WHY THIS EXISTS
───────────────
Every pipeline module printed box-drawing characters (━) and emoji (✅, ⚠️).
On Windows, when stdout is a pipe or a non-UTF-8 console, Python picks the
locale codec — cp1252 on an en-IN/en-US install — and the very first print()
dies:

    UnicodeEncodeError: 'charmap' codec can't encode characters in position
    0-2: character maps to <undefined>

That means README Quick Start steps 2, 3 and 4 were not runnable as written
on a stock Windows shell. This module reconfigures stdout/stderr to UTF-8
where possible, and degrades to ASCII substitutes where it isn't, so the
pipeline runs everywhere without anyone having to set PYTHONIOENCODING or
run `chcp 65001` first.

Import and call enable_utf8_stdout() at the top of main() in each entry point.
"""

from __future__ import annotations

import sys

# Set to True once we know stdout can carry non-ASCII.
_UNICODE_OK = False


def enable_utf8_stdout() -> bool:
    """
    Try to switch stdout/stderr to UTF-8.

    Returns True if non-ASCII output is safe afterwards. Never raises:
    a console we cannot reconfigure is a formatting problem, not a reason
    to fail a training run.
    """
    global _UNICODE_OK

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            # errors="replace" is the safety net: if some glyph still cannot
            # be encoded we print a '?' rather than crashing the pipeline.
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    _UNICODE_OK = "utf" in enc
    return _UNICODE_OK


def unicode_ok() -> bool:
    """True if enable_utf8_stdout() managed to secure a UTF-8 stream."""
    return _UNICODE_OK


def hr(width: int = 62, char: str = "━") -> str:
    """A horizontal rule, falling back to '-' on non-UTF-8 consoles."""
    return (char if _UNICODE_OK else "-") * width


def banner(text: str, width: int = 62) -> str:
    """A titled rule: '━━━ text ━━━' (or '--- text ---')."""
    c = "━" if _UNICODE_OK else "-"
    pad = max(0, width - len(text) - 2)
    left = pad // 2
    return f"{c * left} {text} {c * (pad - left)}"


def sym(name: str) -> str:
    """
    A named symbol with an ASCII fallback.

    Use sym('ok') instead of a literal '✅' so output stays readable on a
    cp1252 console.
    """
    table = {
        "ok": ("✅", "[OK]"),
        "warn": ("⚠️", "[WARN]"),
        "fail": ("❌", "[FAIL]"),
        "stop": ("\U0001f6d1", "[STOP]"),
        "arrow": ("→", "->"),
        "bullet": ("•", "*"),
        "rupee": ("₹", "Rs."),
        "plusminus": ("±", "+/-"),
        "shield": ("\U0001f6e1\ufe0f", "[SENTINEL]"),
    }
    uni, ascii_fallback = table.get(name, ("", ""))
    return uni if _UNICODE_OK else ascii_fallback


def bar(fraction: float, width: int = 50) -> str:
    """A proportional bar for feature-importance printouts."""
    n = max(0, min(width, int(fraction * width)))
    return ("█" if _UNICODE_OK else "#") * n
