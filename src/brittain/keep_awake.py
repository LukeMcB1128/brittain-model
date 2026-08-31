"""Hold the machine awake for the lifetime of a long run.

A training run is idle from the operating system's point of view: it takes no
keyboard or mouse input, and neither Windows nor macOS counts GPU load as
activity. A 32-hour run therefore dies at whatever the sleep timeout happens to
be, and the checkpoint tells you nothing about why.

Changing the global power setting works but has to be remembered twice, once
before and once after. This asks for the reprieve only while the run is alive
and drops it automatically on exit, including on an exception.

    with keep_awake("brittain3 pretraining"):
        train(...)

The display is deliberately left alone. Only sleep and hibernate are held off.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys

# SetThreadExecutionState flags. ES_CONTINUOUS makes the request persist for the
# process rather than for one call; ES_SYSTEM_REQUIRED is the sleep reprieve.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake(reason: str = "", *, quiet: bool = False):
    """Keep the system from sleeping while the block runs."""
    holder = _acquire()
    if not quiet:
        if holder is None:
            print(f"[keep-awake unavailable on {sys.platform}; "
                  f"check the system sleep timeout before a long run]", flush=True)
        else:
            print(f"[holding the system awake{f' for {reason}' if reason else ''}]",
                  flush=True)
    try:
        yield holder is not None
    finally:
        _release(holder)


def _acquire():
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        if kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED):
            return "win32"
        return None
    if sys.platform == "darwin":
        # caffeinate exits by itself when this process does, so an abrupt kill
        # cannot strand it holding the machine awake forever.
        try:
            return subprocess.Popen(
                ["caffeinate", "-s", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, ValueError):
            return None
    return None


def _release(holder) -> None:
    if holder is None:
        return
    if holder == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        return
    with contextlib.suppress(Exception):
        holder.terminate()
