"""Hold Windows awake while a long job runs inside WSL.

keep_awake_while_serving.py follows the server's health endpoint, which is the
wrong signal for a training run -- nothing is listening on 11435 while the GPU
is busy training, so that script would wait forever and hold nothing. This
follows a process inside WSL instead.

The reprieve still has to be taken by a Windows process: SetThreadExecutionState
is a Win32 call, and a WSL process holding it holds nothing at all. The Linux VM
stays up while the host suspends underneath it, taking the GPU with it.

    python scripts/inference/keep_awake_while_training.py
    python scripts/inference/keep_awake_while_training.py --match my_job.py

Ctrl-C releases. The display is left alone, as elsewhere: only sleep and
hibernate are held off, so the screen still blanks and locks.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from brittain.keep_awake import _acquire, _release  # noqa: E402


def running(match: str) -> bool:
    """Is anything in WSL running a command line containing `match`?

    The first character is wrapped in a character class, so the pattern matches
    the job but not the `bash -lc pgrep ...` that carries it. Without that,
    pgrep finds its own caller, every check returns true, and the machine is
    held awake forever -- including when nothing is training at all.
    """
    pattern = "[%s]%s" % (match[0], match[1:]) if match else match
    try:
        done = subprocess.run(
            ["wsl", "bash", "-lc", "pgrep -f '%s' > /dev/null" % pattern],
            capture_output=True, timeout=30)
        return done.returncode == 0
    except (subprocess.SubprocessError, OSError):
        # A transient WSL hiccup should not drop the reprieve mid-run.
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="train_lora.py",
                    help="substring of the WSL command line to follow")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--grace", type=float, default=120.0,
                    help="seconds the process may be absent before releasing")
    args = ap.parse_args()

    if sys.platform != "win32":
        print("[this must run on Windows; a WSL process cannot hold the host awake]",
              file=sys.stderr)
        return 2

    if not running(args.match):
        print("[nothing matching %r is running in WSL]" % args.match, file=sys.stderr)
        return 1

    holder = _acquire()
    if holder is None:
        print("[keep-awake unavailable; check the system sleep timeout]",
              file=sys.stderr)
        return 2

    print("[holding the system awake while %r runs; Ctrl-C to stop]" % args.match,
          flush=True)
    last_seen = time.monotonic()
    try:
        while True:
            time.sleep(args.interval)
            if running(args.match):
                last_seen = time.monotonic()
                continue
            if time.monotonic() - last_seen > args.grace:
                print("[%r finished; releasing the machine to sleep]" % args.match,
                      flush=True)
                return 0
    except KeyboardInterrupt:
        print("\n[released]", flush=True)
        return 0
    finally:
        _release(holder)


if __name__ == "__main__":
    raise SystemExit(main())
