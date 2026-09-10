"""Hold Windows awake while the BRITTAIN-4 server is up.

`serve.py --keep-awake` cannot help here. BRITTAIN-4 is served by vLLM inside
WSL, and the sleep reprieve is a Win32 call (SetThreadExecutionState) that has
to be made by a Windows process. A WSL process holding the flag holds nothing:
the Linux VM stays awake and the host still suspends underneath it, taking the
GPU and the tunnel with it.

So this runs on the Windows side and follows the server rather than owning it.
It polls the health endpoint -- WSL forwards listening ports to Windows
localhost -- and holds the reprieve for exactly as long as the server answers.
Start it any time, before or after the server; stop it with Ctrl-C.

    python scripts/inference/keep_awake_while_serving.py --wait

The display is left alone, as in brittain.keep_awake: only sleep and hibernate
are held off, so the screen still blanks and locks normally.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from brittain.keep_awake import _acquire, _release  # noqa: E402

DEFAULT_URL = "http://localhost:11435/health"


def _healthy(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"health endpoint to follow (default: {DEFAULT_URL})")
    ap.add_argument("--interval", type=float, default=15.0,
                    help="seconds between checks (default: 15)")
    ap.add_argument("--wait", action="store_true",
                    help="wait for the server to appear instead of exiting")
    # A restart takes ~85s from kill to healthy. Dropping the reprieve the
    # instant a check fails would let the machine sleep in that window, which
    # is exactly when nobody is at the keyboard to stop it.
    ap.add_argument("--grace", type=float, default=180.0,
                    help="seconds the server may be unreachable before "
                         "releasing the reprieve (default: 180)")
    args = ap.parse_args()

    if sys.platform != "win32":
        print(f"[keep-awake is a Windows facility; this is {sys.platform}. "
              f"Run it from Windows, not from inside WSL.]", file=sys.stderr)
        return 2

    if not _healthy(args.url, timeout=5.0):
        if not args.wait:
            print(f"[no server at {args.url}; start it first, or pass --wait]",
                  file=sys.stderr)
            return 1
        print(f"[waiting for {args.url}]", flush=True)
        while not _healthy(args.url, timeout=5.0):
            time.sleep(args.interval)

    holder = _acquire()
    if holder is None:
        print("[keep-awake unavailable; check the system sleep timeout]",
              file=sys.stderr)
        return 2

    print(f"[holding the system awake while {args.url} answers; Ctrl-C to stop]",
          flush=True)
    last_seen = time.monotonic()
    degraded = False
    try:
        while True:
            time.sleep(args.interval)
            if _healthy(args.url, timeout=5.0):
                if degraded:
                    print("[server back; still holding]", flush=True)
                    degraded = False
                last_seen = time.monotonic()
                continue
            if not degraded:
                # Announced once, not once per poll: a restart should read as
                # one event in the log, not a wall of identical lines.
                print(f"[server unreachable; holding for up to "
                      f"{args.grace:.0f}s in case it is restarting]", flush=True)
                degraded = True
            if time.monotonic() - last_seen > args.grace:
                print("[server gone; releasing the machine to sleep]", flush=True)
                return 0
    except KeyboardInterrupt:
        print("\n[released]", flush=True)
        return 0
    finally:
        _release(holder)


if __name__ == "__main__":
    raise SystemExit(main())
