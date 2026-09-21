"""Locate the BrittainScript package on PyPI.

The .bs files say "Requires BrittainScript >= 0.3.0" and are run with `bs
<file>`, so there is a console script named bs. An executor means the eval can
score correctness automatically instead of by eye.
"""
import json
import urllib.error
import urllib.request

CANDIDATES = [
    "brittainscript", "brittain-script", "brittain_script", "brittainlang",
    "bscript", "brittain", "bs-lang", "brittainscript-lang",
]

for name in CANDIDATES:
    url = "https://pypi.org/pypi/%s/json" % name
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.load(r)
        info = data["info"]
        print("FOUND  %-22s version=%s" % (name, info.get("version")))
        print("       summary: %s" % (info.get("summary") or "")[:120])
        print("       project: %s" % (info.get("project_url") or info.get("home_page")))
        scripts = list(data.get("releases", {}))[-3:]
        print("       recent releases: %s" % scripts)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print("%-22s HTTP %s" % (name, e.code))
    except Exception as e:  # noqa: BLE001
        print("%-22s %s" % (name, type(e).__name__))
