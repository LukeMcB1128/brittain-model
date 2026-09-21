# -*- coding: utf-8 -*-
"""Remove the machine owner's identity from the training text.

WHY THIS IS A CORRECTNESS FIX, NOT ONLY A PRIVACY ONE
A trained checkpoint answered "made by Luke McLaren." The maker's surname is
not McLaren -- the model had read it out of the home directory that runs
through every harvested session, `/Users/<owner-username>/...`, and parsed the
username as a full name. 29 training targets wrote that path, against 84 that
named the maker correctly, and the two blended.

Scrubbing also removes memorised absolute paths that are noise for tool
selection: a model that has learned one machine's home directory has learned
nothing transferable, while `/Users/user` generalises.

WHY IT RUNS AT BUILD TIME
The harvested corpus is the record of what actually happened in those sessions
and is not rewritten. This applies on the way into the mix, so the record stays
intact and the substitution stays reproducible and reviewable.

WHY THE RULES LIVE OUTSIDE THIS FILE
The rules are literal personal data -- a username, private email addresses, one
of them a school-issued student account. This repository is public, and a
secret committed once is committed forever. So the values live in a JSON file
alongside the corpus, which is not in version control, and this file carries
only the mechanism.

    data/scrub_rules.json
    {
      "substitutions": [["<literal>", "<replacement>"], ...],
      "detect": {"<label>": "<regex>"}
    }

`substitutions` is applied longest-first regardless of file order, so a rule
cannot be shadowed by a shorter one that overlaps it. `detect` is used by the
audit scripts to find what a rule missed.

The maker's display name is deliberately NOT a rule. "Luke Brittain" is the
answer to who made this, and it belongs in the identity data.
"""
import json
import os

DEFAULT_RULES = os.environ.get(
    "BRITTAIN_SCRUB_RULES", "/home/lukeb/brittain4/data/scrub_rules.json")

_cache = {}


def load_rules(path=None):
    """Read the rules, or fail loudly. A silent empty rule set would let
    personal data through while every check downstream reported success."""
    path = path or DEFAULT_RULES
    if path in _cache:
        return _cache[path]
    if not os.path.exists(path):
        raise SystemExit(
            "no scrub rules at %s. Training text would keep the machine "
            "owner's username, home paths and private addresses. Create the "
            "file (see this module's docstring) or set BRITTAIN_SCRUB_RULES."
            % path)
    with open(path, encoding="utf-8") as fh:
        rules = json.load(fh)
    pairs = [(old, new) for old, new in rules.get("substitutions", [])]
    # Longest first: a short rule must not consume the prefix of a longer one.
    pairs.sort(key=lambda pair: -len(pair[0]))
    if not pairs:
        raise SystemExit("%s contains no substitutions" % path)
    _cache[path] = {"substitutions": pairs, "detect": rules.get("detect", {})}
    return _cache[path]


def scrub(value, rules=None):
    """Apply the substitutions to any string, recursing through containers."""
    pairs = (rules or load_rules())["substitutions"]
    if isinstance(value, str):
        for old, new in pairs:
            if old in value:
                value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [scrub(item, rules) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item, rules) for key, item in value.items()}
    return value


def residue(rows, rules=None):
    """Literals that survived, so a silent miss is not read as a clean pass."""
    pairs = (rules or load_rules())["substitutions"]
    found = {}
    for row in rows:
        text = json.dumps(row, ensure_ascii=False)
        for old, _new in pairs:
            if old in text:
                found[old] = found.get(old, 0) + 1
    return found
