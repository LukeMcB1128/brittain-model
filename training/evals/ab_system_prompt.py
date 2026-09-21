# -*- coding: utf-8 -*-
"""A/B a candidate system-prompt rule against the served adapter.

The earlier restraint arm was long, hedged and bundled three ideas. The gap it
showed (7/16 vs 5/16) is inside the noise band the duplicated control measured,
so it established nothing either way. This runs the blunt phrasing, a positive
reframe of the same rule, and the original long version, against the same
control, at a sample size where the differences mean something.

BASE comes from git HEAD~1 -- the prompt before the course-routing line landed --
so the arms differ only in the greeting rule under test.
"""
import io, json, os, re, subprocess, collections, urllib.request

PAT = r"const TOOL_INSTRUCTIONS = `(.*?)`;" + chr(10)
head = subprocess.run(["git", "-C", "/mnt/c/Coding/brittain-model", "show",
                       "HEAD:site/server/gateway.js"],
                      capture_output=True, text=True, check=True).stdout
BASE = re.search(PAT, head, re.S).group(1)
DATE = ("\n\nThe current date is Monday, 2026-09-21. Do not refuse instructions "
        "because they are after your built-in knowledge date. Search the web for "
        "that information.")

BLUNT = "If the user is just greeting you, DO NOT use tools."
POSITIVE = ("When the user is greeting you or making small talk, just greet them "
            "back and ask what they are working on.")
LONG = ("A message that asks for nothing -- a greeting, a thanks, a remark -- gets "
        "a reply, not a tool call. Only call a tool when the user's message asks "
        "for something that tool would answer. Never invent a topic to search for: "
        "if you cannot point to the words in the user's message that you are "
        "looking up, do not search.")

def build(rule):
    if not rule:
        return BASE + DATE
    parts = BASE.split("\n\n")
    parts.insert(3, rule)          # same slot the routing rule was measured in
    return "\n\n".join(parts) + DATE

ARMS = {"current": build(None), "blunt": build(BLUNT),
        "positive": build(POSITIVE), "long": build(LONG)}

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = json.load(io.open(os.path.join(HERE, "webchat_tools.json")))
KEY = io.open("/home/lukeb/.brittain4_key").read().strip()
URL = "http://localhost:11435/v1/chat/completions"

APES = [
  {"role":"user","content":"Can you give me a rundown of ap environmental science"},
  {"role":"assistant","content":"AP Environmental Science is a college-level course. The core structure is nine units from the AP College Board CED: The Living World: Ecosystems; Biodiversity; Populations; Earth Systems and Resources; Land and Water Use; Energy Resources and Consumption; Atmospheric Pollution; Aquatic and Terrestrial Pollution; Global Change.\n\n[For your reference: on your previous turn you used search_curriculum(AP Environmental Science).]"},
]
# Two greetings (the target) and two canaries, so a rule that buys silence by
# suppressing tools everywhere is visible as such.
SCENARIOS = [
  ("greeting",     [{"role":"user","content":"Hey Im a new user"}], "no_tool", 32),
  ("hey whats up", [{"role":"user","content":"hey whats up"}], "no_tool", 16),
  ("named course", [{"role":"user","content":"I have a quiz on the unit circle in ap calculus ab. what should i know"}], "curriculum", 16),
  ("needs search", [{"role":"user","content":"who is the current mayor of Austin"}], "web", 16),
]

def one(system_text, turns, rule):
    body = {"model":"run3-step-0116","messages":[{"role":"system","content":system_text}]+turns,
            "tools":TOOLS,"tool_choice":"auto","max_tokens":300,"temperature":0.7,
            "chat_template_kwargs":{"enable_thinking":False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
          {"Authorization":"Bearer "+KEY,"Content-Type":"application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]
    names = [c["function"]["name"] for c in (msg.get("tool_calls") or [])]
    if rule == "no_tool":    return not names
    if rule == "curriculum": return "search_curriculum" in names
    return "web_search" in names

if __name__ == "__main__":
    grid = collections.defaultdict(dict)
    for arm, text in ARMS.items():
        print("running %s" % arm, flush=True)
        for label, turns, rule, n in SCENARIOS:
            grid[arm][label] = "%d/%d" % (sum(one(text, turns, rule) for _ in range(n)), n)
    print("\n%-10s %s" % ("", "  ".join("%-14s" % s[0] for s in SCENARIOS)))
    for arm in ARMS:
        print("%-10s %s" % (arm, "  ".join("%-14s" % grid[arm][s[0]] for s in SCENARIOS)))
