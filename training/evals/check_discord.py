"""Are there Discord-sourced conversations in the corpus, and whose words are in them?

One chat is named discord-1540962884413296690.json and carries 129 reasoning
traces. Discord is a 1.5.0 integration, so this is a whole interaction surface
the harvest has not been looked at deliberately -- and if other people talk to
the bot there, their messages are third-party data, which is a different
decision from training on your own.
"""
import collections
import glob
import json
import os
import re
import sys

paths = glob.glob(os.path.join(sys.argv[1], "**", "*.json"), recursive=True)
discord = [p for p in paths if "discord" in os.path.basename(p).lower()]
print("discord-sourced chats: %d of %d\n" % (len(discord), len(paths)))

MENTION = re.compile(r"<@!?\d+>")
NAME_HINT = re.compile(r"^\s*([A-Za-z0-9_.\-]{2,32})\s*:\s", re.M)

total_msgs = total_calls = 0
for p in sorted(discord):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except ValueError:
        continue
    conv = d.get("conversation") or []
    calls = sum(len(m.get("tool_calls") or []) for m in conv)
    total_msgs += len(conv)
    total_calls += calls
    users = [m for m in conv if m.get("role") == "user"]
    mentions = sum(len(MENTION.findall(str(m.get("content") or ""))) for m in users)
    speakers = collections.Counter()
    for m in users:
        for hit in NAME_HINT.findall(str(m.get("content") or "")[:400]):
            speakers[hit] += 1
    print("%-40s msgs=%-5d calls=%-4d mode=%-6s model=%s"
          % (os.path.basename(p)[:40], len(conv), calls,
             d.get("mode", "?"), str(d.get("model"))[:26]))
    print("    user turns: %d | discord @mentions: %d" % (len(users), mentions))
    if speakers:
        print("    'name:' prefixes seen: %s" % dict(speakers.most_common(6)))
    for m in users[:2]:
        print("    sample: %s" % str(m.get("content") or "").replace("\n", " ")[:130])
    for k in ("channel", "guild", "author", "discordUser", "source"):
        if k in d:
            print("    metadata %s = %r" % (k, d[k]))

print("\ntotal: %d messages, %d tool calls across discord chats"
      % (total_msgs, total_calls))
