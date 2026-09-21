"""Where does the stock model actually fail on the driver eval?

A headline rate says how much room there is; this says what the room is made
of. Schema failures are the interesting ones: the model picked a real tool and
then got its arguments wrong, which is exactly what baking the schemas into
weights is supposed to fix.
"""
import collections
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
recs = d["records"]
print("== %s ==" % d["summary"]["label"])
print(json.dumps(d["summary"]["rates"], indent=2))
print("n = %d\n" % d["summary"]["n"])

no_call = [r for r in recs if not r["emitted"]]
bad_name = [r for r in recs if r.get("invalid_name")]
bad_schema = [r for r in recs if r.get("schema_error")]

print("emitted no call at all: %d" % len(no_call))
for r in no_call[:6]:
    print("   ref=%-22s raw: %s" % (r["reference"], (r.get("raw") or "").replace("\n", " ")[:90]))

print("\ninvalid tool names: %d" % len(bad_name))
for r in bad_name[:8]:
    print("   ref=%-22s emitted=%s" % (r["reference"], r["invalid_name"]))

print("\nschema failures: %d" % len(bad_schema))
by_tool = collections.Counter(r["emitted"][0] for r in bad_schema if r["emitted"])
for name, n in by_tool.most_common():
    print("   %-24s %d" % (name, n))
print("\nthe actual validation errors:")
seen = collections.Counter(r["schema_error"] for r in bad_schema)
for err, n in seen.most_common(10):
    print("   [%d] %s" % (n, err))

print("\ndisagreements where both are plausible (valid call, different tool):")
diff = [r for r in recs if r["emitted"] and not r.get("invalid_name")
        and r["emitted"][0] != r["reference"]]
pairs = collections.Counter((r["reference"], r["emitted"][0]) for r in diff)
for (ref, got), n in pairs.most_common(12):
    print("   %-24s -> %-24s %d" % (ref, got, n))
