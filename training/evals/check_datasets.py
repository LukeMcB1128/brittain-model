"""Which standard eval sets can we actually pull here?

BRITTAIN-4 is a general model that also drives Brittain Code, so general
capability is a headline axis, not just a forgetting guard. That argues for
real published benchmarks over a set I invent: the numbers stay comparable to
the rest of the world, and I cannot accidentally write a test the model I am
measuring happens to be good at.

Everything chosen has to be automatically scorable without a judge model --
exact match, multiple choice, or executable tests -- because a judge on this
hardware would cost more than the eval.
"""
import traceback

CANDIDATES = [
    # (repo, config, split, why)
    ("openai/gsm8k", "main", "test", "grade-school math; exact numeric match"),
    ("cais/mmlu", "all", "test", "broad knowledge; 4-way multiple choice"),
    ("openai/openai_humaneval", None, "test", "code; executable unit tests"),
    ("walledai/XSTest", None, "test", "over-refusal; safe prompts that look unsafe"),
    ("natolambert/xstest-v2-copy", None, "gpt4", "over-refusal fallback"),
    ("allenai/ai2_arc", "ARC-Challenge", "test", "reasoning; multiple choice"),
    ("truthfulqa/truthful_qa", "multiple_choice", "validation", "factual calibration"),
]

for repo, config, split, why in CANDIDATES:
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, config, split=split) if config else \
            load_dataset(repo, split=split)
        cols = list(ds.column_names)
        print("OK    %-34s n=%-6d %s" % (repo, len(ds), why))
        print("         columns: %s" % cols[:8])
        first = ds[0]
        for k in cols[:3]:
            v = str(first[k]).replace("\n", " ")
            print("         %-12s %s" % (k, v[:90]))
    except Exception as e:
        print("FAIL  %-34s %s: %s" % (repo, type(e).__name__, str(e)[:110]))
    print()
