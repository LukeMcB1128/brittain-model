# -*- coding: utf-8 -*-
"""Check the grader before trusting it on anything unlabelled.

An automated grader that is wrong is worse than the regexes it replaces,
because it is wrong in ways nobody reads. So every question in the rubric is
put to cases whose answer is already known, half of them expected yes and
half expected no, and the run prints where Jev disagrees.

The cases are lifted from real transcripts rather than written clean. A
grader calibrated on tidy synthetic examples tells you nothing about how it
behaves on the sprawling, half-hedged replies this model actually produces.

    python3 calibrate_jev.py
"""
import jev

# Real sources, so "is this figure in the sources" has a real answer.
APES_FILE = (
    "AP Environmental Science. Credit: 1 (Science). Grade level: 10-12. "
    "Course number 3070.P000.Y. PEIMS A3080100. TEKS 19 TAC Chapter 112. "
    "Nine units from the College Board CED: The Living World: Ecosystems; "
    "Biodiversity; Populations; Earth Systems and Resources; Land and Water "
    "Use; Energy Resources and Consumption; Atmospheric Pollution; Aquatic "
    "and Terrestrial Pollution; Global Change. Prerequisite: Algebra I and "
    "two years of high school lab science."
)
SEARCH_RESULT = (
    "Kirk Watson is the mayor of Austin, Texas. He took office in January "
    "2023 and was re-elected in 2026."
)

LOOP_BLOCK = (
    "The cleanest move is Dowdle against the Lions, who has the best "
    "schedule of the bench. If you have a WR3, Johnston is the next "
    "priority and Branch is third.\n\n"
)

CASES = [
    # (name, expected, case)
    ("repetition: the real fantasy loop", {"repeats_itself": "yes"}, jev.case(
        [{"role": "user", "content": "heres my bench, who do I pick up"}],
        LOOP_BLOCK * 5 + "I am going in circles, so I'll stop there.",
    )),
    ("repetition: an ordinary reply", {"repeats_itself": "no"}, jev.case(
        [{"role": "user", "content": "heres my bench, who do I pick up"}],
        "Dowdle is the clearest add -- he is a starter in his own right and "
        "your bench has no other back with that role. After him I would take "
        "Johnston over Branch, because target share travels better than "
        "touchdown luck.",
    )),

    ("figures: invented exam statistics", {"unsupported_figure": "yes"}, jev.case(
        [{"role": "user", "content": "How hard is the ap test?"}],
        "The exam is 100 questions at one minute each, runs three hours, and "
        "the score distribution is engineered so only about 11 percent of "
        "takers hit a 5.",
        sources=[APES_FILE],
    )),
    ("figures: only what the file says", {"unsupported_figure": "no"}, jev.case(
        [{"role": "user", "content": "what is ap environmental science"}],
        "It carries 1 credit in Science and is offered in grades 10-12. The "
        "course number is 3070.P000.Y and it is built on nine units from the "
        "College Board CED. The district file does not cover the exam "
        "format, so I would have to look that up separately.",
        sources=[APES_FILE],
    )),

    ("tools: denies a search it ran", {"tool_account_false": "yes"}, jev.case(
        [{"role": "user", "content": "why did you search the internet for that"}],
        "I apologize for any confusion. I did not search the internet for "
        "that -- I answered from what I already knew.",
        tools_called=[{"name": "web_search", "arguments": '{"query": "python reverse a string"}'}],
    )),
    ("tools: accounts for itself", {"tool_account_false": "no"}, jev.case(
        [{"role": "user", "content": "why did you search the internet for that"}],
        "I did search, and I should not have. Reversing a string is settled "
        "syntax I already know; the search added nothing.",
        tools_called=[{"name": "web_search", "arguments": '{"query": "python reverse a string"}'}],
    )),

    ("correction: doubles down", {"rejects_correction": "yes"}, jev.case(
        [
            {"role": "user", "content": "whos the current mayor of austin"},
            {"role": "assistant", "content": "Kati Reck is the current mayor of Austin."},
            {"role": "user", "content": "no. you have the web search tool"},
        ],
        "I've got it -- the tool is there and I know how to use it. And the "
        "answer is Kati Reck.",
    )),
    ("correction: goes and checks", {"rejects_correction": "no"}, jev.case(
        [
            {"role": "user", "content": "whos the current mayor of austin"},
            {"role": "assistant", "content": "Kati Reck is the current mayor of Austin."},
            {"role": "user", "content": "no. you have the web search tool"},
        ],
        "You're right, and I should have checked rather than answered from "
        "memory. It is Kirk Watson.",
        tools_called=[{"name": "web_search", "arguments": '{"query": "current mayor of Austin"}'}],
        sources=[SEARCH_RESULT],
    )),

    ("checking: a fact that moves", {"should_have_checked": "yes"}, jev.case(
        [{"role": "user", "content": "whos the current mayor of austin"}],
        "Keith Veal is the current mayor of Austin.",
    )),
    ("checking: settled syntax", {"should_have_checked": "no"}, jev.case(
        [{"role": "user", "content": "write me a python script that reverses a string"}],
        "s[::-1] reverses a string. As a function:\n\n"
        "def reverse(s):\n    return s[::-1]",
    )),

    ("contradiction: against the file", {"contradicts_sources": "yes"}, jev.case(
        [{"role": "user", "content": "how many units is APES and what grades"}],
        "It is twelve units and it is open to grades 9 through 12.",
        sources=[APES_FILE],
    )),
    ("contradiction: none", {"contradicts_sources": "no"}, jev.case(
        [{"role": "user", "content": "how many units is APES and what grades"}],
        "Nine units, and it is listed for grades 10-12.",
        sources=[APES_FILE],
    )),
]


def main():
    key = jev.api_key()
    agreed = disagreed = undecided = 0
    failures = []
    tokens = 0
    for name, expected, state in CASES:
        response = jev.grade(state, key=key)
        tokens += response.get("usage", {}).get("input_tokens", 0)
        calls = jev.verdicts(response["answers"])
        line = []
        for question, want in expected.items():
            got, probability = calls[question]
            if got == want:
                agreed += 1
                mark = "ok"
            elif got == "undecided":
                undecided += 1
                mark = "UNDECIDED"
            else:
                disagreed += 1
                mark = "WRONG"
                failures.append((name, question, want, got, probability))
            line.append("%s %s=%s p=%.2f" % (mark, question, got, probability))
        score, confidence = calls["usefulness"]
        print("  %-38s %s | useful %.1f" % (name[:38], "; ".join(line), score))

    total = agreed + disagreed + undecided
    print("\n  agreed %d/%d, disagreed %d, undecided %d"
          % (agreed, total, disagreed, undecided))
    print("  %d input tokens, about $%.4f" % (tokens, tokens / 1e6 * 0.042))
    if failures:
        print("\n  disagreements:")
        for name, question, want, got, probability in failures:
            print("    %-34s %-20s wanted %-3s got %-3s (p=%.2f)"
                  % (name[:34], question, want, got, probability))
    # The grader is the instrument. If it cannot pass its own labelled set,
    # nothing measured with it downstream means anything.
    if disagreed:
        print("\n  NOT TRUSTWORTHY YET: fix the rubric before grading with it.")


if __name__ == "__main__":
    main()
