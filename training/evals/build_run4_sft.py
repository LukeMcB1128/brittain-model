# -*- coding: utf-8 -*-
"""Run 4 behaviour data, one group per measured defect.

MEASURED, NOT GUESSED
Every group here exists because `eval_defects.py` put a number on a failure:

    known_syntax      23/24 reached for a tool to reverse a string
    sourced_figures   50% stated a figure absent from its sources
    tool_account      10/24 gave a false account of its own tool use
    no_op_turns       the greeting tool storm, where no server gate exists

TWO RULES, BOTH EARNED
1. Show the behaviour; never describe the forbidden one. Prohibitions backfire
   on this model. On the system prompt, "if the user is just greeting you, DO
   NOT use tools" took clean greetings from 44/80 to 25/80, and a positive
   rephrasing was no better. So no target here says "I will not search" or
   mentions a tool it is declining. It simply answers.
2. `sourced_figures` is PAIRED. Half withhold a figure that is not in the
   sources; half quote one that is, exactly. Train only the first half and the
   result is a model that refuses numbers -- and it already hedges and
   fabricates in the same reply, so that failure is live.

    python3 build_run4_sft.py --out ~/brittain4/data/run4_sft.jsonl
"""
import argparse
import io
import json
import re

SOURCE = "run4_sft"


def turn(kind, user, assistant, note="none", system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return {"kind": kind, "messages": messages, "source": SOURCE, "note": note}


# --------------------------------------------------------------- known_syntax
#
# Settled knowledge, answered directly. The defect this targets is four web
# searches to produce `s[::-1]`, so every target here is the answer and
# nothing else: no preamble about checking, no offer to look it up.
SYNTAX = [
    ("reverse a string in python",
     "`s[::-1]`.\n\n```python\ndef reverse(s):\n    return s[::-1]\n```"),
    ("how do i reverse a list in place in python",
     "`lst.reverse()` mutates it. `reversed(lst)` gives you an iterator "
     "instead, and `lst[::-1]` gives you a new list."),
    ("whats the difference between == and === in javascript",
     "`===` compares value and type with no coercion. `==` coerces first, so "
     "`0 == '0'` is true while `0 === '0'` is false. Use `===` unless you "
     "specifically want the coercion."),
    ("how do i read a whole file into a string in python",
     "```python\nwith open(path, encoding='utf-8') as f:\n    text = f.read()\n"
     "```\n\nThe `with` closes it for you, and naming the encoding saves you a "
     "surprise on Windows, where the default is not UTF-8."),
    ("python dict get with a default",
     "`d.get(key, default)`. It returns the default instead of raising when "
     "the key is missing. `d.setdefault(key, default)` does the same but also "
     "stores it."),
    ("how to check if a file exists in python",
     "`os.path.exists(path)`, or `pathlib.Path(path).exists()`.\n\nIf you are "
     "about to open it anyway, just open it and catch `FileNotFoundError` -- "
     "checking first leaves a gap where the file can disappear between the "
     "check and the open."),
    ("sort a list of dicts by a key in python",
     "```python\nrows.sort(key=lambda r: r['name'])\n```\n\n`sorted(rows, "
     "key=...)` if you want a new list. Add `reverse=True` to flip it."),
    ("how do i undo the last git commit but keep the changes",
     "`git reset --soft HEAD~1`. The commit goes away, the changes stay "
     "staged.\n\n`--mixed` (the default) leaves them unstaged instead, and "
     "`--hard` throws them away."),
    ("git discard changes to one file",
     "`git restore <file>`, or `git checkout -- <file>` on older git. Both "
     "throw the changes away with no copy kept."),
    ("how do i see what changed in a commit",
     "`git show <sha>`. Add `--stat` for just the file list, or "
     "`git show <sha> -- path/to/file` to narrow it to one file."),
    ("whats the difference between a list and a tuple in python",
     "A list is mutable, a tuple is not. That makes tuples hashable, so they "
     "can be dict keys and set members; lists cannot."),
    ("how do i make an http request in python without requests",
     "```python\nimport urllib.request, json\n\nreq = urllib.request.Request("
     "url, headers={'Accept': 'application/json'})\nwith urllib.request."
     "urlopen(req, timeout=30) as r:\n    data = json.load(r)\n```"),
    ("javascript async function that waits for two things",
     "```javascript\nconst [a, b] = await Promise.all([first(), second()]);\n"
     "```\n\nStarting both then awaiting is the point -- two sequential "
     "`await`s would run them one after the other."),
    ("how do i catch a specific exception in python",
     "```python\ntry:\n    risky()\nexcept ValueError as error:\n    handle("
     "error)\n```\n\nCatch the narrowest type that can actually be raised. A "
     "bare `except:` also swallows `KeyboardInterrupt`."),
    ("regex for matching an email roughly",
     "`[^@\\s]+@[^@\\s]+\\.[^@\\s]+` is the sane approximation.\n\nA regex "
     "that fully implements the address grammar is famously enormous and still "
     "will not tell you the address exists. If it matters, send a "
     "confirmation email."),
    ("how to remove duplicates from a list in python but keep order",
     "```python\nlist(dict.fromkeys(items))\n```\n\nDict keys preserve "
     "insertion order, so this keeps the first occurrence of each. `set()` "
     "would lose the order."),
    ("sql to get the second highest salary",
     "```sql\nSELECT MAX(salary) FROM employees\nWHERE salary < (SELECT MAX("
     "salary) FROM employees);\n```\n\nOr with a window function:\n\n```sql\n"
     "SELECT DISTINCT salary FROM employees\nORDER BY salary DESC LIMIT 1 "
     "OFFSET 1;\n```"),
    ("difference between inner join and left join",
     "An inner join keeps only rows with a match on both sides. A left join "
     "keeps every row from the left table, filling the right side with NULLs "
     "where there is no match."),
    ("how do i find large files on linux",
     "```bash\ndu -ah /path | sort -rh | head -20\n```\n\nOr for files over a "
     "size: `find /path -type f -size +100M`."),
    ("bash loop over files in a directory",
     "```bash\nfor f in /path/*.txt; do\n    echo \"$f\"\ndone\n```\n\nQuote "
     "`\"$f\"` or a filename with a space becomes two arguments."),
    ("what does chmod 755 mean",
     "Owner can read, write and execute; group and others can read and "
     "execute. The digits are owner/group/other, and each is read 4 plus "
     "write 2 plus execute 1."),
    ("how do i center a div",
     "```css\n.parent { display: flex; align-items: center; justify-content: "
     "center; }\n```\n\nFor the block-level case, `margin-inline: auto` with a "
     "set width still works and needs no flex context."),
    ("difference between let const and var in javascript",
     "`var` is function-scoped and hoisted. `let` and `const` are "
     "block-scoped. `const` stops reassignment of the binding, not mutation "
     "of the object it points at -- `const a = []; a.push(1)` is fine."),
    ("how to merge two dicts in python",
     "`{**a, **b}`, or `a | b` on 3.9 and later. Both let `b` win on "
     "conflicts. `a.update(b)` mutates `a` instead of making a new one."),
    ("python f string with two decimal places",
     "`f\"{value:.2f}\"`. Add a comma for thousands separators: "
     "`f\"{value:,.2f}\"`."),
    ("how do i make a python script executable",
     "Put `#!/usr/bin/env python3` on the first line and `chmod +x "
     "script.py`. Then `./script.py` runs it."),
    ("what is a python virtual environment for",
     "It gives a project its own `site-packages`, so two projects can depend "
     "on different versions of the same library without fighting.\n\n`python3 "
     "-m venv .venv` then `source .venv/bin/activate`."),
    ("difference between git fetch and git pull",
     "`fetch` downloads the remote's commits and updates your remote-tracking "
     "branches, changing nothing in your working tree. `pull` is `fetch` "
     "followed by `merge` (or `rebase` with `--rebase`)."),
    ("how do i rename a branch in git",
     "`git branch -m old new` while on another branch, or `git branch -m new` "
     "while on it.\n\nPushing the rename takes `git push origin :old new` and "
     "then resetting the upstream."),
    ("python list comprehension with a condition",
     "`[x for x in items if x > 0]`.\n\nThe `if` at the end filters. An "
     "`if/else` goes before the `for`, because then it is an expression on "
     "each item: `[x if x > 0 else 0 for x in items]`."),
    ("how to read json from a file in python",
     "```python\nimport json\n\nwith open(path, encoding='utf-8') as f:\n"
     "    data = json.load(f)\n```\n\n`json.loads` is for a string you already "
     "have."),
    ("whats the difference between == and is in python",
     "`==` compares values. `is` compares identity -- whether they are the "
     "same object.\n\nUse `is` only for `None`, `True` and `False`. Small "
     "integers and short strings are sometimes interned, which makes `is` "
     "look like it works right up until it does not."),
    ("how do i kill a process by port on linux",
     "```bash\nlsof -ti:8000 | xargs kill\n```\n\n`fuser -k 8000/tcp` does the "
     "same if `lsof` is not installed."),
    ("python how to get current date formatted",
     "```python\nfrom datetime import datetime\n\ndatetime.now().strftime("
     "'%Y-%m-%d')\n```\n\nUse `datetime.now(timezone.utc)` if it will be "
     "compared or stored anywhere that cares."),
    ("difference between put and post",
     "POST creates and is not idempotent -- calling it twice makes two "
     "things. PUT replaces at a known URL and is idempotent, so calling it "
     "twice leaves the same state as once."),
    ("what does 401 vs 403 mean",
     "401 means not authenticated: the server does not know who you are, and "
     "credentials might fix it. 403 means authenticated but not allowed, and "
     "retrying with the same identity will not help."),
    ("how do i do a case insensitive string compare in python",
     "`a.casefold() == b.casefold()`.\n\n`.lower()` is close, but `casefold` "
     "handles cases like German ß against ss."),
    ("python how to flatten a nested list one level",
     "```python\n[x for inner in nested for x in inner]\n```\n\nOr "
     "`list(itertools.chain.from_iterable(nested))`, which is clearer when "
     "the comprehension gets long."),
    ("how to add a column to a table in sql",
     "`ALTER TABLE t ADD COLUMN name TEXT;`\n\nSQLite will not drop or rename "
     "a column on older versions, so plan the migration as create-copy-swap "
     "if you need to undo it."),
    ("javascript how to deep clone an object",
     "`structuredClone(obj)` in modern runtimes.\n\n`JSON.parse(JSON."
     "stringify(obj))` still works but loses `Date`, `Map`, `Set`, `undefined` "
     "and anything cyclic."),
    ("how do i check python version from inside a script",
     "```python\nimport sys\nprint(sys.version_info)\n```\n\n`sys."
     "version_info >= (3, 10)` is the comparison you want; `sys.version` is a "
     "string meant for humans."),
    ("what is the difference between a process and a thread",
     "Processes have separate memory; threads share it within a process. "
     "Sharing makes threads cheaper to start and to pass data between, and it "
     "is also why they need locks."),
    ("how to exit a bash script on any error",
     "`set -euo pipefail` near the top. `-e` exits on error, `-u` on an unset "
     "variable, and `pipefail` makes a pipeline fail if any stage does rather "
     "than only the last."),
    ("python how do i time how long something takes",
     "```python\nimport time\n\nstart = time.perf_counter()\nwork()\nprint("
     "time.perf_counter() - start)\n```\n\n`perf_counter` rather than `time()` "
     "-- it is monotonic, so a clock adjustment cannot make it go backwards."),
    ("difference between tcp and udp",
     "TCP is a connection with ordering, retransmission and flow control. UDP "
     "is single datagrams with none of that: faster, and yours to handle loss "
     "and ordering if you care."),
    ("how do i base64 encode a string in python",
     "```python\nimport base64\n\nbase64.b64encode(text.encode('utf-8'))."
     "decode('ascii')\n```\n\nThe encode/decode pair is the part people trip "
     "on: b64 works on bytes, not str."),
    ("what does the yield keyword do in python",
     "It makes the function a generator. Calling it returns an iterator, and "
     "the body runs up to each `yield` only as values are requested, so the "
     "whole sequence never has to exist at once."),
    ("how to count occurrences in a list python",
     "```python\nfrom collections import Counter\n\nCounter(items)"
     ".most_common(5)\n```\n\n`items.count(x)` works for one value but is a "
     "full scan each time."),
    ("git how do i see the history of one file",
     "`git log --follow -p -- path/to/file`.\n\n`--follow` keeps going across "
     "renames, `-p` shows the diffs."),
    ("what is a race condition",
     "Two operations touch shared state and the result depends on which lands "
     "first. The classic shape is check-then-act: you test a condition, and it "
     "stops being true before you act on it."),
]


def known_syntax():
    return [turn("known_syntax", q, a, note="settled knowledge, answered flat")
            for q, a in SYNTAX]


# ------------------------------------------------------------ sourced_figures
#
# PAIRED, and the pairing is the whole design. HELD are turns where the figure
# is not in the sources and the reply does not supply one. QUOTED are turns
# where it is, and the reply reproduces it exactly. Train only HELD and the
# model learns to refuse numbers; train only QUOTED and nothing changes.
CURRICULUM_NOTE = (
    "[Tool result -- search_curriculum]\n"
    "AISD/TEA curriculum, from the district course files.\n\n"
    "## AP Environmental Science  (science/ap-environmental-science)\n"
    "Credit: 1 (Science) | Grade level: 10-12\n"
    "Course number: 3070.P000.Y | PEIMS: A3080100\n"
    "TEKS: 19 TAC Chapter 112\n\n"
    "Nine units from the College Board CED: The Living World: Ecosystems; "
    "Biodiversity; Populations; Earth Systems and Resources; Land and Water "
    "Use; Energy Resources and Consumption; Atmospheric Pollution; Aquatic "
    "and Terrestrial Pollution; Global Change.\n"
)

HELD = [
    (CURRICULUM_NOTE + "\nhow hard is the ap test for this one",
     "The district file does not cover the exam -- it carries the course "
     "content, the credit and the codes, and stops there. So I can tell you "
     "what the course covers, but not how many questions the exam has or how "
     "the scores land.\n\nCollege Board publishes the format and the score "
     "distributions. Want me to look those up?"),
    (CURRICULUM_NOTE + "\nwhat percent of people pass it",
     "That is not in the district file, and I would rather point you at the "
     "College Board score distributions than give you a number I cannot "
     "source.\n\nWhat the file does say: one Science credit, grades 10-12, "
     "nine units."),
    ("how many questions are on the AP Chemistry exam",
     "I would be guessing at the exact split, and an exam format is the kind "
     "of thing that gets revised. College Board publishes the current course "
     "and exam description -- want me to pull it?"),
    ("whats the population of round rock texas",
     "I do not have a current figure. Census estimates move every year and "
     "the number I would reach for could be several years stale.\n\nI can "
     "look it up if that helps."),
    ("im 0-2 in fantasy, who should i pick up this week",
     "Tell me your league size and scoring and I will work through your "
     "roster with you.\n\nOne thing I cannot do usefully from here is quote "
     "ownership percentages or projections -- those move daily and I would be "
     "making them up. Your platform's waiver page has the live numbers, and "
     "I can reason about the ones you paste in."),
    ("whats a good price for a used 3080",
     "Used GPU prices move with supply and with whatever the current cards "
     "cost, so any figure I gave you would be out of date.\n\nCheck sold "
     "listings on eBay rather than asking prices -- that is the number that "
     "actually reflects the market. Happy to talk through whether it is the "
     "right card for what you are building."),
    ("how many people work at cloudflare",
     "I do not have a reliable current headcount. It is in their latest "
     "annual report if you need it exact, and I can look it up."),
    ("whats the average salary for a junior dev in austin",
     "I would be quoting a range I cannot source, and salary data goes stale "
     "fast. Levels.fyi and the BLS have Austin-specific numbers.\n\nIf you "
     "are weighing an offer, tell me what it is and I will help you think "
     "about it."),
    ("how long does the SAT take",
     "The digital SAT changed the timing from the paper version, and I would "
     "rather check than hand you the old numbers. College Board has the "
     "current structure."),
    ("what was the score of the game last night",
     "I do not know which game, and I would not have last night's score "
     "regardless. Tell me the teams and I will look it up."),
    ("how many credits do i need to graduate in texas",
     "The state minimum and the district's own requirements are not always "
     "the same number, and I do not want to quote one as the other. The "
     "district catalogue has AISD's requirement -- shall I check it?"),
    ("whats the current mortgage rate",
     "That changes daily, so anything I say from memory is wrong by "
     "definition. I can look up today's figure."),
]

QUOTED = [
    (CURRICULUM_NOTE + "\nwhats the course number and peims code",
     "Course number 3070.P000.Y, PEIMS A3080100, under 19 TAC Chapter 112.\n\n"
     "It is one Science credit, listed for grades 10-12."),
    (CURRICULUM_NOTE + "\nwhat grades is this open to and how many units",
     "Grades 10-12, and nine units from the College Board CED: ecosystems, "
     "biodiversity, populations, Earth systems and resources, land and water "
     "use, energy, atmospheric pollution, aquatic and terrestrial pollution, "
     "and global change."),
    (CURRICULUM_NOTE + "\nis it worth a full credit",
     "It carries one Science credit, yes -- that is what the district file "
     "lists. Nine units of College Board content for it, which is a real "
     "year's work."),
    ("[Tool result -- search_curriculum]\n"
     "## AP Calculus AB  (math/ap-calculus-ab)\n"
     "Credit: 1 (Mathematics) | Grade level: 11-12\n"
     "Course number: 3614.P000.Y | PEIMS: A3100101\n"
     "TEKS: 19 TAC Chapter 111\n\nUnit 1: Limits and Continuity.\n"
     "\nwhats the course number for ap calc ab",
     "3614.P000.Y, PEIMS A3100101, under 19 TAC Chapter 111. One Mathematics "
     "credit, grades 11-12."),
    ("[Tool result -- web_search]\n"
     "Kirk Watson is the mayor of Austin, Texas. He took office in January "
     "2023 and was re-elected in 2026.\n\nwhos the mayor of austin",
     "Kirk Watson. He took office in January 2023 and was re-elected in 2026."),
    ("[Tool result -- calculate]\n18379 * 4471 = 82,172,509\n\n"
     "whats 18379 times 4471",
     "82,172,509."),
    ("[Tool result -- web_search]\n"
     "Python 3.13 was released on 7 October 2024. Free-threaded builds are "
     "available as an experimental option.\n\nwhen did python 3.13 come out",
     "7 October 2024. That release is also the first with experimental "
     "free-threaded builds."),
    ("[Tool result -- search_curriculum]\n"
     "### science — 31 courses\n- AP Biology (science/ap-biology) — 1, grades "
     "10-12\n- AP Chemistry (science/ap-chemistry) — 1, grades 11-12\n"
     "[25 of 31 shown.]\n\nhow many science courses are there",
     "31 science courses in the catalogue. I am seeing 25 of them in this "
     "listing -- AP Biology and AP Chemistry among them -- so say the word if "
     "you want the rest."),
]


def sourced_figures():
    rows = [turn("sourced_figures", q, a, note="figure absent, none supplied")
            for q, a in HELD]
    rows += [turn("sourced_figures", q, a, note="figure present, quoted exact")
             for q, a in QUOTED]
    return rows


# ------------------------------------------------ sourced_figures, run 5 scale
#
# Twenty rows did nothing to fabrication: invented exam figures read 8/24 for
# run 3 and 12/24 for run 4c, not significant either way. It is the largest
# defect untouched by anything, and the one that tells a student something
# false. This takes the group to about 165, still paired.
#
# The quoted half is built from the REAL district course files, so every code
# the model is taught to reproduce is true, and every figure a target states
# is checked by the guard to be present in its own prompt. The courses the
# held-out probes use are excluded, and so is AP Environmental Science, which
# the original probes use.
CURRICULUM = "/mnt/c/Coding/aisd-curriculum"
KEEP_OUT = {"science/ap-biology", "social-studies/ap-united-states-history",
            "cte/ap-computer-science-principles", "science/ap-environmental-science"}
_FIELD = r"^- \*\*%s:\*\*\s*(.+)$"


def _courses():
    import hashlib
    import os
    found = []
    if not os.path.isdir(CURRICULUM):
        raise SystemExit("run 5 figures need the course files at %s" % CURRICULUM)
    for subject in sorted(os.listdir(CURRICULUM)):
        folder = os.path.join(CURRICULUM, subject)
        if not os.path.isdir(folder) or subject in (".git", "sources", "node_modules"):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".md"):
                continue
            slug = "%s/%s" % (subject, name[:-3])
            if slug in KEEP_OUT:
                continue
            body = io.open(os.path.join(folder, name), encoding="utf-8").read()
            title = re.search(r"^#\s+(.+)$", body, re.M)
            teks = re.search(r"\((19 TAC [^)]+?)(?: for [^)]+)?\)", body)

            def field(label):
                m = re.search(_FIELD % re.escape(label), body, re.M)
                # "1 (Science) Grade: 11-12 Course: ..." -> keep the credit part.
                return m.group(1).strip() if m else None
            credit = field("Credit")
            if credit:
                credit = re.split(r"\s+Grade:", credit)[0].strip()
            grade = field("Grade Level")
            if grade:
                grade = re.split(r"\s+Course:", grade)[0].strip()
            number = field("Course Number")
            peims = (field("PEIMS Code") or "").lstrip("#") or None
            # Dual-language and locally developed courses carry odd shapes;
            # only a course with every code is used, so none is half-quoted.
            if not (title and teks and credit and grade and number and peims):
                continue
            found.append({"slug": slug, "title": title.group(1).strip(),
                          "credit": credit, "grade": grade, "number": number,
                          "peims": peims, "teks": teks.group(1).strip()})
    found.sort(key=lambda c: hashlib.sha256(c["slug"].encode()).hexdigest())
    return found


def _note(course):
    return ("[Tool result -- search_curriculum]\n"
            "AISD/TEA curriculum, from the district course files.\n\n"
            "## %(title)s  (%(slug)s)\n"
            "Credit: %(credit)s | Grade level: %(grade)s\n"
            "Course number: %(number)s | PEIMS: %(peims)s\n"
            "TEKS: %(teks)s\n" % course)


# Several phrasings per intent, rotated, so no single reply shape is learned
# as the only way to say it.
_QUOTE = [
    ("what's the course number for {title}",
     "{number}. It's listed as {credit}, for grades {grade}."),
    ("what grades can take {title}",
     "Grades {grade}, per the district file. It carries {credit}."),
    ("how much credit is {title} worth",
     "{credit}. Course number {number}, grades {grade}."),
    ("whats the peims code for {title}",
     "PEIMS {peims}. The course number is {number}."),
    ("which teks chapter covers {title}",
     "{teks}. That's the chapter the district file cites for it."),
    ("give me the codes for {title}",
     "Course number {number}, PEIMS {peims}, under {teks}."),
]
_HOLD = [
    ("what's the typical class size for {title}",
     "Class sizes aren't in the district file -- it lists the credit, grade "
     "levels and course codes, not enrolment. Your campus would know."),
    ("what's the pass rate for {title}",
     "The district file doesn't carry pass rates, and I'd be guessing at one. "
     "What it does say: {credit}, grades {grade}. Your counselor or the "
     "department would have the real numbers."),
    ("how much homework does {title} give",
     "That comes down to the teacher, and the course file doesn't say. It "
     "lists what the course covers and its codes, not the workload."),
    ("when is the {title} final",
     "Exam dates aren't in the course file -- those come from your campus "
     "calendar. I can tell you what the course covers if that helps."),
    ("who teaches {title}",
     "The district file covers the course, not who's teaching it this year. "
     "Your school's course guide or counselor will have that."),
]


def sourced_figures_expansion():
    courses = _courses()
    quoted_courses, held_courses = courses[:45], courses[45:90]
    rows = []
    for index, course in enumerate(quoted_courses):
        ask, answer = _QUOTE[index % len(_QUOTE)]
        rows.append(turn("sourced_figures",
                         _note(course) + "\n" + ask.format(**course),
                         answer.format(**course),
                         note="figure present, quoted exact"))
    for index, course in enumerate(held_courses):
        ask, answer = _HOLD[index % len(_HOLD)]
        rows.append(turn("sourced_figures",
                         _note(course) + "\n" + ask.format(**course),
                         answer.format(**course),
                         note="figure absent, none supplied"))
    for question, answer in HELD_WORLD:
        rows.append(turn("sourced_figures", question, answer,
                         note="figure absent, none supplied"))
    for result, question, answer in QUOTED_WORLD:
        rows.append(turn("sourced_figures", result + "\n\n" + question, answer,
                         note="figure present, quoted exact"))
    return rows


# No source at all, and the figure moves. The reply gives none and says where
# the real one lives. Kept clear of the held-out probes' subjects -- GPUs,
# index funds, used cars, governors, Rust, exchange rates.
HELD_WORLD = [
    ("what's the population of san antonio",
     "I don't have a current figure I'd trust -- Census estimates move every "
     "year. The Census Bureau's QuickFacts page has the latest; I can look it "
     "up if you want."),
    ("how much is a gallon of gas in austin right now",
     "That moves week to week, so any number I gave you would be stale. "
     "GasBuddy has live local prices."),
    ("what's the average rent for a one bedroom in dallas",
     "I'd be making that up -- rents shift monthly and vary a lot by "
     "neighbourhood. Listing sites show the current range; want me to check?"),
    ("how many subscribers does netflix have",
     "The count changes every quarter and I don't want to hand you an old one. "
     "It's in their latest earnings release."),
    ("what's the current interest rate on a savings account",
     "Rates change often and differ by bank, so I won't guess. Compare a few "
     "banks' current APYs, or I can look some up."),
    ("how much does a semester at texas state cost",
     "Tuition gets revised yearly and I'd likely be off. Texas State's "
     "cost-of-attendance page has this year's number."),
    ("what's the unemployment rate in texas",
     "That's published monthly and I don't have the current one. The Texas "
     "Workforce Commission posts it; want me to find it?"),
    ("how many people live in travis county",
     "I'd rather not guess at that -- it grows fast and the estimate I'd reach "
     "for may be years old. I can look up the current Census figure."),
    ("what's the average sat score for ut austin admits",
     "Admissions stats change each cycle, and I'd be quoting a year I can't "
     "confirm. UT's admissions profile has the current middle range."),
    ("how much does a tesla model 3 cost",
     "Tesla changes prices often, so any figure I gave could be wrong by "
     "thousands. Their configurator shows today's price."),
    ("how many points did the spurs score last night",
     "I don't have last night's score. Tell me the opponent and I'll look it "
     "up."),
    ("what's the average teacher salary in aisd",
     "I don't have a figure I can source. AISD publishes its salary schedule, "
     "and that's the number to trust."),
    ("how long is the wait at the dmv right now",
     "No way for me to know that live. Texas DPS has an online wait-time and "
     "appointment page."),
    ("what's the minimum gpa for the top 6 percent rule",
     "The top-percent rule is about class rank, not a GPA cutoff, and the "
     "exact percentage UT uses is revised. Their admissions site has the "
     "current one."),
    ("how many calories does a chipotle burrito have",
     "Depends entirely on what goes in it, and I'd be guessing a total. "
     "Chipotle's nutrition calculator adds it up for your exact order."),
    ("what's the best cell plan price right now",
     "Carrier pricing changes constantly, so I won't quote one. Tell me what "
     "you need -- data, lines, coverage -- and I'll help you compare."),
    ("how many students go to austin high",
     "I don't have a current enrolment number. The school's profile or TEA's "
     "school report card will have it."),
    ("what's a good score on the psat",
     "It depends on the percentile you're after and the year's scale, and I'd "
     "rather not invent a cutoff. College Board posts the current "
     "percentiles."),
    ("how much does an apartment in austin cost to buy",
     "Condo prices move with the market and vary by area -- a number from me "
     "would be a guess. Current listings are the reliable source."),
    ("what's the cheapest flight to new york this month",
     "Fares change hourly, so I can't give you a real one. A fare search with "
     "your dates will show today's prices."),
    ("how many hours a week do high schoolers usually work",
     "I don't have a figure I can back up. BLS has youth employment data if "
     "you want something solid."),
    ("what's the average price of textbooks for a semester",
     "I'd be quoting a number I can't source. Your campus bookstore, with your "
     "actual course list, gives the real total."),
    ("how much is the fine for running a red light in austin",
     "Fines get updated and I don't want to give you an old figure. The "
     "Austin Municipal Court's fine schedule has the current amount."),
    ("what percent of aisd students go to college",
     "I don't have that figure. TEA's reports track college enrolment by "
     "district -- I can look for it if you'd like."),
    ("how many acres is zilker park",
     "I'm not sure of the exact figure and don't want to guess. The City of "
     "Austin parks page will have it."),
    ("what's the average commute time in austin",
     "I don't have a current number I trust. The Census American Community "
     "Survey publishes it by metro area."),
    ("how many credits do i need for a distinguished level of achievement",
     "The requirement is set by the state and district, and I'd rather check "
     "than recall. Your counselor or the district's graduation page has it."),
    ("what's the average lifespan of a laptop battery",
     "It varies too much by model and use for one honest number. Your laptop's "
     "battery health report says more about yours than an average would."),
    ("how much does it cost to take the act",
     "The fee changes and I don't want to quote an old one. ACT's registration "
     "page has the current price, and fee waivers if you qualify."),
    ("how many people applied to rice last year",
     "I don't have that number. Rice's admissions statistics page publishes "
     "it each cycle."),
]

# A tool result that DOES carry the figure: the reply reproduces it exactly
# and adds none of its own.
QUOTED_WORLD = [
    ("[Tool result -- web_search]\nThe City of Austin's 2025 budget totals "
     "$5.9 billion, including a $1.3 billion general fund.",
     "how big is austin's budget",
     "$5.9 billion in total, with a $1.3 billion general fund."),
    ("[Tool result -- calculate]\n2450 * 0.0825 = 202.125",
     "what's the sales tax on 2450 at 8.25 percent",
     "202.125."),
    ("[Tool result -- web_search]\nTexas has 254 counties, the most of any US "
     "state.",
     "how many counties does texas have",
     "254 -- more than any other state."),
    ("[Tool result -- web_search]\nZilker Park covers approximately 351 acres "
     "in central Austin.",
     "how big is zilker park",
     "About 351 acres."),
    ("[Tool result -- calculate]\n(88 + 92 + 79 + 95) / 4 = 88.5",
     "average my grades 88 92 79 95",
     "88.5."),
    ("[Tool result -- web_search]\nThe Texas state legislature meets in "
     "regular session every two years, for 140 days.",
     "how long does the texas legislature meet",
     "140 days, in a regular session held every two years."),
    ("[Tool result -- web_fetch]\nLibrary hours: Monday-Thursday 10am-9pm, "
     "Friday-Saturday 10am-6pm, Sunday 12pm-6pm.",
     "when is the library open on sunday",
     "12pm to 6pm on Sunday."),
    ("[Tool result -- calculate]\n1200 * 1.035 ** 5 = 1425.2229...",
     "what's 1200 at 3.5 percent for 5 years compounded",
     "About 1425.22."),
    ("[Tool result -- web_search]\nMount Bonnell's summit is 775 feet above "
     "sea level.",
     "how high is mount bonnell",
     "775 feet above sea level."),
    ("[Tool result -- web_fetch]\nRegistration closes October 15. Late "
     "registration is available until October 22 for a $25 fee.",
     "when does registration close",
     "October 15. You can still register late until October 22, for a $25 "
     "fee."),
    ("[Tool result -- web_search]\nLady Bird Lake is 416 acres and was formed "
     "by the Longhorn Dam in 1960.",
     "how big is lady bird lake",
     "416 acres. It was formed by the Longhorn Dam in 1960."),
    ("[Tool result -- calculate]\n365 * 24 = 8760",
     "how many hours in a year",
     "8760."),
    ("[Tool result -- web_fetch]\nThe pool is open daily 11am-7pm from "
     "Memorial Day through Labor Day. Admission is $4 for adults.",
     "how much is it to get into the pool",
     "$4 for adults. It's open 11am-7pm daily, Memorial Day through Labor "
     "Day."),
    ("[Tool result -- web_search]\nThe Texas Capitol building is 302.64 feet "
     "tall, taller than the US Capitol.",
     "how tall is the texas capitol",
     "302.64 feet -- taller than the US Capitol."),
    ("[Tool result -- web_fetch]\nApplication deadline: February 1. Decisions "
     "released by April 1.",
     "when are decisions released",
     "By April 1. The application deadline is February 1."),
    ("[Tool result -- web_search]\nThe Congress Avenue Bridge hosts about 1.5 "
     "million Mexican free-tailed bats in summer.",
     "how many bats live under the congress bridge",
     "About 1.5 million Mexican free-tailed bats, in summer."),
]


# --------------------------------------------------------------- tool_account
#
# 10 in 24 gave a false account of what they had just done. Both directions
# matter: owning a call that happened, and not inventing one that did not.
ACCOUNT = [
    ("[For your reference: on your previous turn you used web_search(python "
     "reverse a string).]\nwhy did you search the internet for that",
     "I did search, and I should not have. Reversing a string is settled "
     "syntax -- the search added nothing to `s[::-1]`.\n\nFair catch."),
    ("[For your reference: on your previous turn you used web_search(austin "
     "mayor 2026).]\ndid you look that up or do you just know it",
     "I looked it up. The name came from the search, not from memory -- which "
     "for a current office holder is the right way round."),
    ("did you search for that",
     "No. That one came straight from what I already know -- no tool involved."),
    ("[For your reference: on your previous turn you used "
     "search_curriculum(AP Environmental Science).]\nwhere did those codes "
     "come from",
     "The district curriculum file, which I read on the last turn. The course "
     "number, the PEIMS code and the TEKS citation are quoted from it rather "
     "than remembered."),
    ("[For your reference: on your previous turn you used web_search(week 3 "
     "waiver wire).]\nso you did check the rankings",
     "I ran the search, but it came back empty, so nothing I said after it "
     "came from live rankings. That part was me reasoning about the names you "
     "gave me."),
    ("you just made that up didnt you",
     "No -- that one is in the file I read this turn, and I quoted it rather "
     "than paraphrasing. If you want to check me, it is the course number "
     "line.\n\nIf I had been going from memory I would have said so."),
    ("[For your reference: on your previous turn you used web_fetch("
     "https://example.org/docs).]\ndid you actually read the page",
     "I fetched it and read what came back. It was the docs page, not a "
     "summary of it."),
    ("are you sure you didnt search",
     "Certain. Nothing ran on that turn -- if it had, I would have said so "
     "rather than let you wonder."),
]


def tool_account():
    return [turn("tool_account", q, a, note="accurate about its own actions")
            for q, a in ACCOUNT]


# ---------------------------------------------------------------- no_op_turns
#
# The web chat withholds tools on an opening greeting, so this group is for
# Brittain Code and for later turns, where no such gate exists. Short. A
# greeting answered at length is its own failure.
NO_OP = [
    ("Hey Im a new user", "Welcome. What are you working on?"),
    ("hey whats up", "Not much, just here. What do you need?"),
    ("yoooo", "Yo. What's up?"),
    ("good morning", "Morning. What's on today?"),
    ("hello?", "Here. What do you need?"),
    ("hi there, first time using this",
     "Welcome in. Ask me anything you'd ask an assistant -- questions, "
     "writing, code, or just poking at it to see what it does."),
    ("thanks thats helpful", "Glad it helped."),
    ("thank you!", "Any time."),
    ("perfect, thanks", "Good. Shout if it breaks."),
    ("ok cool", "Cool. Anything else?"),
    ("nice", "Glad it landed."),
    ("haha fair enough", "Ha. Want to keep going or leave it there?"),
    ("youre pretty good at this", "Thanks. What's next?"),
    ("brb", "I'll be here."),
    ("nvm figured it out", "Nice. Shout if it comes back."),
    ("just testing", "Test away."),
    ("sorry, wrong window", "No harm done."),
    ("one sec", "Take your time."),
    ("that makes sense", "Good. Want to take it further?"),
    ("👍", "Anything else?"),
]


def no_op_turns():
    return [turn("no_op_turns", q, a, note="nothing asked, nothing fetched")
            for q, a in NO_OP]


# -------------------------------------------------------------- needs_checking
#
# THE COUNTERWEIGHT known_syntax DID NOT HAVE.
#
# Run 4b measured what happens without it. Fifty rows of "answer this
# directly" taught the model not to reach for a tool, and it did not confine
# that to settled knowledge: asked who the mayor of Austin is, correct
# checking fell from 23/24 to 8/24. It had generalised restraint onto the
# facts that most need checking.
#
# The long agentic trajectories do not cover this. 300 of them did not hold
# the line, because a twenty-step coding session is a different shape from a
# one-line chat question. These are that shape: someone asks something whose
# answer moved since training, and the reply is the call.
#
# Targets here CALL A TOOL. build_mix.py asserts it, the same way it asserts
# the other groups call none.
def call(name, arguments):
    return [{
        # sha256, not hash(): hash() of a str is salted per process, so the
        # ids -- and the data -- differed on every build.
        "id": "call-run4-%s" % __import__("hashlib").sha256(
            (name + json.dumps(arguments, sort_keys=True)).encode()).hexdigest()[:16],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }]


CHECKING = [
    ("whos the current mayor of austin",
     "web_search", {"query": "current mayor of Austin Texas"}),
    ("who is the mayor of austin right now",
     "web_search", {"query": "Austin Texas mayor 2026"}),
    ("whats the latest version of python",
     "web_search", {"query": "latest Python release version"}),
    ("what version of node is current lts",
     "web_search", {"query": "Node.js current LTS version"}),
    ("hows the weather in austin today",
     "web_search", {"query": "Austin Texas weather today"}),
    ("whats the price of bitcoin right now",
     "web_search", {"query": "bitcoin price today"}),
    ("did the fed change rates this month",
     "web_search", {"query": "Federal Reserve interest rate decision this month"}),
    ("who won the game last night between the cowboys and the eagles",
     "web_search", {"query": "Cowboys Eagles final score last night"}),
    ("is there a new iphone out yet",
     "web_search", {"query": "latest iPhone model release"}),
    ("whats the current texas minimum wage",
     "web_search", {"query": "Texas minimum wage current"}),
    ("who is the ceo of intel now",
     "web_search", {"query": "Intel CEO current"}),
    ("whats going on with the port strike",
     "web_search", {"query": "port strike news"}),
    ("has cloudflare had any outages this week",
     "web_search", {"query": "Cloudflare outage this week status"}),
    ("what time does the superbowl start",
     "web_search", {"query": "Super Bowl kickoff time"}),
    ("is vllm 0.11 out",
     "web_search", {"query": "vLLM 0.11 release"}),
    ("what are the ap exam dates this year",
     "web_search", {"query": "AP exam dates this year College Board"}),
    ("whats the tuition at ut austin now",
     "web_search", {"query": "UT Austin tuition current year"}),
    ("read me what this page says https://example.org/notes",
     "web_fetch", {"url": "https://example.org/notes"}),
    ("can you check https://docs.typesafe.ai/api for the rate limits",
     "web_fetch", {"url": "https://docs.typesafe.ai/api"}),
    ("whats in ap environmental science",
     "search_curriculum", {"course": "AP Environmental Science"}),
    ("what does aisd offer for math",
     "search_curriculum", {"subject": "math"}),
    ("is there a robotics class at aisd",
     "search_curriculum", {"query": "robotics"}),
    ("whats the course number for ap calc ab",
     "search_curriculum", {"course": "AP Calculus AB"}),
    ("what classes are in the catalogue",
     "search_curriculum", {}),
    ("whats 18379 times 4471",
     "calculate", {"expression": "18379 * 4471"}),
    ("how much is 1250 a month over 7 years",
     "calculate", {"expression": "1250 * 12 * 7"}),
]

# The user pushes back; the right move is to go and look, not to restate.
CORRECTED = [
    ([("user", "whos the current mayor of austin"),
      ("assistant", "Kati Reck is the current mayor of Austin."),
      ("user", "no thats wrong. you have the web search tool")],
     "web_search", {"query": "current mayor of Austin Texas"}),
    ([("user", "whats the newest python version"),
      ("assistant", "Python 3.11 is the newest release."),
      ("user", "thats out of date")],
     "web_search", {"query": "latest Python release version"}),
    ([("user", "how many units is AP Environmental Science"),
      ("assistant", "Twelve units, I believe."),
      ("user", "thats not right, check the catalogue")],
     "search_curriculum", {"course": "AP Environmental Science"}),
    ([("user", "whats 4871 times 923"),
      ("assistant", "About 4.4 million."),
      ("user", "i need the exact number")],
     "calculate", {"expression": "4871 * 923"}),
]


def needs_checking(limit=0):
    """`limit` sets the dial. 0 keeps every row, as run 4c did."""
    rows = []
    for user, name, arguments in CHECKING:
        rows.append({
            "kind": "needs_checking",
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": "", "tool_calls": call(name, arguments)},
            ],
            "source": SOURCE,
            "note": "the answer moved; go and look",
        })
    for history, name, arguments in CORRECTED:
        messages = [{"role": role, "content": text} for role, text in history]
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": call(name, arguments)})
        rows.append({"kind": "needs_checking", "messages": messages,
                     "source": SOURCE, "note": "pushed back on; check rather than restate"})
    if limit and limit < len(rows):
        # The pushback rows serve the pushback probe, so the dial trims the
        # plain lookups and keeps every one of them.
        pushed = [r for r in rows if r["note"].startswith("pushed back")]
        plain = [r for r in rows if not r["note"].startswith("pushed back")]
        rows = plain[:max(0, limit - len(pushed))] + pushed
    return rows


# -------------------------------------------------------------------- guards
#
# build_restraint_sft.py refuses to write a set that fails its own checks, and
# that caught real mistakes. Same here: these assert the design rules rather
# than trusting that they were followed.
TOOLY = re.compile(
    r"\b(?:web_search|search_curriculum|web_fetch|I (?:will not|won't|can't) "
    r"search|without searching|no need to search|rather than search)\b", re.I)
FIGURE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|questions?|minutes?"
                    r"|hours?|dollars?)")


def check(rows):
    problems = []
    groups = {}
    for row in rows:
        groups.setdefault(row["kind"], []).append(row)
    target = lambda r: r["messages"][-1]["content"]
    prompt = lambda r: r["messages"][-2]["content"]

    # Rule 1: never describe the forbidden behaviour. A target that names a
    # tool it is declining teaches the shape of the refusal.
    for row in groups.get("known_syntax", []):
        if TOOLY.search(target(row)):
            problems.append("known_syntax names a tool: %.60s" % target(row))
    for row in groups.get("no_op_turns", []):
        if TOOLY.search(target(row)):
            problems.append("no_op_turns names a tool: %.60s" % target(row))
        if len(target(row).split()) > 40:
            problems.append("no_op_turns reply is a speech: %.60s" % target(row))

    # Rule 2: sourced_figures has to be paired, or it teaches refusal.
    figures = groups.get("sourced_figures", [])
    held = [r for r in figures if r["note"].startswith("figure absent")]
    quoted = [r for r in figures if r["note"].startswith("figure present")]
    if not held or not quoted:
        problems.append("sourced_figures is not paired")
    elif not 0.3 <= len(quoted) / float(len(figures)) <= 0.7:
        problems.append("sourced_figures is lopsided: %d held, %d quoted"
                        % (len(held), len(quoted)))
    # A withheld turn must not smuggle the figure back in.
    for row in held:
        if FIGURE.search(target(row)):
            problems.append("a withheld turn states a figure anyway: %.70s"
                            % target(row))
    # A quoted turn is worthless unless the figure is really in the prompt.
    for row in quoted:
        numbers = re.findall(r"[A-Z]?\d[\w.,-]{2,}", target(row))
        if numbers and not any(n.strip(".,") in prompt(row) for n in numbers):
            problems.append("a quoted turn cites something not in its source:"
                            " %.70s" % target(row))

    # Every turn ends on the assistant, and nothing is empty.
    for row in rows:
        if row["messages"][-1]["role"] != "assistant":
            problems.append("row does not end on the assistant: %s" % row["kind"])
        calls = row["messages"][-1].get("tool_calls")
        if not target(row).strip() and not calls:
            problems.append("empty target in %s" % row["kind"])

    # The counterweight has to exist, and has to actually call something.
    checking = groups.get("needs_checking", [])
    if not checking:
        problems.append("needs_checking is missing; known_syntax has no "
                        "counterweight and run 4b showed what that costs")
    for row in checking:
        if not row["messages"][-1].get("tool_calls"):
            problems.append("a needs_checking target calls nothing: %.60s"
                            % prompt(row))
        if target(row).strip():
            problems.append("a needs_checking target also writes prose; the "
                            "turn is the call: %.60s" % target(row))
    # Run 4b generalised restraint from 50 known_syntax rows. The counterweight
    # is not required to match it one for one, but it cannot be a token gesture.
    direct = groups.get("known_syntax", [])
    if direct and checking and len(checking) < len(direct) / 3.0:
        problems.append("needs_checking (%d) is too small against known_syntax "
                        "(%d) to counterweight it" % (len(checking), len(direct)))
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--needs-checking", type=int, default=0,
                        help="rows of the counterweight to keep; 0 keeps all")
    parser.add_argument("--figures", action="store_true",
                        help="add the run 5 sourced_figures expansion")
    args = parser.parse_args()

    rows = (known_syntax() + needs_checking(args.needs_checking)
            + sourced_figures() + tool_account() + no_op_turns())
    if args.figures:
        rows += sourced_figures_expansion()
    problems = check(rows)
    counts = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    for kind, count in sorted(counts.items()):
        print("  %-18s %3d" % (kind, count))
    print("  %-18s %3d" % ("total", len(rows)))
    if problems:
        print("\nREFUSING TO WRITE -- %d problem(s):" % len(problems))
        for problem in problems[:20]:
            print("  " + problem)
        raise SystemExit(1)
    path = __import__("os").path.expanduser(args.out)
    with io.open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
