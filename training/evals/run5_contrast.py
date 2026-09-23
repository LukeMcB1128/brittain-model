# -*- coding: utf-8 -*-
"""Contrast pairs for run 5c: settled against moving, inside each domain.

WHY
On clean probes, reaching for a tool on settled knowledge and checking a fact
that moves travel together. With no counterweight run 4b over-reached 5 times
in 24 and checked a moving fact 9 times; with thirty counterweight rows run 4c
over-reached 20 and checked 23. No point between them clears both halves of
the gate, because the data moved one thing: how readily the model calls a
tool.

What the data never did was draw the line inside a domain. Every settled row
was coding and almost every moving row was world news, so "coding means
answer, news means search" was as learnable as "settled means answer, moving
means search". These pairs put both sides in the same domain -- how to write
a decorator next to what the latest Django is; why the sky is blue next to
whether it rains tomorrow -- so only the second rule fits.

Settled answers are correct and state their facts plainly. None names a tool
or says it is not searching: prohibitions backfire on this model, measured.
Everything here is kept clear of the held-out probes' subjects, and
audit_contamination.py confirms it before anything trains.
"""

# Settled coding knowledge, answered directly. -> known_syntax
SETTLED_CODING = [
    ("how do i convert a list to a set in python",
     "`set(items)`. Duplicates collapse and the order is lost; "
     "`dict.fromkeys(items)` keeps the first-seen order if you need it."),
    ("what does git rebase do",
     "It replays your branch's commits on top of another branch, so the history "
     "reads as if you had started from there. Unlike a merge it rewrites those "
     "commits, so don't rebase commits other people have already pulled."),
    ("what is a decorator in python",
     "A function that takes a function and returns a new one, usually wrapping "
     "the original with extra behaviour. `@timer` above a `def` is shorthand "
     "for `f = timer(f)`."),
    ("how do i read an environment variable in python",
     "`os.environ.get('NAME')`, which returns `None` when it's unset. "
     "`os.environ['NAME']` raises `KeyError` instead -- what you want when the "
     "program can't run without it."),
    ("what does git cherry-pick do",
     "It copies one commit from elsewhere onto your current branch as a new "
     "commit: `git cherry-pick <sha>`. Conflicts resolve the same way as in a "
     "merge."),
    ("how do i loop with an index in python",
     "`for i, item in enumerate(items):`. Use `enumerate(items, start=1)` to "
     "count from one."),
    ("how do i parse a date string in python",
     "```python\nfrom datetime import datetime\n\n"
     "datetime.strptime('2024-03-15', '%Y-%m-%d')\n```\n\n"
     "`datetime.fromisoformat` handles most ISO strings directly."),
    ("how do i remove a key from a dict in python",
     "`d.pop('key', None)` removes it and returns the value, or `None` if it "
     "wasn't there. `del d['key']` raises `KeyError` when the key is missing."),
    ("what is a closure in javascript",
     "A function that keeps access to variables from the scope it was created "
     "in, even after that scope has returned. It's how a counter function "
     "remembers its count between calls."),
    ("how do i check a variable's type in python",
     "`isinstance(x, int)` for a check, since it respects subclasses. `type(x)` "
     "gives the exact type, which is mostly useful when debugging."),
    ("what does git blame show",
     "Who last changed each line of a file, and in which commit. "
     "`git blame -L 10,20 file` narrows it to a range of lines."),
    ("how do i split a string on commas in python",
     "`text.split(',')`. If there are spaces after the commas: "
     "`[part.strip() for part in text.split(',')]`."),
    ("difference between null and undefined in javascript",
     "`undefined` means nothing was ever assigned; `null` is an explicit "
     "'nothing' that someone set. `null == undefined` is true, "
     "`null === undefined` is false."),
    ("what does __init__ do in a python class",
     "It initialises a new instance: it runs right after the object is created, "
     "and it's where you set the starting attributes on `self`."),
    ("what's the time complexity of a python dict lookup",
     "O(1) on average, because it's a hash table. The worst case is O(n) under "
     "pathological hash collisions, which ordinary keys won't hit."),
    ("how do i pass command line arguments to a python script",
     "They arrive in `sys.argv`, with the script name at index 0. For more than "
     "one or two, use `argparse` -- you get `--help` and type checking for "
     "free."),
    ("what does npm install --save-dev do",
     "It installs the package and records it under `devDependencies` in "
     "`package.json`: things needed to build or test, not to run in "
     "production."),
    ("what does a 404 mean",
     "The server understood the request but has nothing at that URL. It says "
     "nothing about whether the resource ever existed."),
    ("how do i comment out several lines in python",
     "Put `#` at the start of each line; most editors do a whole selection with "
     "Ctrl+/ (Cmd+/ on a Mac). A triple-quoted string also works, but it's a "
     "string rather than a comment."),
    ("difference between a stack and a queue",
     "A stack is last in, first out, like a pile of plates. A queue is first "
     "in, first out, like a line at a counter."),
    ("what does parsefloat do in javascript",
     "It reads a decimal number from the start of a string and stops at the "
     "first character that can't be part of one: `parseFloat('3.14abc')` is "
     "`3.14`. `Number()` is stricter and gives `NaN` unless the whole string "
     "is numeric."),
    ("what is recursion",
     "A function solving a problem by calling itself on a smaller version of "
     "it, with a base case that stops the calls. Factorial is the textbook "
     "case: `n! = n * (n-1)!`, stopping at 1."),
    ("how do i write rows to a csv in python",
     "```python\nimport csv\n\nwith open('out.csv', 'w', newline='', "
     "encoding='utf-8') as f:\n    csv.writer(f).writerows(rows)\n```\n\n"
     "`newline=''` stops blank lines appearing between rows on Windows."),
    ("difference between git clone and git pull",
     "`clone` makes a new local copy of a repository. `pull` updates a copy you "
     "already have with the remote's new commits."),
    ("what is an api",
     "An application programming interface: the requests one piece of software "
     "accepts from another, and what it returns. A web API usually means HTTP "
     "endpoints that return JSON."),
]

# Settled general knowledge, answered directly. -> known_general
SETTLED_GENERAL = [
    ("why is the sky blue",
     "Sunlight scatters off the molecules in the air, and shorter wavelengths "
     "scatter far more than longer ones -- Rayleigh scattering. Blue light is "
     "scattered across the whole sky, so it reaches you from every direction."),
    ("what causes the seasons",
     "The tilt of Earth's axis, about 23.5 degrees. For part of the orbit your "
     "hemisphere leans toward the sun and gets more direct light and longer "
     "days; half a year later it leans away. Distance from the sun has very "
     "little to do with it."),
    ("what's the boiling point of water",
     "100°C, or 212°F, at sea level. It drops as you go higher, because the "
     "air pressure is lower."),
    ("who wrote pride and prejudice",
     "Jane Austen. It was published in 1813."),
    ("what's the capital of australia",
     "Canberra -- not Sydney, which is the largest city."),
    ("what is photosynthesis",
     "How plants make their own food: they use light to turn carbon dioxide and "
     "water into glucose, releasing oxygen along the way. It happens mostly in "
     "the chloroplasts of leaf cells."),
    ("how does a bill become a law in the us",
     "It's introduced in the House or the Senate, goes through committee, and "
     "has to pass both chambers in the same form. The president then signs or "
     "vetoes it, and Congress can override a veto with a two-thirds vote in "
     "both chambers."),
    ("difference between weather and climate",
     "Weather is what the atmosphere is doing over hours or days. Climate is "
     "the long-run pattern of weather in a place, usually averaged over thirty "
     "years or more."),
    ("when did world war two end",
     "1945. Germany surrendered in May, and Japan in August, with the formal "
     "surrender signed on 2 September."),
    ("what is the pythagorean theorem",
     "In a right triangle, the square of the hypotenuse equals the sum of the "
     "squares of the other two sides: a^2 + b^2 = c^2."),
    ("how fast is light",
     "About 299,792 kilometres per second in a vacuum -- roughly 186,000 miles "
     "per second."),
    ("who painted the mona lisa",
     "Leonardo da Vinci, in the early 1500s."),
    ("what's the chemical symbol for gold",
     "Au, from the Latin aurum."),
    ("how many bones are in the human body",
     "206 in an adult. Babies are born with more, and some fuse as they grow."),
    ("what does dna stand for",
     "Deoxyribonucleic acid."),
    ("what's the biggest planet",
     "Jupiter -- more than twice as massive as all the other planets combined."),
    ("what is a prime number",
     "A whole number greater than 1 whose only divisors are 1 and itself. 2, "
     "3, 5, 7 and 11 are the first few."),
    ("why do we have leap years",
     "A year is about 365.24 days, not 365. Adding a day every four years keeps "
     "the calendar lined up with the seasons, and skipping most century years "
     "corrects the small overshoot."),
    ("what is inflation",
     "A general rise in prices over time, so each unit of money buys a little "
     "less. It's usually measured as the yearly change in a consumer price "
     "index."),
    ("difference between a virus and bacteria",
     "Bacteria are living single cells that reproduce on their own, and many "
     "are harmless or useful. Viruses are far smaller and can only reproduce "
     "inside a host's cells. Antibiotics work on bacteria, not viruses."),
]

# Moving facts, in both domains. The target is the call. -> needs_checking
MOVING = [
    ("what's the latest version of django", "web_search", {"query": "latest Django release"}),
    ("is python 3.9 still getting security updates", "web_search", {"query": "Python 3.9 end of life date"}),
    ("what's new in the most recent typescript release", "web_search", {"query": "latest TypeScript release notes"}),
    ("has numpy dropped support for python 3.10 yet", "web_search", {"query": "NumPy supported Python versions"}),
    ("what's the current stable postgres version", "web_search", {"query": "PostgreSQL current stable release"}),
    ("what's the newest react version", "web_search", {"query": "latest React release"}),
    ("did github change copilot pricing recently", "web_search", {"query": "GitHub Copilot pricing change"}),
    ("which ubuntu lts is newest", "web_search", {"query": "latest Ubuntu LTS release"}),
    ("is the requests library still maintained", "web_search", {"query": "python requests library maintenance status"}),
    ("what's the current version of pip", "web_search", {"query": "latest pip release"}),
    ("has node 18 reached end of life", "web_search", {"query": "Node.js 18 end of life"}),
    ("is npm having an outage right now", "web_search", {"query": "npm registry status outage"}),
    ("who won the most recent world series", "web_search", {"query": "most recent World Series winner"}),
    ("what's the population of texas now", "web_search", {"query": "Texas population latest estimate"}),
    ("is i-35 backed up right now", "web_search", {"query": "I-35 Austin traffic now"}),
    ("what's the uv index in austin today", "web_search", {"query": "Austin UV index today"}),
    ("did the austin city council pass the budget yet", "web_search", {"query": "Austin city council budget vote"}),
    ("what are the central library's hours today", "web_search", {"query": "Austin Central Library hours today"}),
    ("is it going to rain tomorrow in austin", "web_search", {"query": "Austin weather tomorrow rain"}),
    ("who is the current secretary of state", "web_search", {"query": "current US Secretary of State"}),
]
