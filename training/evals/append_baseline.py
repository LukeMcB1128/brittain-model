# -*- coding: utf-8 -*-
"""Record the default-prompt numbers in BASELINE.md."""

SECTION = """

## Default system prompt (chat template)

`chat_template_brittain4.jinja` is Qwen's template plus one else-branch in each
of the two system-message paths, injecting a default system prompt **only when
the caller sends none**. Brittain Code sends its own `systemPrompt()`, so it is
unaffected; this reaches the web chat and bare API callers. Built by
`evals/make_template.py`, served via `--chat-template`.

Verified on the served stack (`evals/verify_template.py`, all pass):

| check | result |
|---|---|
| identity with no system message, 4 probes incl. pressure and sideways | pass, no other lab named |
| caller system message wins, default absent | pass |
| tools still serialise, tool call parses as JSON | pass |
| identity holds with tools present | pass |

### The prompt has no response-style paragraph, deliberately

Adding style guidance ("lead with the conclusion", "do not restate the
question", "say what you are unsure of") made the model **refuse live-data
questions** — "I cannot provide today's exchange rate because I do not have
real-time access" — instead of searching, and invent file contents rather than
reading them.

No single paragraph causes it. Alone, each is fine. It is an interaction
between the identity and style paragraphs (`evals/prompt_isolation_pairs.py`,
n=10 per cell, live-data search):

| system prompt | search | refusals | restraint |
|---|---|---|---|
| identity + tools **(shipped)** | 10/10 | 0 | 3/3 |
| tools + style | 8/10 | 0 | 3/3 |
| identity + style | 0/10 | 2 | 3/3 |
| all three | 0/10 | 4 | 3/3 |
| neutral "You are a helpful assistant." | 4/10 | 0 | — |

Ruled out first: the Jinja injection is faithful (identical text sent as a
caller system message scores identically), and length is not the cause (filler
prose of the same length scored 5/6 where the real text scored 3/6) —
`evals/prompt_isolation_length.py`. Three rewordings failed to break the
interaction: scoping to "once you have what you need", deleting the uncertainty
clause, and reordering style before tools (worst, 6 refusals).

Shipped prompt on the served path, 8 tool-needing + 5 answerable-from-knowledge
tasks (`evals/score_prompt_toolcalling.py`): **13/13** — 8/8 uses a tool where
one is needed, 5/5 answers directly where none is.

Restraint is scored alongside use on purpose: a prompt that shells out to
compute 2+2 is the opposite failure, not an improvement.

## Identity training data

`data/identity_sft.jsonl`, 161 examples from `evals/build_identity_sft.py`:
113 identity-bearing (direct, origin, pressure, sideways-completion,
comparison, architecture-deflection) and 48 neutral turns that mention no
identity at all — the counterweight that stops the adapter learning to announce
itself unprompted.

Each identity example carries one of three system contexts. The **foreign** one
(a Brittain Code-shaped prompt with no identity line) is the reason to train
this rather than rely on the template: in the app the caller always supplies its
own system prompt, so the template's default never fires and identity has to
come from the weights.

## Serving note

The API key moved from `--api-key` to the `VLLM_API_KEY` environment variable.
`--api-key` put the key in the process command line, visible to anyone who can
run `ps auxww`.
"""

p = "/home/lukeb/brittain4/BASELINE.md"
existing = open(p, encoding="utf-8").read()
if "Default system prompt (chat template)" in existing:
    raise SystemExit("section already present; not appending twice")
open(p, "a", encoding="utf-8").write(SECTION)
print("appended %d chars to %s" % (len(SECTION), p))
print("file is now %d lines" % (len(open(p, encoding="utf-8").read().splitlines())))
