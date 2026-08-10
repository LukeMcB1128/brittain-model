# brittain-shakespeare

A small narrative-prose model that writes short stories to order. It is a
separate model from the Brittain code family, but it reuses the Brittain3
engineering stack: `src/brittain/model_v3.py`, `src/brittain/training_v3.py`, and
`scripts/train/pretrain_v3.py` are shared without modification.

Training target hardware is a single RTX 3060 12GB with 32GB DDR4 and an
i3-12100F. Everything below is sized for that machine.

## The idea

Every pretraining document carries the constraints that describe it, as a tag
block placed before the text:

```
<|story_start|><|tags|>[Voice: Modern] [Genre: Tragedy] [Setting: Tavern]<|end_tags|>
The wooden door slammed...
```

Because the tags are present from the first pretraining token, they are native
structure rather than a post-hoc instruction layer. By the end of training they
act as control levers. A later SFT pass wraps them in natural-language
instructions; the special tokens for that are reserved in the tokenizer now.

## Scope

Two constraints shape the corpus and the model size.

**Story prose only.** World facts are allowed as story furniture — a city exists,
it is on a coast, a ship sails from it. They are never the subject. Essays,
treatises, science writing, reference works, sermons, travelogues, and biography
are excluded. At this parameter count there is no capacity to spend on knowledge
that never appears inside a story.

**Both registers, tag-selectable.** Early Modern English and modern prose are
both trained, reached through `[Voice: Shakespearean]` and `[Voice: Modern]`.

## Tag schema

`src/brittain/tags.py` defines nine tags with closed value vocabularies and a
canonical order. Tag values are ordinary bracketed text, not special tokens, so
an unseen value such as `[Setting: Lighthouse]` still composes from BPE pieces at
inference. Only `<|tags|>` and `<|end_tags|>` are special.

| Tag | Values | Where the label comes from |
|---|---|---|
| `Voice` | Shakespearean, Victorian, Modern | archaic morphology, else author life dates |
| `Genre` | Tragedy, Comedy, Romance, Mystery, Ghost, Adventure, Fable, Drama | bookshelves, else subject headings |
| `POV` | First, Third-Limited, Third-Omniscient | pronoun statistics over narration |
| `Tense` | Past, Present | verb morphology over narration |
| `Setting` | Tavern, Castle, Sea, Forest, City, Household, Court, Road, Battlefield | place lexicon |
| `Tone` | Dark, Wry, Tender, Bleak, Rousing, Uneasy | affect lexicon |
| `Cast` | Solo, Pair, Ensemble | distinct repeated character names |
| `Length` | Flash, Short, Long | exact token count of the window |
| `Twist` | Betrayal, Revelation, Reversal, Death, Reunion, None | synthetic stories only |

## Where the labels come from

Eight of the nine tags are facts about the text that can be computed from the
text or its bibliographic metadata. `src/brittain/story_tagger.py` computes them
with regular expressions and lexicons — no model, no API, no per-token cost.

This matters for two reasons beyond cost.

1. **Labels are correct by construction.** A tag the model cannot verify against
   the text teaches it that tags are noise, and the control lever goes dead.
2. **The same code scores generated samples.** Tag adherence becomes a measured
   number rather than an opinion. See the evaluation section.

Every extractor returns nothing when the evidence is weak — below a minimum
window length, or without a clear margin over the runner-up value. An absent tag
is already in distribution because of tag dropout, so silence is always cheaper
than a wrong label. `Third-Limited` versus `Third-Omniscient` is the weakest
distinction in the set and declines most often.

One trap is worth recording, because it would have been invisible in training.
**`Voice` cannot come from publication year.** Gutenberg's `dcterms:issued` is the
date Gutenberg released the book, not the date it was written — Dracula is stamped
`1995-10-01`, not 1897 — and original publication year is not in the metadata at
all. Deriving the register from it would have labelled nearly the whole corpus
`Modern` and trained the register lever on noise, while every test still passed.
`Voice` therefore comes from archaic morphology in the text, falling back to the
author's death year, which the RDF does carry and which approximates the end of a
career well enough to place a register. Text outranks dates so that a modern
author writing in period voice is labelled by what they actually wrote.

`Twist` is the one genuinely interpretive tag and has no extractor. Real books
carry no `Twist`; the synthetic story set carries it on every example. If pilot
adherence for `Twist` comes back weak, the fallback is a small classifier trained
on the pilot's own hidden states plus a few thousand hand-labeled windows, run
offline on the 3060. That is not on the critical path.

## Corpus

The versioned policy is `configs/data/shakespeare_corpus.json`. Target is about
1.6B training tokens, roughly 20x parameters.

| Source | Tokens | Notes |
|---|---:|---|
| Project Gutenberg, fiction subset | ~1.2B | LCC and subject filtered |
| Standard Ebooks | ~120M | clean typography, optional |
| World texture | ~80M, capped at 5% | legends, folk tales, war memoirs, Bible stories |
| Early Modern | ~12M, upsampled 4x | Shakespeare, Marlowe, Jonson, Webster, KJV narrative books |
| Synthetic tagged stories | ~60M | dense `Twist` and `Tone` supervision |

Gutenberg comes from the **official mirror**, not a Hugging Face mirror. The
fiction filter and the `Genre` tag both run on Library of Congress subject
headings, and the HF mirrors generally ship the text with those headings
stripped. Two feeds carry what is needed:

- `https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2`, daily, the
  richer of the two — LCSH subjects, LCC class, curated bookshelves, author birth
  and death dates, language, and rights.
- `https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv`, weekly, a flatter
  summary.

Standard Ebooks bulk downloads are gated behind Patrons Circle membership; the
free route is cloning the per-book repositories on GitHub. At roughly 7% of the
corpus it is optional rather than blocking.

The fiction filter is the load-bearing part of the scope constraint. A document
must carry a literature LCC class (`PR`, `PS`, `PQ`, `PT`, `PZ`), match a
narrative subject heading, match no excluded heading, and clear a dialogue-density
floor and a narrative-verb-ratio floor. The LCC class is a structural signal and
a far cleaner first gate than matching the word "fiction" against free text, but
it is necessary rather than sufficient — `PR` also holds poetry, criticism, and
essays, which is why the other rules stay. Boilerplate, transcriber notes, front
matter, and indexes are stripped.

### World texture

A second, capped acceptance path admits narrative prose that happens to be
shelved under history, religion, or folklore: legends and folk tales, saints'
lives, first-person war memoirs, Bible-story retellings. It exists so a story set
during the Crusades or a war has period furniture to draw on.

It is not a route to encyclopedic knowledge, and the distinction is worth stating
precisely, because the obvious version of this idea is wrong. Of the 23 books in
the catalog whose subjects mention the Crusades, 13 are *already accepted* by the
fiction path — Scott's *The Talisman* and *The Betrothed*, Henty's *Winning His
Spurs*, *Via Crucis*, *"God Wills It!"*. What the LCC gate rejects is *The History
of the Crusades* in three volumes. Historical fiction is what teaches the model to
write a knight outside Acre; campaign chronology teaches it dates. The texture
path deliberately keeps the first and not the second.

Three rules gate it, and a book must pass all of them:

- a history, religion, or folklore LCC prefix — philosophy, psychology, ethics,
  and doctrinal and practical theology are deliberately absent, being argument
  rather than narrative;
- a narrative subject marker such as legends, tales, folklore, chronicles, or
  personal narratives;
- the text floors, set stricter than on the fiction path.

The floors do the real work. They are what separates *Tales and Legends of the
English Lakes* from a regnal chronology.

`biography` is allowed as a marker here although the fiction path excludes it,
because saints' lives and first-person war memoirs are the most narrative material
in these classes. That relaxation needed compensating exclusions: without them it
also admitted Mussolini's *My autobiography*, travel writing, a linguistics
reference, and regimental unit histories. Excluding travel, politics, languages,
campaigns, and unit histories cut the admitted set from 2,366 books to 1,736,
which sample as folk tales, war memoirs, mythology, and frontier narrative.

The Bible is handled separately. The KJV needs **book-level** selection — Genesis,
Exodus, Judges, Samuel, Kings, the Gospels, and Acts are narrative, while Leviticus
is law and the Epistles are doctrinal argument — and the text floors run per file,
so they cannot make that cut inside a single KJV file. The narrative books are
therefore listed explicitly in the local Early Modern source, which also earns them
the 4x upsampling and the Shakespearean register. Gutenberg's `BS` and `BX` classes
contribute Bible-*story* retellings through the ordinary texture path.

### Measured funnel

Run against the full catalog, snapshot 2026-08-10, 79,148 records in 84 seconds:

| Stage | Books |
|---|---:|
| Catalog records read | 79,148 |
| Rejected: non-literature LCC class | 30,737 |
| Rejected: not English | 16,309 |
| Rejected: excluded subject | 7,103 |
| Rejected: no narrative subject | 2,486 |
| Rejected: no author dates | 2,448 |
| Rejected: rights | 769 |
| **Passed metadata rules** | **21,032** |
| — of which fiction | 19,296 |
| — of which world texture | 1,736 |

At typical novel length that is comfortably above the 1.2B token target before the
text floors are applied, so the filter can afford to stay strict.

The LCC gate was checked rather than assumed: only 51 of its 30,737 rejections
lacked an LCC code at all. The rest carry real non-literature classes — AP
periodicals, DA British history, PN criticism and journalism, and religion,
biology, and history classes below those. It is discarding what it is meant to.

Metadata tag coverage across the 19,296 survivors is `Voice` 98.5% and `Genre`
57.6%. The genre-less remainder is dominated by `Category: Novels` and the bare
heading `Fiction` — books that are genuinely genre-neutral. Leaving them
unlabeled is correct; tag dropout already makes an absent `Genre` in
distribution, and inventing a label would be the one thing that breaks a lever.

`Voice` splits Victorian 54.9%, Modern 41.4%, Shakespearean 2.3%. The Early Modern
share is thin, which is what the 4x upsampling of the local Early Modern corpus
and the `voice_from_text` fallback exist to address.

Whole books go to train or validation, never individual windows. The book
identifier occupies the `repository` field so the existing
`data_v3.split_by_repository` performs the split unchanged.

The i3-12100F has four cores, and tokenizing 1.6B tokens is the CPU-bound step of
the whole project. Corpus preparation runs in resumable chunks and is expected to
take overnight.

## Data preparation

`data_v3.encode_document` truncates a document to one block, which is right for
source files and wrong for novels. `src/brittain/data_story.py` replaces that step
with windowing, then reuses `pack_segments`, `token_controlled_mix`, and
`write_dataset` unchanged.

Four randomization policies decide whether the levers actually work. They live in
`tags.TagPolicy` and are mirrored in the corpus config.

- **Independent tag dropout**, p=0.30 per tag, plus **block dropout**, p=0.10.
  Without this the model only works with a full nine-tag specification and
  degrades on partial ones. This is the most important knob in the project.
- **Order shuffle**, p=0.15, so tag position is not load-bearing.
- **Tag-block loss masking**, p=0.80. Predicting tags from nothing is a
  high-entropy task that wastes capacity at this scale. The remaining 20% keeps
  auto-tag-completion and reverse tagging working.
- **Reverse examples**, p=0.05, with the block after the story, which strengthens
  the association in both directions cheaply.

## Tokenizer

`tokenizers/brittain-shakespeare-prose-8k/`, a byte-level BPE with a vocabulary
of 8,192, trained on a held-out slice of the prose corpus including the Early
Modern text so that archaic spellings and elisions get merges.

`tokenizer_v3.validate_tokenizer` hard-asserts a vocabulary of 24,576 and code
samples, so the prose tokenizer gets a sibling module rather than a parameter.

Special tokens, with the chat set reserved now because adding it later would
strand the embeddings:

```
<|endoftext|> <|pad|> <|story_start|> <|story_end|> <|tags|> <|end_tags|>
<|title|> <|system|> <|user|> <|assistant|> <|end_message|>
```

## Model and training

Maximum context is 4,096 — about 3,000 words, a real short story, and comfortable
in 12GB.

| | Pilot | Production |
|---|---:|---:|
| Parameters | ~20M | ~80M |
| Layers / width | 8 / 384 | 16 / 640 |
| Query / KV heads | 6 / 2 | 10 / 5 |
| SwiGLU width | 1,024 | 1,792 |
| Vocabulary | 8,192 | 8,192 |
| Tokens | ~200M | ~1.6B |
| Estimated wall clock | ~5 h | ~3 days |

Context stages run 1K, then 2K, then 4K, holding tokens per optimizer update
constant by trading microbatch against accumulation, as the Brittain3 plans do.
bf16, activation checkpointing on, `torch.compile` on, AdamW at the fixed
validated betas and clip.

Run the pilot first and read its adherence scores before committing three days of
GPU. The pilot answers one question: do the tags steer? If `[POV: First]` does not
reliably produce first person at 20M, the fix is in the data policy above, and
finding that out costs five hours instead of three days.

## Evaluation

`scripts/evaluate/tag_adherence.py` generates samples across a held-out grid of
tag combinations, runs `story_tagger.py` over the output, and reports:

- **Per-tag adherence rate.** `Voice`, `POV`, `Tense`, `Setting`, `Cast`, and
  `Length` are scored exactly.
- **Lever independence.** Whether setting `Genre` disturbs `POV`. Tags should be
  compositional, not entangled.
- **Unconditional quality.** Validation loss and bits per byte on held-out books.
- **`Genre`, `Tone`, `Twist`.** A fixed manual rubric under `benchmarks/`.

Plus a contamination check that no validation book appears in training.

## Status

Implemented:

- `src/brittain/tags.py` — schema, render/parse, randomization policy
- `src/brittain/story_tagger.py` — deterministic extractors
- `src/brittain/corpus_story.py` — Gutenberg RDF catalog reader, fiction and
  world-texture filters, boilerplate stripping, dialogue and narrative-verb floors
- `src/brittain/tokenizer_story.py` — prose tokenizer interface and validation
- `src/brittain/data_story.py` — windowing, tag block, loss-masking packer
- `scripts/prepare/build_story_corpus.py` — two-stage corpus builder
- `configs/data/shakespeare_corpus.json` — corpus policy
- `tests/test_story_tags.py`, `test_story_corpus.py`, `test_story_data.py`

Not yet built: the trained 8K prose BPE (the interface exists, the artifact does
not), `prepare_shakespeare.py`, model and training configs, adherence evaluation,
sampler tag support.

## Running the corpus builder

Nothing in this pipeline downloads anything. Fetch the catalog first:

```
https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2
```

Stage one applies the metadata rules and writes a funnel report without reading a
single book. It is cheap, so run it on a slice while calibrating thresholds:

```bash
python3 scripts/prepare/build_story_corpus.py --catalog data/raw/gutenberg/rdf-files.tar.bz2 --catalog-only --limit 2000
```

The report gives per-rule rejection counts and the `Voice` and `Genre`
distributions across surviving books. Read it before committing to a full run:
the dialogue-density and narrative-verb floors in the config are estimates, and a
filter that is too tight starves the corpus silently.

Stage two adds a local text mirror, applies the text floors, and writes the
corpus JSONL:

```bash
python3 scripts/prepare/build_story_corpus.py --catalog data/raw/gutenberg/rdf-files.tar.bz2 --text-dir data/raw/gutenberg/text --output data/raw/brittain-shakespeare/corpus.jsonl
```

Rejection reasons are evaluated in a fixed order and a book is attributed to the
first rule it fails, so funnel counts stay comparable between runs.

`--emit-ids` writes the surviving ebook identifiers and their category, one per
line. Only 21,032 of 79,148 books survive, so fetching those specific books is
cheaper than pulling the 10 GB `txt-files.tar.zip` archive:

```bash
python3 scripts/prepare/build_story_corpus.py --catalog data/raw/gutenberg/rdf-files.tar.bz2 --catalog-only --emit-ids data/raw/brittain-shakespeare/wanted-ids.tsv
```

Fetch the text through an official mirror rather than www.gutenberg.org, which
blocks automated bulk access.

## Windowing and packing

`data_story.py` replaces two pieces of the Brittain3 data path.

It replaces `encode_document` because that function truncates a document to a
single block. Correct for a source file, wrong for a novel: it would keep the
first 1,024 tokens of Bleak House and discard the rest. Windows are cut at
chapter headings where present, paragraph boundaries otherwise, and sentence
boundaries only when a single paragraph exceeds the maximum. A window is never
cut mid-sentence.

It also replaces `pack_segments`, which derives labels purely by shifting and has
no way to exclude a span from the loss. Excluding the tag block from the loss on
most examples is deliberate, so the packer carries a per-token supervision mask
and writes `-100` for both padding and masked spans.

Two details are load-bearing:

- **`Length` is recomputed at encoding time**, from the window's own token count.
  It describes the window, and the window is only known after windowing, so a
  value carried over from the whole book would be wrong.
- **The story is trimmed to fit the block, never the tag block.** A truncated tag
  block is a malformed condition; a trimmed story is merely shorter.

Paragraph token counts are accumulated approximately for speed, but the maximum
window size is checked exactly at flush, because joining paragraphs costs a
little more than the sum of their parts — the tokenizer merges across the join.
