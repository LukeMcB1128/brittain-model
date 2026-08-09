# Data configurations

Future corpus-mixture and preprocessing configurations belong here. Store paths,
mixture weights, seeds, and provenance—not downloaded corpus contents.

Brittain3 prepared JSONL rows use the fields `repository`, `path`, `text`,
`source`, and `is_code`. Repository identity controls the train/validation split.

`brittain3_tokenizer_corpus.json` is the separate tokenizer-corpus policy. It
defines byte quotas, language shares, license rules, filters, local sources, and
remote sources. See `docs/BRITTAIN3_TOKENIZER_CORPUS.md` before remote use.
