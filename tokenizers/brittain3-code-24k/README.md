# Brittain3 tokenizer

This directory contains the accepted Brittain3 24,576-token byte-level BPE.
It was trained on the versioned 2GB corpus policy in
`docs/BRITTAIN3_TOKENIZER_CORPUS.md`.

Tokenizer SHA-256:

```text
40defb2b987470b2c14dcaee234a2ff95d2c8eaddbb89037980338d6708602a8
```

The held-out 10MB evaluation passed the 3% Brittain2 code-regression gate.
Aggregate code token count increased by 1.054%. Every code-language group
passed separately; TypeScript was the largest increase at 2.447%.
Documentation used 3.729% fewer tokens, structured text was effectively equal,
and tool-protocol text used 35.071% fewer tokens. English used 5.525% more
tokens, which is accepted for this code-focused vocabulary.

See `validation.json` for the complete metrics and special-token identifiers.
Do not commit the multi-gigabyte source or evaluation corpora.
