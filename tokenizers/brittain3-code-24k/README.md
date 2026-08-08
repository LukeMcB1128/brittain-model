# Brittain3 tokenizer

This directory will contain the Brittain3 24,576-token byte-level BPE. The
tokenizer is not committed until it passes the validation gates in
`scripts/prepare/train_tokenizer_v3.py`.

Training the real tokenizer reads a prepared local corpus and can take time. It
is not part of the local smoke test.
