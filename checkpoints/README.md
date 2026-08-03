# Checkpoints

Local model checkpoints live here. Weight formats such as `.pt`, `.ckpt`, and
`.safetensors` are ignored by Git, so the four release weights must be downloaded
from the matching GitHub release.

| Display name | File | Tokenizer | Context |
|---|---|---|---:|
| `brittain2-xs-coder:50m-bs` | `brittain2_50m_bs.pt` | `tokenizer.json` | 512 |
| `brittain2-xs-brittainscript-specialist:50m-bs` | `brittain2_xs_bs_mixed.pt` | `tokenizer.json` | 512 |
| `brittain2-coder:235m-base-1k` | `brittain2_235m_weights.pt` | `tokenizer.json` | 1,024 |
| `brittain2-coder:235m-fim-1k` | `brittain2_235m_fim.pt` | `tokenizer_fim.json` | 1,024 |

The 50M files are bare BrittainScript `ModuleList` state dictionaries. The 235M
files include their architecture and tokenizer metadata. Use
`scripts/inference/sample.py` for either format; it detects the difference.

SHA-256 hashes are recorded in
[`brittain2-first-models-notes.md`](../brittain2-first-models-notes.md#downloads-and-checksums).
