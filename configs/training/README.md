# Training configurations

Run configurations belong here: schedule, batch size, token budget, checkpoint
destination, and resume behavior.

Brittain3 configurations use the `brittain3-training-v1` format. Context stages
keep the effective token batch constant by reducing the microbatch when context
increases.

Use `brittain3_49m_curriculum_probe.json` for the 10M-token continuation probe.
Run it before `brittain3_49m_curriculum.json`, which is the conditional
100M-token continuation.

The 49M and 181M commands are training runs, not software smoke tests. See
`docs/BRITTAIN3_NEXT_PHASE.md` for the required data and evaluation commands.
