# Training configurations

Run configurations belong here: schedule, batch size, token budget, checkpoint
destination, and resume behavior.

Brittain3 configurations use the `brittain3-training-v1` format. Context stages
keep the effective token batch constant by reducing the microbatch when context
increases.

The 49M and 181M commands are long training runs. Do not start them as smoke
tests. See `docs/BRITTAIN3.md` for safe local commands.
