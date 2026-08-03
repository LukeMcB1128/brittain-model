# Human-written BrittainScript benchmark

Place human-written `.bs` programs in this directory, one self-contained program
per file. The capability evaluator measures BPB over these files and checks that
they run, but it does not use them for training.

Good submissions should:

- be written directly in BrittainScript rather than translated from Python;
- be deterministic and terminate without interactive input;
- avoid filesystem, network, GUI, and other external side effects;
- run in under five seconds on a typical laptop;
- exercise real language features rather than being a collection of literals;
- never have appeared in the specialist training corpus.

Several small programs covering different styles are more useful than one giant
file. Suggested names include `sorting.bs`, `text_processing.bs`,
`state_machine.bs`, and `tensor_math.bs`.
