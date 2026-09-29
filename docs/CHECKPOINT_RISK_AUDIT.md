# Research Session: Risk audit: identify every way the codebase could lose or overwrite a model checkpoint — unguarded writes, missing atomicity, resume-source overwrites, filename collisions with release artifacts, and assumptions that files exist. Rank findings by failure severity. Read-only.

## Observations
- **Observation**: [RANK 1 — CRITICAL] scripts/train/pretrain.py saves non-atomically to a file that is ALSO its only resume source, with resume hardcoded on and no CLI to redirect recovery. save_ckpt (lines 131-136) does a bare torch.save directly to out_path (e.g. checkpoints/brittain_235m.pt for the 9-day cloud_235m preset, line 60-71); the resume path reads ONLY out_path (`if resume and os.path.exists(out_path): ck = torch.load(out_path...)`, lines 114-121; `resume = True` hardcoded line 74). A crash or disk-full mid-save truncates the sole resume point; the _best.pt copy (line 199) is never read back by this script, and pretrain.py has no argparse at all (config-block only, PRESET env var), so recovery requires editing the source. Contrast: the v3 trainer's atomic_torch_save (tmp + os.replace) makes the same failure leave the previous good checkpoint intact. (Evidence: scripts/train/pretrain.py:60-74,114-121,131-136,195-199)
- **Observation**: [RANK 2 — CRITICAL] pretrain_v3.py warm-start can silently destroy the source run's resume state: --init-from loads checkpoints/X/weights.pt (lines 184-194) but nothing checks that the new run's output_dir differs from the source run's directory. output_dir comes from the training config (`args.output_dir or training["output_dir"]`, line 203) and the first eval then overwrites X/latest.pt (the ONLY file holding optimizer+data-cursor state, line 253) and X/weights.pt (line 256). Re-running a stage with its own config plus --init-from its own weights.pt — a plausible continue-the-pilot workflow — leaves the original run unresumable. Same shape for --resume from best.pt into the same dir. No guard compares output_dir against the init/resume source path anywhere in the file. (Evidence: scripts/train/pretrain_v3.py:184-194,203,247-256)
- **Observation**: [RANK 3 — HIGH] No trainer guards against --out colliding with --base or with any existing file: grep for equality guards between args.out and args.base in fim.py/code_sft.py/specialist.py returns nothing. All three load the base fully at startup then save to args.out at the first eval (fim.py save() line 208-215 called at 253; code_sft.py save at 155; specialist.py save_ckpt at 209), so `--out` pointing at the base file (or a name variant of it) overwrites the base checkpoint mid-run with a fine-tuned version — the original base is gone, and since .pt is gitignored there is no VCS recovery. specialist.py and code_sft.py make --base required (good provenance) but never validate it differs from --out. (Evidence: scripts/train/fim.py:42-43,253; scripts/train/code_sft.py:39-40,155; scripts/train/specialist.py:64-66,209 (no equality guard found by grep))
- **Observation**: [RANK 4 — HIGH] scripts/train/sft.py hardcodes its OUTPUT to a shipped release model. OUT = str(CHECKPOINT_DIR / "brittain_124m_sft.pt") (line 33) and torch.save directly to OUT every epoch (lines 102-103), non-atomic. brittain_124m_sft.pt is a 494MB release artifact documented in docs/MODELS.md and listed in checkpoints/README.md — but unlike the .pt pattern, re-running sft.py (documented as a two-command workflow, lines 17-18) rewrites it in place, losing provenance (payload has only model/cfg/epoch — no base, no val, no tokenizer field, which code_sft.py:12-14 calls out as a defect). The base brittain_124m_best.pt is read-only here, so the loss is "release file silently replaced", not "training destroyed". (Evidence: scripts/train/sft.py:17-18,33,102-103; checkpoints/README.md:6-13; docs/MODELS.md:193)
- **Observation**: [RANK 5 — MODERATE] fim.py's DEFAULT --base no longer exists under that name: ap.add_argument("--base", default=str(CHECKPOINT_DIR / "brittain_235m_best.pt")) (line 42) but the on-disk inventory shows no brittain_235m_best.pt — the finished run was renamed to brittain2_235m_weights.pt (docs/MODELS.md:571-573 states shipped checkpoints are finals, _best files kept only as crash-recovery artifacts). Running fim.py with no args produces an unguarded FileNotFoundError from torch.load (line 127). Not corruption, but it is the assumption-a-file-exists hazard the audit asked about, on a documented default invocation. (Evidence: scripts/train/fim.py:42,127; docs/MODELS.md:571-573; ls of checkpoints/ (no brittain_235m_best.pt present))
- **Observation**: [RANK 6 — MODERATE] Smoke-test default silently writes into a real run directory: pretrain_v3.py smoke_configuration() hardcodes output = resolve_project_path(args.output_dir or "runs/brittain3_smoke") (line 147) with no exists-check, and smoke mode also generates train.npz/validation.npz into that dir. On a machine that has already run the real pilot config... wait — corrected: configs/training/brittain3_49m_pilot.json writes to checkpoints/brittain3_49m_pilot, so smoke cannot collide with the pilot; but repeated smoke runs (the documented dev loop, runs/brittain3_smoke* exists in six variants) overwrite runs/brittain3_smoke's best/latest/weights in place via atomic_torch_save — dev history loss, not training loss. Atomicity makes each individual write safe; nothing makes the old smoke run recoverable. (Evidence: scripts/train/pretrain_v3.py:143-160 (smoke_configuration), runs/ tree showing brittain3_smoke variants)
- **Observation**: [RANK 7 — LOW] Every load site assumes existence: sample.py:87, chat.py:31, all evaluate scripts (bs_capabilities.py:152, brittain_script.py:125, generate_humaneval.py:260, smoke_v3, novice, compare via load_any) and the trainer resume paths (pretrain_v3.py:167, fim.py:98) call torch.load with no os.path.exists check and no try/except — a mistyped or moved path is a raw FileNotFoundError traceback. This is a usability hazard, not data loss: nothing is written. The only mitigations that exist are serving-side: serve.py catches load failures per checkpoint ([skip] log, startup loop) and sample.py/serve.py skip "backup"-named files in auto-discovery. (Evidence: scripts/inference/sample.py:87; scripts/inference/chat.py:31; scripts/train/pretrain_v3.py:167; scripts/inference/serve.py:455-475)
- **Observation**: [RANK 8 — LOW, cross-cutting amplifier] No checkpoint history exists anywhere: every trainer keeps at most latest + best (two files, both overwritten in place each eval) — no numbered/rotated checkpoints, no retention window. Combined with .gitignore line 2 `*.pt` (and line 32 `runs/`), NOTHING in the repo's VCS can recover a checkpoint once overwritten — .gitignore header says 'large binaries — keep out of git'. So the blast radius of every overwrite above is total: the previous state of an overwritten .pt is unrecoverable by any tool in this repository. Atomicity (v3 only) prevents a torn file, but never preserves history. (Evidence: .gitignore:1-4,32; save sites in scripts/train/pretrain.py:195-199, pretrain_v3.py:247-256 (latest/best/weights, all in-place, no rotation))
- **Observation**: [RANK 9 — INFO, positive findings] Things that are NOT risks, verified: (a) the v3 write path cannot tear a file — atomic_torch_save writes a .tmp sibling then os.replace (checkpoint_v3.py:71-75), so mid-write crash leaves the previous checkpoint intact; (b) no code path in src/ or scripts/ deletes checkpoints — grep for os.remove/unlink/rmtree/shutil finds only shutil.which (PATH lookup) and a data-sandbox rmtree in prepare_bs.py:486; (c) .tmp collisions between concurrent runs cannot happen because each run owns a distinct output directory; (d) serve.py/sample.py auto-discovery explicitly excludes the unloadable brittain_model_backup.pt by name, and runs/ checkpoints are not auto-touched by the server at all. (Evidence: src/brittain/checkpoint_v3.py:71-75; grep os.remove|unlink|rmtree|shutil over src/ and scripts/; scripts/inference/serve.py:303-306)

## Summary
RISK AUDIT — ways this code can lose or overwrite a checkpoint, ranked by failure severity (full evidence with file:line in the log's observations):

CRITICAL
1. pretrain.py's sole resume source is also its non-atomic save target, with resume hardcoded on and no CLI (config-block only). A crash/disk-full mid-save truncates the one file the trainer reads on restart; _best.pt is never read back; recovery requires editing source. scripts/train/pretrain.py:74,114-121,131-136,195-199.
2. pretrain_v3.py warm-start (--init-from) can silently clobber the SOURCE run's resume state: nothing checks that the new run's output_dir differs from the checkpoint's own run directory; first eval overwrites that directory's latest.pt (the only holder of optimizer + data-cursor state) and weights.pt. Also applies to --resume from best.pt into the same dir. pretrain_v3.py:184-194,203,247-256.

HIGH
3. No --out vs --base collision guard in fim.py, code_sft.py, or specialist.py: pointing --out at the base file overwrites the base checkpoint with a fine-tuned version at the first eval. Grep confirms zero equality guards.
4. sft.py hardcodes its output to shipped release model brittain_124m_sft.pt (494MB, documented as a release artifact) and rewrites it in place, non-atomically, every epoch — a re-run silently replaces a release file, and the payload (model/cfg/epoch only) carries no provenance.

MODERATE
5. fim.py's default --base (checkpoints/brittain_235m_best.pt) no longer exists on disk (finished runs are renamed to finals like brittain2_235m_weights.pt), so the documented no-args invocation dies with an unguarded FileNotFoundError.
6. Smoke mode writes into the same runs/brittain3_smoke dir every time (in-place overwrite of best/latest/weights + npz) — dev-history loss, not training loss; atomic writes keep each file whole.

LOW
7. Every load site (sample.py, chat.py, all evaluate scripts, trainer resume paths) calls torch.load with no exists-check/try-except — mistyped paths are raw tracebacks. Usability, not loss; serve.py is the only graceful loader.
8. Cross-cutting amplifier: no checkpoint rotation/history anywhere (at most latest+best, both overwritten in place) and .gitignore excludes *.pt and runs/ — once overwritten, a checkpoint is unrecoverable by any tool in the repo.

NOT RISKS (verified): v3 writes are atomic (tmp + os.replace) and cannot tear; no code path deletes checkpoints; .tmp collisions can't happen across concurrent runs (distinct output dirs); auto-discovery filters the unloadable 604M backup by name; runs/ checkpoints are never auto-touched by serving.


---

# FINAL SUMMARY — Checkpoint lifecycle, ranked risks, and the one change to make first

## How checkpointing works

Two generations of convention coexist.

**Brittain3 (current).** `scripts/train/pretrain_v3.py` writes a per-run directory
(from the training config's `output_dir`, e.g. `checkpoints/brittain3_49m_pilot/`)
containing three files, saved through `src/brittain/checkpoint_v3.py`:

- `latest.pt` — full payload (model + optimizer + scheduler + data cursors +
  RNG state + training_config), written at every eval interval; the sole
  crash-resume source.
- `best.pt` — same payload, only when validation improves.
- `weights.pt` — optimizer-stripped copy for inference/new stages.

Every write goes through `atomic_torch_save`: save to `<name>.pt.tmp`, then
`os.replace` — crash-safe. The payload is versioned by `architecture` and
`architecture_version`; loading validates both before touching weights.

**Brittain1/2 and fine-tunes (legacy).** Flat files in `checkpoints/`:
`<run>.pt` (latest, non-atomic, doubles as the resume source) plus
`<run>_best.pt` via filename `.replace(".pt", "_best.pt")`. Payloads are ad-hoc
dicts `{iter, model, optim, cfg, tokenizer, best_val, val}`; older scripts
(`sft.py`) omit even the tokenizer field.

**Loading.** Family is decided by what `torch.load` returns: dict with `cfg` +
`architecture=="brittain3"` → Brittain3; dict with `cfg` only → Brittain1/2;
bare ModuleList state_dict → BrittainScript. `src/brittain/loading.py:load_any`
is the unified branch (compare.py, novice.py); sample.py/serve.py duplicate it
inline; three eval scripts have their own copies; chat.py assumes family 1/2
unconditionally. Resume vs warm-start is a payload contract: `--resume` demands
optimizer state and `training_config` (SystemExit otherwise); `--init-from`
requires exact model-shape and tokenizer-identity match
(`validate_initialization_checkpoint`).

## Ranked risks

1. **CRITICAL — `pretrain.py`'s only resume source is its own non-atomic save
   target.** `resume = True` hardcoded, no CLI; a crash mid-save truncates the
   sole resume point; `_best.pt` is never read back. (scripts/train/pretrain.py:74,114-121,131-136,195-199)
2. **CRITICAL — `pretrain_v3.py --init-from` can silently clobber the source
   run.** No check that the new run's `output_dir` differs from the
   checkpoint's own directory; first eval overwrites `latest.pt` (the only
   holder of optimizer + data-cursor state). (pretrain_v3.py:184-194,203,247-256)
3. **HIGH — no `--out` vs `--base` collision guard** in fim.py, code_sft.py,
   specialist.py; pointing `--out` at the base overwrites the base checkpoint
   at the first eval.
4. **HIGH — `sft.py` hardcodes its output to shipped release model**
   `brittain_124m_sft.pt` (494 MB) and rewrites it in place, non-atomically,
   every epoch, with a provenance-free payload.
5. **MODERATE — `fim.py` default `--base` no longer exists** on disk
   (`brittain_235m_best.pt` was renamed to `brittain2_235m_weights.pt`); the
   documented no-args invocation dies with an unguarded `FileNotFoundError`.
6. **MODERATE — smoke mode always writes into `runs/brittain3_smoke`**,
   overwriting prior smoke results in place (dev-history loss only).
7. **LOW — every load site assumes existence**: no exists-check/try-except
   around `torch.load` outside serve.py; missing paths are raw tracebacks.
8. **AMPLIFIER — no history anywhere**: at most latest+best, both overwritten
   in place, and `.gitignore` excludes `*.pt` and `runs/`. Any overwrite above
   is unrecoverable by any tool in this repository.

## The single change I would make first

**Route all legacy trainer saves through `atomic_torch_save`, starting with
`pretrain.py`.** It is a three-line change per script (import the helper from
`brittain.checkpoint_v3`, call it instead of `torch.save`) that converts Risk 1
from "one crash during a 9-day run destroys the only resume point" to "crash
leaves the previous good checkpoint intact" — the largest severity reduction
per line changed. (It mitigates, but does not fix, Risks 3 and 4: atomicity
prevents torn files, not overwrites.)

Everything else — the `out != base` assertion (Risk 3), the
`output_dir != source` check (Risk 2), and a default-path existence check with
a helpful message (Risk 5) — follows, because each is also a few lines and each
addresses a distinct failure mode.
