"""
Re-migrates the ORIGINAL 32-context checkpoint (brittain_model_backup.pt) into
a 256-slot position embedding table WITHOUT introducing untrained random noise.

Instead of randomly initializing the new position slots (32-255), we TILE the
32 trained position embeddings cyclically: new_pos[i] = old_pos[i % 32].

Why this helps:
- The model was only ever trained to interpret positions 0-31.
- Random init for slots 32-255 means every token past position 31 gets a
  meaningless positional signal -> gibberish (what you saw).
- Tiling means position 32 gets the *same* embedding as position 0, position
  33 same as position 1, etc. The model has actually seen these vectors
  during training, so it can meaningfully use them again. Since attention is
  driven heavily by content (keys/queries), this acts like a soft "relative
  position mod 32" signal instead of a hard cliff into noise.
- This is NOT as good as truly training long-range positions, but it is a
  strict improvement over noise, and costs zero training time.

Run this once. It reads brittain_model_backup.pt (untouched original) and
writes a fresh brittain_model.pt you can use immediately with generate.py.
"""
import torch
import shutil

BACKUP_PATH = "brittain_model_backup.pt"
OUTPUT_PATH = "brittain_model.pt"
NEW_CONTEXT = 256

print(f"Loading original untouched checkpoint from '{BACKUP_PATH}'...")
checkpoint = torch.load(BACKUP_PATH, map_location="cpu")
state_dict = checkpoint['model_state_dict']

# 1. Tile the Position Embedding Table
pos_key = "position_embedding_table.weight"
old_pos_weights = state_dict[pos_key]  # Shape: [32, 1024]
old_slots, n_embd = old_pos_weights.shape

# Repeat the trained rows cyclically to fill the new, larger table
repeats = (NEW_CONTEXT + old_slots - 1) // old_slots
tiled = old_pos_weights.repeat(repeats, 1)[:NEW_CONTEXT, :]
state_dict[pos_key] = tiled
print(f"--> Tiled position table from {old_slots} trained rows -> {NEW_CONTEXT} rows (position % {old_slots}).")

# 2. Expand the Attention Mask (Tril) Buffers to match the new context
new_tril = torch.tril(torch.ones(NEW_CONTEXT, NEW_CONTEXT))
updated_masks = 0
for key in list(state_dict.keys()):
    if "sa.heads" in key and "tril" in key:
        state_dict[key] = new_tril
        updated_masks += 1
print(f"--> Expanded {updated_masks} causal attention mask matrices to {NEW_CONTEXT}x{NEW_CONTEXT}.")

checkpoint['model_state_dict'] = state_dict
torch.save(checkpoint, OUTPUT_PATH)
print(f"\n[Operation Complete] Wrote tiled checkpoint to '{OUTPUT_PATH}'.")
print("You can now try increasing EFFECTIVE_CONTEXT in generate.py above 32 (e.g. 64, 96, 128)")
print("and see how far coherence holds before quality degrades.")
