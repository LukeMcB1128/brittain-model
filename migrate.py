import torch
import os

checkpoint_path = "brittain_model.pt"

print("Opening checkpoint for matrix adjustments...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint['model_state_dict']

# 1. Migrate the Position Embedding Table
pos_key = "position_embedding_table.weight"
old_pos_weights = state_dict[pos_key] # Shape: [32, 1024]
old_slots, n_embd = old_pos_weights.shape

# Initialize a new blank canvas for 256 context slots
new_pos_weights = torch.zeros((256, n_embd))
# Apply standard normal initialization to the new rows so gradients flow smoothly
new_pos_weights.normal_(mean=0.0, std=0.02)

# Graft your 8-hour structural memory directly into the first 32 slots
new_pos_weights[:32, :] = old_pos_weights
state_dict[pos_key] = new_pos_weights
print(f"--> Successfully extended position table from {old_slots} to 256 rows.")

# 2. Re-scale the Attention Mask (Tril) Buffers
new_tril = torch.tril(torch.ones(256, 256))
updated_masks = 0

for key in list(state_dict.keys()):
    if "sa.heads" in key and "tril" in key:
        state_dict[key] = new_tril
        updated_masks += 1

print(f"--> Successfully expanded {updated_masks} causal attention mask matrices to 256x256.")

# Save the updated architecture maps directly back to the file
checkpoint['model_state_dict'] = state_dict
torch.save(checkpoint, checkpoint_path)
print("\n[Operation Complete] Checkpoint mutated successfully! You are ready to run.")