import os
import pickle
import numpy as np

# --- Configuration ---
# Point this to any folder on your Mac you want to train on
TARGET_DIR = os.path.expanduser("~/Downloads/Coding")  # e.g., your code workspace
OUTPUT_DIR = "./data"
ALLOWED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.java', '.swift', '.txt', '.md', '.json'}
IGNORED_DIRS = {'node_modules', '.git', '__pycache__', 'dist', 'build', '.idea', '.vscode'}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Scanning {TARGET_DIR} for training data...")

# 1. Gather all unique characters first to build a robust vocabulary
all_text_chunks = []
unique_chars = set()

for root, dirs, files in os.walk(TARGET_DIR):
    # Prune ignored directories in-place so os.walk skips descending into them
    dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
    
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in ALLOWED_EXTENSIONS:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        unique_chars.update(content)
                        all_text_chunks.append(content)
            except Exception:
                # Safely bypass locked system files or unreadable formats
                continue

# Add a fallback unique token for rare anomalies if necessary
chars = sorted(list(unique_chars))
vocab_size = len(chars)
print(f"Completed scan. Total files found: {len(all_text_chunks)}")
print(f"Unique vocabulary characters discovered: {vocab_size}")

# 2. Build mapping dictionaries
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Save metadata so your main model script can decode outputs later
meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(OUTPUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# 3. Tokenize all text into integers and write directly to disk as uint16
# Splitting dataset: 90% training, 10% validation
all_text = "\n\n".join(all_text_chunks)
n = len(all_text)
train_text = all_text[:int(n*0.9)]
val_text = all_text[int(n*0.9):]

print("Encoding text into integer tokens...")
train_ids = [stoi[c] for c in train_text]
val_ids = [stoi[c] for c in val_text]

print(f"Train set has {len(train_ids):,} tokens.")
print(f"Val set has {len(val_ids):,} tokens.")

# Convert to numpy arrays (uint16 supports vocab sizes up to 65,535)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(OUTPUT_DIR, 'train.bin'))
val_ids.tofile(os.path.join(OUTPUT_DIR, 'val.bin'))
print("Successfully generated binary datasets inside ./data/ folder!")