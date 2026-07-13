import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle

# 1. Setup Environment
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_DIR = "./data"
checkpoint_path = "brittain_model.pt"

# 2. Load Metadata Vocabulary
with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# 3. Hyperparameters (Must perfectly match your training configuration)
block_size = 256       
n_embd = 1024          
n_head = 8           
n_layer = 16           
num_experts = 4       

# The model was ORIGINALLY trained with block_size=32. If you ran
# migrate_tile.py, positions 32-255 are TILED copies of the trained 0-31
# embeddings (position % 32) instead of random noise. IMPORTANT: tiling can
# still confuse attention once T exceeds 32, because two tokens that are
# exactly 32 apart now get IDENTICAL positional signals, which can cause the
# model to misattend (this is worse than it sounds - it's not "no signal",
# it's "actively wrong signal"). Start at 32 to confirm the baseline still
# works, then increase in small steps (40, 48, 64...) and listen for where
# it breaks down - it may be much lower than 128.
EFFECTIVE_CONTEXT = 128

# 4. Re-declare Architecture Layout (So PyTorch can map the weights accurately)
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        wei = self.query(x) @ self.key(x).transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, n_embd, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.router = nn.Linear(n_embd, num_experts)
    def forward(self, x):
        B, T, C = x.shape
        flat_x = x.view(-1, C)
        router_logits = self.router(flat_x)
        weights, selected_experts = torch.topk(F.softmax(router_logits, dim=-1), k=1, dim=-1)
        out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            token_indices = (selected_experts.squeeze(-1) == i).nonzero().squeeze(-1)
            if token_indices.numel() > 0:
                out[token_indices] = weights[token_indices] * expert(flat_x[token_indices])
        return out.view(B, T, C)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = MixtureOfExperts(n_embd, num_experts=num_experts)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Brittain(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        return self.lm_head(self.ln_f(self.blocks(x)))
    
    def generate_stream(self, idx, max_new_tokens, temperature = 0.3):
        """ Generates tokens one by one to create a live streaming terminal effect """
        for _ in range(max_new_tokens):
            # IMPORTANT: The model was originally trained with block_size=32.
            # migrate.py only *randomly initialized* position embeddings for
            # slots 32-255 (never trained), so using more than 32 tokens of
            # context causes the model to hit untrained positional embeddings
            # and produce incoherent output. Keep the sliding window at the
            # originally trained length (EFFECTIVE_CONTEXT) to stay coherent.
            idx_cond = idx[:, -EFFECTIVE_CONTEXT:]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next.item()

# 5. Initialize and Load Model State
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Missing weight map file: {checkpoint_path}")

print("Loading 604M Parameter Weights...")
model = Brittain().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set model to evaluation flag
print("Weights loaded successfully! System online.")
print("-" * 60)

# 6. Interactive Prompt Loop
while True:
    try:
        user_prompt = input("\nEnter prompt starting seed (or Ctrl+C to exit): ")
        if not user_prompt:
            continue
            
        print("\n--> Output Generation Stream:")
        # Print your prompt prefix out first to maintain context lookups
        print(user_prompt, end="", flush=True)
        
        # Convert prompt to numbers and load to M3 Max GPU
        context_tokens = torch.tensor([encode(user_prompt)], dtype=torch.long, device=device)
        
        # Stream the completion characters out in real-time
        with torch.no_grad():
            for next_token in model.generate_stream(context_tokens, max_new_tokens=500, temperature=0.3):
                print(decode([next_token]), end="", flush=True)
        print("\n" + "-" * 40)
        
    except KeyboardInterrupt:
        print("\nShutting down stream interface.")
        break