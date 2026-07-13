import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle
import numpy as np

# 1. Hardware Selection: Force Apple Silicon GPU Acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- Running training on device: {device} ---")

# 2. Hyperparameters
batch_size = 16       # How many independent sequences to process in parallel
block_size = 256       # Maximum context length (how many characters to look back)
max_iters = 1000     # Number of training loops (~2-3 epochs over the ~12M token dataset)
eval_interval = 300   # How often to check the loss and print text generation
learning_rate = 3e-4  # Speed of weight adjustments (lowered for stability at larger model size)
n_embd = 1024          # Embedding dimensions
n_head = 8             # Number of self-attention heads (head_size = 1024/8 = 128)
n_layer = 16           # Number of sequential transformer blocks
num_experts = 4        # MoE experts (memory scales with num_experts, not compute!)

checkpoint_path = "brittain_model.pt"
data_dir = "./data"

meta_path = os.path.join(data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    raise FileNotFoundError("Missing dataset binaries. Please run prepare.py first!")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']

# Re-declare mapping lambdas matching the new vocabulary
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Memory-map the flat binary files straight off your SSD
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split='train'):
    """ High-performance disk-streaming batch loader """
    data = train_data if split == 'train' else val_data
    
    # Select random entry points across the flat file array
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Gather chunks and explicitly cast array memories to standard integer tensors
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    
    # Instantly stream data to your M3 Max GPU unified memory space
    x, y = x.to(device), y.to(device)
    return x, y

# 4. Building the Transformer Architecture

class AttentionHead(nn.Module):
    """ A single head of causal self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register causal mask to prevent looking into the future
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Calculate attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C**-0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        
        # Perform the weighted aggregation of values
        v = self.value(x) 
        return wei @ v 

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)
    
class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)
    
class MixtureOfExperts(nn.Module):
    def __init__(self, n_embd, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.router = nn.Linear(n_embd, num_experts)

    def forward(self, x):
        B, T, C = x.shape
        flat_x = x.view(-1, C)

        router_logits = self.router(flat_x)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Top-1 Routing: select the single best expert for each token
        weights, selected_experts = torch.topk(routing_weights, k=1, dim=-1)

        out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            # Identify which tokens belong to this specific expert
            token_indices = (selected_experts.squeeze(-1) == i).nonzero().squeeze(-1)
            if token_indices.numel() > 0:
                expert_out = expert(flat_x[token_indices])
                out[token_indices] = weights[token_indices] * expert_out
                
        return out.view(B, T, C)

class TransformerBlock(nn.Module):
    """ Combines communication (Attention) and computation (FeedForward) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = MixtureOfExperts(n_embd, num_experts=num_experts)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Using residual connections around layer normalization
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Brittain(nn.Module):
    """ The overarching Language Model assembly """
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """ Autoregressively predict the next character """
        for _ in range(max_new_tokens):
            # Crop current context window if it exceeds our block size
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # Pull the logits of the final character step
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            # Sample next token based on probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

# 5. Initialization and Training Loop
model = Brittain().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_iter = 0
if os.path.exists(checkpoint_path):
    print(f"--> Found saved weights! Loading checkpoint: '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
   # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iteration'] + 1
    print(f"--> Successfully resumed model. Starting from iteration: {start_iter}")
else:
    print("--> No prior checkpoint found. Initializing brand new model weights.")
    print("--- Initial model capabilities (Untrained Random Output): ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
    print("-" * 50)

print(f"Starting training session from iteration {start_iter} to {max_iters + start_iter}...")
for iteration in range(start_iter, max_iters + start_iter):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iteration % eval_interval == 0 and iteration > start_iter:
        print(f"Iteration {iteration:4d} | Current Loss: {loss.item():.4f}")
        print("Generated Text Snippet:")
        sample_context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_output = model.generate(sample_context, max_new_tokens=80)[0].tolist()
        print(decode(generated_output).replace('\n', ' '))
        print("-" * 40)

        # Save a checkpoint periodically so an overnight crash doesn't lose progress
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"--> Periodic checkpoint saved at iteration {iteration}.")

# 8. Checkpoint Saving Logic
print("Training run finished. Packaging and saving model weights...")
torch.save({
    'iteration': max_iters + start_iter,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, checkpoint_path)
print(f"--> Checkpoint safely written to '{checkpoint_path}'!")
print(f"Total parameter count: {sum(p.numel() for p in model.parameters()):,}")