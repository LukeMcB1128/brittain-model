"""
Loader for the model trained by train_50m.bs (brittain2-xs-coder:50m).

That checkpoint is a bare `nn.ModuleList.state_dict()` and its architecture is
NOT the one in model.py — BrittainScript couldn't express RoPE, SwiGLU, or weight
tying, so the .bs model uses learned position embeddings, a GELU MLP, and an
untied head. This module rebuilds that exact layout so the checkpoint loads.

Module layout written by train_50m.bs:
    0                 token embedding
    1                 position embedding
    2 + b*6 + 0..5    block b: ln1, qkv, proj, ln2, fc, fcproj
    2 + n_layer*6     final layernorm
    3 + n_layer*6     lm head

    python3 model_bs.py brittain_50m_bs.pt        # interactive completion
"""
import sys
import codecs

import torch
import torch.nn as nn
from torch.nn import functional as F

from tok_util import CodeTok


class BrittainBS(nn.Module):
    def __init__(self, vocab=32000, n_layer=6, n_head=8, n_embd=512, block=512):
        super().__init__()
        self.vocab, self.n_layer = vocab, n_layer
        self.n_head, self.n_embd, self.block = n_head, n_embd, block
        self.head_dim = n_embd // n_head
        m = [nn.Embedding(vocab, n_embd), nn.Embedding(block, n_embd)]
        for _ in range(n_layer):
            m += [nn.LayerNorm(n_embd),
                  nn.Linear(n_embd, 3 * n_embd, bias=False),
                  nn.Linear(n_embd, n_embd, bias=False),
                  nn.LayerNorm(n_embd),
                  nn.Linear(n_embd, 4 * n_embd, bias=False),
                  nn.Linear(4 * n_embd, n_embd, bias=False)]
        m += [nn.LayerNorm(n_embd), nn.Linear(n_embd, vocab, bias=False)]
        self.mods = nn.ModuleList(m)

    def forward(self, idx):
        B, T = idx.shape
        dev = idx.device
        mask = torch.ones(T, T, device=dev).tril().bool()
        x = self.mods[0](idx) + self.mods[1](torch.arange(T, device=dev))
        for b in range(self.n_layer):
            base = 2 + b * 6
            qkv = self.mods[base + 1](self.mods[base](x))
            q, k, v = qkv.chunk(3, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v, mask)
            att = att.transpose(1, 2).contiguous().view(B, T, self.n_embd)
            x = x + self.mods[base + 2](att)
            h = F.gelu(self.mods[base + 4](self.mods[base + 3](x)))
            x = x + self.mods[base + 5](h)
        x = self.mods[2 + self.n_layer * 6](x)
        return self.mods[3 + self.n_layer * 6](x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.2, top_p=0.95):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block:])[:, -1, :] / max(temperature, 1e-5)
            if top_p is not None:
                sl, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                drop = cum > top_p
                drop[..., 1:] = drop[..., :-1].clone()
                drop[..., 0] = False
                for b in range(logits.size(0)):
                    logits[b, si[b, drop[b]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def load(path, device):
    model = BrittainBS().to(device)
    sd = torch.load(path, map_location=device)
    # train_50m.bs saves the ModuleList's own state_dict, so keys are "0.weight",
    # not "mods.0.weight" — load into the ModuleList directly.
    if not any(k.startswith("mods.") for k in sd):
        model.mods.load_state_dict(sd)
    else:
        model.load_state_dict(sd)
    model.eval()
    return model, CodeTok()


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "brittain_50m_bs.pt"
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    model, enc = load(ckpt, device)
    print(f"Loaded {ckpt} ({model.num_params():,} params) — trained by BrittainScript")
    print("-" * 60)
    while True:
        try:
            prompt = input("\nPrompt: ")
            if not prompt.strip():
                continue
            ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
            utf8 = codecs.getincrementaldecoder("utf-8")("replace")
            print(prompt, end="", flush=True)
            for _ in range(200):
                ids = model.generate(ids, 1)
                nxt = ids[0, -1].item()
                if nxt == enc.eot:
                    break
                print(utf8.decode(enc.token_bytes(nxt)), end="", flush=True)
            print("\n" + "-" * 40)
        except KeyboardInterrupt:
            print("\nbye")
            break
