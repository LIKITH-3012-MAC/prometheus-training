import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import os

# --- 1. CLI ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Train a Nano-GPT from scratch on custom data.")
parser.add_argument('--data', type=str, default='data.txt', help='Path to your training text file')
parser.add_argument('--batch_size', type=int, default=32, help='How many independent sequences to process in parallel')
parser.add_argument('--block_size', type=int, default=64, help='Maximum context length (time steps) for prediction')
parser.add_argument('--max_iters', type=int, default=1000, help='Number of training iterations')
parser.add_argument('--eval_interval', type=int, default=100, help='How often to print loss stats')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for AdamW optimizer')
parser.add_argument('--device', type=str, default='auto', help='Device to run on: cpu, cuda, or mps (for Mac)')
parser.add_argument('--n_embd', type=int, default=64, help='Embedding dimension size')
parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer blocks')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')

args = parser.parse_args()

# --- 2. DEVICE SETUP ---
if args.device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps' # This uses your Mac's GPU
    else:
        device = 'cpu'
else:
    device = args.device

print(f"--> Using device: {device}")

# --- 3. DATA PROCESSING ---
torch.manual_seed(1337)

with open(args.data, 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level tokenizer (Simple & Fast)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"--> Vocab size: {vocab_size} unique characters")

# Train/Val Split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - args.block_size, (args.batch_size,))
    x = torch.stack([data_source[i:i+args.block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+args.block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 4. THE TRANSFORMER ARCHITECTURE ---

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(args.n_embd, head_size, bias=False)
        self.query = nn.Linear(args.n_embd, head_size, bias=False)
        self.value = nn.Linear(args.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(args.block_size, args.block_size)))
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,16)
        q = self.query(x) # (B,T,16)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(args.n_embd, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, args.n_embd)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embd)
        self.blocks = nn.Sequential(*[Block(args.n_embd, args.n_head) for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -args.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 5. INITIALIZATION & TRAINING ---
model = BigramLanguageModel()
m = model.to(device)

print(f"--> Model Parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

print("--> Starting Training...")
for iter in range(args.max_iters):
    if iter % args.eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- 6. GENERATION TEST ---
print("\n--> Training Complete. Generating Text based on your data:\n")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# --- ADD THIS AT THE VERY BOTTOM OF train.py ---
import pickle

print("--> Saving model and vocabulary...")
# 1. Save the Model Weights
torch.save(model.state_dict(), 'prometheus.pth')

# 2. Save the Vocabulary (So chat.py knows the letters)
with open('vocab.pkl', 'wb') as f:
    pickle.dump(chars, f)

print("--> Saved to 'prometheus.pth' and 'vocab.pkl'")