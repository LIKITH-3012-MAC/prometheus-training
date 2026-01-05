import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import argparse

# --- 1. SETUP ---
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='auto')
args = parser.parse_args()

# Detect Device
if args.device == 'auto':
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = args.device

print(f"--> Loading Prometheus AI on {device}...")

# --- 2. LOAD VOCABULARY ---
try:
    with open('vocab.pkl', 'rb') as f:
        chars = pickle.load(f)
except FileNotFoundError:
    print("Error: vocab.pkl not found. Did you run train.py first?")
    exit()

vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- 3. DEFINE MODEL ARCHITECTURE (Must match train.py exactly) ---
# We hardcode the config here for simplicity. 
# If you changed these in train.py, change them here too!
n_embd = 64
n_head = 4
n_layer = 4
block_size = 64
dropout = 0.0

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
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
        return logits, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 4. LOAD WEIGHTS ---
model = BigramLanguageModel()
try:
    model.load_state_dict(torch.load('prometheus.pth', map_location=device))
    print("--> Model weights loaded successfully!")
except FileNotFoundError:
    print("Error: prometheus.pth not found. Run train.py first.")
    exit()

model.to(device)
model.eval() # Switch to evaluation mode

# --- 5. CHAT LOOP ---
print("\n" + "="*40)
print("PROMETHEUS AI (CLI VERSION)")
print("Type 'exit' to quit.")
print("="*40 + "\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    
    if not user_input:
        continue

    # Encode input
    try:
        context = torch.tensor([encode(user_input)], dtype=torch.long, device=device)
    except KeyError:
        print("Error: You used a character the model has never seen in training.")
        continue

    # Generate
    print("Prometheus: ", end='', flush=True)
    
    # We generate 200 tokens. You can adjust this.
    generated_indices = model.generate(context, max_new_tokens=200)
    
    # Decode and print (skip the original prompt)
    response = decode(generated_indices[0].tolist())
    print(response[len(user_input):]) # Only print what's new
    print("-" * 20)