import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import argparse

# --- 1. CONFIGURATION ---
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='auto')
args = parser.parse_args()

if args.device == 'auto':
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = args.device

print(f"--> Loading Prometheus (Memory Enabled) on {device}...")

# --- 2. LOAD VOCABULARY ---
try:
    with open('vocab.pkl', 'rb') as f:
        chars = pickle.load(f)
except FileNotFoundError:
    print("Error: vocab.pkl not found. Run train.py first.")
    exit()

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# --- 3. MODEL ARCHITECTURE (Must match train.py) ---
n_embd = 64
n_head = 4
n_layer = 4
block_size = 64 # This is the "Context Window" size
dropout = 0.0
vocab_size = len(chars)

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
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens so we don't overflow context
            idx_cond = idx[:, -block_size:] 
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # focus only on the last time step
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop if we generate a newline (simple way to stop the AI from rambling)
            if idx_next.item() == stoi.get('\n', -1):
                break
                
        return idx

# --- 4. LOAD & RUN ---
model = BigramLanguageModel()
model.load_state_dict(torch.load('prometheus.pth', map_location=device))
model.to(device)
model.eval()

print("\n" + "="*40)
print("PROMETHEUS (THREADED MODE)")
print("The AI now remembers previous turns.")
print("Type 'reset' to clear memory.")
print("="*40 + "\n")

# Start with an empty context (or a specialized start token if you had one)
# We initialize with a newline character to start fresh
history = torch.zeros((1, 0), dtype=torch.long, device=device)

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit': break
    if user_input.lower() == 'reset':
        history = torch.zeros((1, 0), dtype=torch.long, device=device)
        print("--> Memory cleared.")
        continue

    # 1. Add User Input to History
    # We add a newline before the user input to separate it from previous turn
    user_ids = encode("\nUser: " + user_input + "\nPrometheus:")
    user_tensor = torch.tensor([user_ids], dtype=torch.long, device=device)
    
    # Append to existing history
    history = torch.cat((history, user_tensor), dim=1)

    # 2. Generate Response
    # The generate function now handles the appending internally
    # We ask it to generate up to 100 new tokens
    history_with_response = model.generate(history, max_new_tokens=100)
    
    # 3. Extract ONLY the new part (The AI's response)
    new_tokens = history_with_response[0].tolist()[history.shape[1]:]
    response_text = decode(new_tokens)
    
    print(f"Prometheus:{response_text}")
    
    # 4. Update History for next turn
    # Ideally, we keep the history growing until it hits block_size limit
    history = history_with_response
    
    # If history is too long, we might need to trim it safely later, 
    # but for now, the model.generate handles the cropping (idx_cond)