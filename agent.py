import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import argparse
import re

# --- 1. SETUP ---
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='auto')
args = parser.parse_args()
device = 'mps' if (args.device=='auto' and torch.backends.mps.is_available()) else 'cpu'

# --- 2. LOAD BRAIN ---
try:
    with open('vocab.pkl', 'rb') as f: chars = pickle.load(f)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
except:
    print("Error: Run train.py first.")
    exit()

# Model Config (Must match train.py)
n_embd = 64; n_head = 4; n_layer = 4; block_size = 64
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x): return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(len(chars), n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, len(chars))
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x), None
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == stoi.get('\n', -1): break
        return idx

model = BigramLanguageModel()
model.load_state_dict(torch.load('prometheus.pth', map_location=device))
model.to(device)
model.eval()

# --- 3. THE AGENT LOGIC (Function Calling) ---
def solve_math(query):
    # Regex to find simple math (e.g., "2+2", "45*10")
    math_pattern = r'(\d+[\+\-\*\/]\d+)'
    match = re.search(math_pattern, query)
    if match:
        expression = match.group(1)
        try:
            # DANGEROUS IN PROD, OK FOR LOCAL: Use Python's eval()
            result = eval(expression)
            return f"The answer is {result} (Calculated by Tool)"
        except:
            return None
    return None

print("\n" + "="*40)
print("PROMETHEUS AGENT (WITH TOOLS)")
print("I can now solve math correctly.")
print("="*40 + "\n")

history = torch.zeros((1, 0), dtype=torch.long, device=device)

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit': break

    # STEP 1: CHECK TOOLS
    # Before asking the LLM, we check if we can solve it programmatically
    tool_result = solve_math(user_input)

    if tool_result:
        # If tool solves it, print that and DO NOT run the LLM (or feed it back)
        print(f"Prometheus: {tool_result}")
        
        # Optional: Add this interaction to memory so LLM remembers it happened
        memory_str = f"\nUser: {user_input}\nPrometheus: {tool_result}"
        ctx = torch.tensor([encode(memory_str)], dtype=torch.long, device=device)
        history = torch.cat((history, ctx), dim=1)
        continue

    # STEP 2: IF NO TOOL, USE LLM (The "Brain")
    # Add User Input to History
    user_ids = encode("\nUser: " + user_input + "\nPrometheus:")
    user_tensor = torch.tensor([user_ids], dtype=torch.long, device=device)
    history = torch.cat((history, user_tensor), dim=1)

    # Generate
    history_with_response = model.generate(history, max_new_tokens=100)
    
    # Extract response
    new_tokens = history_with_response[0].tolist()[history.shape[1]:]
    response_text = decode(new_tokens)
    
    print(f"Prometheus:{response_text}")
    history = history_with_response