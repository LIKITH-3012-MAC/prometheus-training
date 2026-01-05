import random
import string

# --- 1. CONFIGURATION ---
filename = "data.txt"
num_math_examples = 10000
num_chat_repeats = 500

print(f"--> Generating {filename}...")

# --- 2. GENERATE MATH DATA ---
math_data = []
ops = ['+', '-', '*']
for _ in range(num_math_examples):
    a = random.randint(0, 100)
    b = random.randint(0, 100)
    op = random.choice(ops)
    
    if op == '+': res = a + b
    elif op == '-': res = a - b
    elif op == '*': res = a * b
    
    # We use format without special symbols first to be safe, 
    # but the "Character Soup" below will fix the vocab issue.
    math_data.append(f"User: Calculate {a}{op}{b}\nPrometheus: The answer is {res}\n")

# --- 3. GENERATE CONVERSATION DATA ---
conversations = [
    "User: Hi\nPrometheus: Hello! I am Prometheus AI. How can I help you?\n",
    "User: Who are you?\nPrometheus: I am Prometheus, an AI developed by Likith Naidu.\n",
    "User: What is your purpose?\nPrometheus: I am designed to assist with coding, math, and general questions.\n",
    "User: How are you?\nPrometheus: I am functioning perfectly on your Mac's neural engine.\n",
    "User: bye\nPrometheus: Goodbye! Have a great day.\n",
]

# --- 4. THE MAGIC FIX: CHARACTER SOUP ---
# This ensures every printable character is in the vocabulary.
# We repeat it a few times so the model doesn't think they are "rare" errors.
char_soup = string.printable # Contains 0-9, a-z, A-Z, and !@#$%^&*()_+={}[]|\:;"'<>,.?/
soup_lines = [f"System: Learning symbols {char_soup}\n" for _ in range(100)]

# --- 5. COMBINE & SAVE ---
with open(filename, 'w', encoding='utf-8') as f:
    # 1. Write the Character Soup (Crucial for Vocab)
    for line in soup_lines:
        f.write(line)
        
    # 2. Write Math Data
    for line in math_data:
        f.write(line + "\n")
    
    # 3. Write Conversation Data
    for _ in range(num_chat_repeats):
        for line in conversations:
            f.write(line + "\n")

print(f"--> Successfully created {filename} with FULL vocabulary coverage!")