import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
block_size = 128
max_seq_len = 80
batch_size = 64
max_iters = 3000
eval_iters = 100
eval_interval = 500
learning_rate = 3e-4
n_embd = 384
n_layer = 8
n_head  = 8
dropout = 0.2

input_file = "all_data.txt"

# Use a set to track unique characters
unique_chars = set()

# Read the file line by line and add characters to the set
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        unique_chars.update(line)

# Sort the unique characters and calculate the vocabulary size
chars = sorted(list(unique_chars))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")

string_to_int = {ch:i for i,ch in enumerate(chars) }
int_to_string = {i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# data = torch.tensor(encode(text), dtype = torch.long)
# print(data[:100])

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        if self.tril.size(0) < T or self.tril.size(1) < T:
            raise ValueError(f"The tril tensor is too small for sequence length T={T}.")
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

        
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.head = nn.ModuleList([Head(head_size) for _ in range (num_heads) ])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.head], dim = -1)
        out = self.dropout(self.proj(out))
        return out
          

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self,x):
         return self.net(x)
        
            
        

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln_1(x+y)
        y = self.ffwd(x)
        x = self.ln_2(x+y)
        return x
        
    

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size+1,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0,  std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0,  std = 0.02)
    
    def forward(self, index, targets=None):
        B, T = index.shape
        if torch.max(index) >= vocab_size:
            raise IndexError(f"Token index out of range. Max index: {torch.max(index)}, Vocab size: {vocab_size}")
        tok_emb = self.token_embedding_table(index)
        if T > block_size:
            raise IndexError(f"Position index out of range. T: {T}, Block size: {block_size}")
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    
    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            if index.size(1) == block_size:

            # Process the sequence in chunks of block_size
                (logits, loss) = self.forward(index)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                index_next = torch.multinomial(probs, num_samples=1)
                index = index_next
            else:
                (logits, loss) = self.forward(index)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                index_next = torch.multinomial(probs, num_samples=1)
                index = torch.cat((index, index_next), dim=1)
        return index

model = GPTLanguageModel(vocab_size)

print('Model loading parameters')
with open('model-01.pkl','rb') as f:
    model = pickle.load(f)
print('loaded successfully')
m = model.to(device)



while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt),dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_token = 150)[0].tolist())
    print(f'Completion:\n{generated_chars}')
    
    

            
        
    

    