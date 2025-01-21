import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

#hyperparamters
batch_size = 32
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-2

torch.manual_seed(1337)

#reading file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

#tokenization
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda e:[stoi[s] for s in e]
decode = lambda d:''.join([itos[i] for i in d])

data = torch.tensor(encode(text), dtype = torch.long)

#split data
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

#get batch
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#single head self_attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias= False)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        weight = q @ k.transpose(-2, -1) * C**-0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim = -1)
        v = self.value(x)
        out = weight @ v
        return out


#multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out
    
#feedforward layer
class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    def forward(self,x):
        return self.network(x)

class Blocks(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        #communication
        self.sa = MultiHeadAttention(n_head, head_size)
        #computation
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

#bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.postional_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(
            Blocks(n_embd, n_head=4),
            Blocks(n_embd, n_head=4),
            Blocks(n_embd, n_head=4),
            Blocks(n_embd, n_head=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, target = None):
        B,T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.postional_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        logits = self.lm_head(x)
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for steps in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

#train
optimizer = optim.AdamW(model.parameters(), lr = learning_rate)

for iters in range(max_iters):
    if iters % eval_interval == 0:
        loss = estimate_loss()
        print(f"For step {iters}: train loss = {loss['train']:.4f} and test loss = {loss['test']:.4f}")


    #train
    xa, ya = get_batch('train')
    logits, loss = model(xa, ya)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=800)[0].tolist()))
