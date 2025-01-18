import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 #No of independent sequences processed in parallel (B)
block_size = 8  #sequence lenght/time (T)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#reading the file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#extracting the unique characters from the file
chars = sorted(list(set(text)))
vocab_size = len(chars) #channel_size

#tokenization

#1. mapping the string to int and vice versa
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

#2. encode and decode
encode = lambda e:[stoi[s] for s in e]
decode = lambda d:''.join([itos[i] for i in d])

data = torch.tensor(encode(text), dtype = torch.long)

#spliting the data into train and test
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

#get diff chunks of data of particular sequence length
def get_batch(split):
    data = train_data if split == 'train' else test_data
    randx = torch.randint(len(data)- block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in randx]) #(B,T)
    y = torch.stack([data[i+1:i+block_size+1] for i in randx]) #(B,T)
    x,y = x.to(device), y.to(device)
    return x,y

#creating a simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, target = None):
        logits = self.token_embedding_table(idx) #(B,T,C) => raw, unnormalized scores 
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            #reshaping logits as cross entropy is expected N*C (N= number of prob distribution, C = Channel size)
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target) # -ve log likelihood (-log(P))
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for steps in range(max_new_tokens):

            #get predictions/logits
            logits, loss = model(idx)
            
            #focus only the last time step/ last block of each batch
            logits = logits[:,-1,:]

            #softmax
            probs = F.softmax(logits, dim=-1)

            #sampling new token
            idx_next = torch.multinomial(probs, num_samples=1)

            #appending the generated token
            idx = torch.cat((idx, idx_next), dim = 1)
        
        return idx



xa, ya = get_batch('train')
model = BigramLanguageModel()
m = model.to(device)
logits,loss = model(xa, ya)
print(logits.shape)
print(loss.item())

#generating
context = torch.zeros((1,1), dtype= torch.long) #initial token
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))







