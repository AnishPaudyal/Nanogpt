import torch
import torch.nn as nn

#hyperparameters
batch_size = 32 
block_size = 8
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
    x = torch.stack([data[i:i+block_size] for i in randx]) #(B*T)
    y = torch.stack([data[i+1:i+block_size+1] for i in randx]) #(B*T)
    x,y = x.to(device), y.to(device)
    return x,y

xa, ya = get_batch('train')
print(xa.shape)
print(xa)
print(ya.shape)






