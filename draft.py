'''input and context'''
# x= train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the output is {target}")

'''Mathematical trick in self-attention'''
# torch.manual_seed(1337)
# B,T,C = 4,8,2
# x = torch.randn((B,T,C))
# x.shape

'''version 1:averaging past context using for loops: weakest form of aggregation (Tokens talking to prev tokens in simple words)'''
#initialize a bag-of-words tensor with zeros 
# xbow = torch.zeros((B,T,C))
# for b in range(B): #batch-wise
#     for t in range(T): #Time/Blockwise
#         xprev = x[b, :t+1] #say b = 0 and t = 2 then xprev = first 3 tokens of first batch (i.e. previous tokens including the current token)
#         xbow[b,t] = torch.mean(xprev, dim = 0)

'''version 2: The trick in self attention :: matrix multiply as weighted aggregation'''
# weight = torch.tril(torch.ones((T,T)))
# weight = weight/weight.sum(dim = 1, keepdim = True)
# xbow2 = weight @ x #weighted sum:::: weight(B,T,T) and x(B,T,C) => xbow2 (B,T,C)


# # #check version1 and version2 xbows
# torch.allclose(xbow2, xbow3)

'''version 3: adding softmax'''
# tril = torch.tril(torch.ones((T,T)))
# weight = torch.zeros((T,T))
# weight = weight.masked_fill(tril == 0, float('-inf'))
# weight = F.softmax(weight, dim = -1)
# xbow3 = weight @ x
# torch.allclose(xbow2, xbow3)


'''version 4: Self Attention'''
# torch.manual_seed(1337)
# B,T,C = 4, 8 ,32
# x = torch.randn(B,T,C)
# #single head performing self attention
# head_size = 16
# query = nn.Linear(C, head_size, bias = False)
# key = nn.Linear(C, head_size, bias = False)
# value = nn.Linear(C, head_size, bias = False)

# q = query(x) #(B, T, 16)
# k = key(x) #(B,T,16)

# weight = q @ k.transpose(-2,-1)  # (B, T, 16) * (B, 16, T) => (B,T,T)

# tril = torch.tril(torch.ones(T,T))
# weight = weight.masked_fill(tril == 0, float('-inf'))
# weight = F.softmax(weight,dim = -1)
# v = value(x) #(B, T, 16)

# out = weight @ v

# out.shape

'''Batch Normalization'''
# class BatchNormId:
#     def __init__(self, dim, eps = 1e-5, momentum = 0.1)
#     self.eps = eps
#     self.momentum = momentum
#     self.training = True
#     #parameters (Trained with backpropagation)
#     self.gamma = torch.ones(dim)
#     self.beta = torch.zeros(dim)
#     #buffers (trained with a running 'momentum update')
#     self.running_mean = torch.zeros(dim)
#     self.running_var = torch.ones(dim)

#     def __call__(self,x):
#         #calculate forward pass
#         if self.training:
#             xmean = x.mean(0, keepdim = True) #batch mean
#             xvar = x.var(0, keepdim = True)
#         else:
#             xmean = self.running_mean
#             xvar = self.running_var
#         xhat = (x-xmean)/torch.sqrt(xvar + self.eps) #normalize to unit variance
#         self.out = self.gamma * xhat + self.beta
#         #update the buffers
#         if self.training
#             with torch.no_grad():
#                 self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
#                 self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
#         return self.out

#     def parameters(self):
#         return [self.gamma, self.beta]

# torch.manual_seed(1337)
# module = BatchNormId(100)
# x = torch.randn(32, 100)
# x = module(x)
# x.shape

'''Layernormalization'''
# class LayerNormId:
#     def __init__(self, dim, eps = 1e-5):
#         self.eps = eps
#         #parameters (Trained with backpropagation)
#         self.gamma = torch.ones(dim)
#         self.beta = torch.zeros(dim)
        

#     def __call__(self,x):
#         #calculate forward pass
        
#         xmean = x.mean(1, keepdim = True) #batch mean
#         xvar = x.var(1, keepdim = True)
   
#         xhat = (x-xmean)/torch.sqrt(xvar + self.eps) #normalize to unit variance
#         self.out = self.gamma * xhat + self.beta
        
#         return self.out

#     def parameters(self):
#         return [self.gamma, self.beta]

# torch.manual_seed(1337)
# module = BatchNormId(100)
# x = torch.randn(32, 100)
# x = module(x)
# x.shape