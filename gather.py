import torch
import time

#! 对比dim=1的选择
# x = torch.arange(100 * 8).view(2, 4, 100)
# idx = torch.tensor([[2,3],
#         [1,0]],dtype=torch.long)

# y = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(2,-1,100))
# print(y)
# a = time.time()
# for i in range(100000):
#     x = torch.arange(1000 * 8).view(2, 4, 1000)
#     idx = torch.tensor([[2,3],
#             [1,0]],dtype=torch.long)

#     y = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(2,-1,100))
# b = time.time()
# print(b-a)

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)

        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError
        
# a = time.time()
# for i in range(100000):
#     x = torch.arange(1000 * 8).view(2, 4, 1000)
#     idx = torch.tensor([[2,3],
#             [1,0]],dtype=torch.long)

#     y = batch_index_select(x, idx)
# b = time.time()
# print(b-a)

# #! 选择行列
# x = torch.arange(1200).view(2, 6, 10, 10)
# print(x[1][0])
# idx = torch.tensor([[2,3],
#         [1,0]],dtype=torch.long)    #[B,left]

# y = torch.gather(x, dim=3, index=idx.unsqueeze(1).unsqueeze(1).expand(-1,6,10,-1))
# print(y[1][0])
# y = torch.gather(y, dim=2, index=idx.unsqueeze(1).unsqueeze(-1).expand(-1,6,-1,2))
# # print(y.shape)
# print(y[1][0])


#! 对比dim 为1的选择
x = torch.arange(16 * 196 * 384).view(16, 196, 384)
idx = torch.arange(100, dtype=torch.long, device=x.device)

a = time.time()
for i in range(10000):
    y = torch.gather(x, dim=1, index=idx.unsqueeze(0).unsqueeze(-1).expand(16,-1,384))
b = time.time()
print(b-a)

a2 = time.time()
for i in range(10000):
    y = batch_index_select(x,idx.unsqueeze(0).expand(16,-1))
b2 = time.time()
print(b2-a2)


