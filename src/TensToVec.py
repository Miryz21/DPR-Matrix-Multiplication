import numpy as np
import torch

u = np.matrix('1 0 0 1; 0 0 1 1; 1 0 0 0; 0 0 0 1; 1 1 0 0; -1 0 1 0; 0 1 0 -1; 0 0 0 0')
v = np.matrix('1 0 0 1; 1 0 0 0; 0 1 0 -1; -1 0 1 0; 0 0 0 1; 1 1 0 0; 0 0 1 1; 0 0 0 0')
w = np.matrix('1 0 0 1; 0 0 1 -1; 0 1 0 1; 1 0 1 0; -1 1 0 0; 0 0 0 1; 1 0 0 0; 0 0 0 0')

# u = np.matrix('1 0 0 0; 0 1 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; 0 0 1 0; 0 0 0 1')
# v = np.matrix('1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1; 1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1')
# w = np.matrix('1 0 0 0; 1 0 0 0; 0 1 0 0; 0 1 0 0; 0 0 1 0; 0 0 1 0; 0 0 0 1; 0 0 0 1')

T = torch.tensor([[[0]*4]*4]*4)
# print(T)

for i in range(8):
    # print(torch.outer(torch.tensor(u[i])[0], torch.tensor(v[i])[0]))
    summ = torch.outer(torch.tensor(u[i])[0], torch.tensor(v[i])[0])
    for j in range(4):
        T[j] += summ * np.array(w)[i][j]

for i in range(6):
    summ = torch.outer(torch.tensor(u[i])[0], torch.tensor(v[i])[0])
    for j in range(4):
        T[j] -= summ * np.array(w)[i][j]

print(T)