import torch

n_embd = 2560

PATH = './models/WeLM/2_7B_ckpt.pt'

model = torch.load(PATH)
tmp = model['language_model']['encoder']

for name, data in tmp.items():
    print(name, data.shape)
    print(data)