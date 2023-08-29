import torch

n_embd = 2560

PATH = './models/WeLM/2_7B_ckpt.pt'
PATH2 = './2_7B_ckpt.pt'

model = torch.load(PATH)
tmp = model['language_model']['encoder']

l_weight = len('query_key_value.weight')
l_bias = len('query_key_value.bias')

tmp1 = {}
for name, data in tmp.items():
    if name.endswith('query_key_value.weight'):
        q = data[:n_embd]
        k = data[n_embd:2*n_embd]
        v = data[2*n_embd:]
        tmp1[name[:-l_weight]+'query.weight'] = q
        tmp1[name[:-l_weight]+'key.weight'] = k
        tmp1[name[:-l_weight]+'value.weight'] = v
    elif name.endswith('query_key_value.bias'):
        q = data[:n_embd]
        k = data[n_embd:2*n_embd]
        v = data[2*n_embd:]
        tmp1[name[:-l_bias]+'query.bias'] = q
        tmp1[name[:-l_bias]+'key.bias'] = k
        tmp1[name[:-l_bias]+'value.bias'] = v
    else:
        tmp1[name] = data

model['language_model']['encoder'] = tmp1
torch.save(model, PATH2)