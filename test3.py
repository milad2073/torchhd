from torchhd import functional
import torch

vsa_tensors = ["BSC","MAP","HRR","FHRR",
               "BSBC","MCR", "VTB"]

for vsa in vsa_tensors:
    if vsa == "BSBC" or vsa == "MCR":
        hv  = functional.level(11, 1000000, vsa, block_size=1024)
        [a_random_vec] = functional.random(1, 1000000, vsa, block_size=1024)
        [one] = functional.identity(1, 1000000, vsa, block_size=1024)
        
    else:
        hv  = functional.level(11, 1000000, vsa)
        [a_random_vec] = functional.random(1, 1000000, vsa)
        [one] = functional.identity(1, 1000000, vsa)
    
    x,y = hv[0], hv[2]

    
    sim1 = x.cosine_similarity(y)
    sim2 = x.bind(a_random_vec).bind(y.inverse()).cosine_similarity(a_random_vec)
    if torch.isclose(sim1,sim2):
        print(f'{vsa}: Passed sim1 = sim2 = {sim1}')
    else:
        print(f'{vsa}: Failed sim1 = {sim1}, sim2 = {sim2}')