from torchhd import functional
from torchhd.functional import cosine_similarity as sim
import torch

vsa_tensors = ["BSC","MAP","HRR","FHRR",
               "BSBC","VTB","MCR"]

for vsa in vsa_tensors:
    
    if vsa == "BSBC" or vsa == "MCR":
        hvs = functional.level(6, 1000000, vsa, block_size=1024)
    else:
        hvs = functional.level(6, 1000000, vsa)

    a,b,c = hvs[0], hvs[1], hvs[3]
    
    if torch.isclose(1/sim(a,c) ,  1/sim(a,b) + 1/sim(b,c) ):
        
        print(f'{vsa}: Ok ')
    else:
        print(f'{vsa}: Error ==> sim(a,c)={sim(a,c)} , sim(a,b)={sim(a,b)} + sim(b,c)={sim(b,c)}')
        