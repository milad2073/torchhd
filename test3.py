from torchhd import functional
import torch
from matplotlib import pyplot as plt


vsa_tensors = ["BSC","MAP","HRR","FHRR",
               "BSBC","MCR", "VTB"]


for vsa in vsa_tensors:
    if vsa == "BSBC" or vsa == "MCR":
        hv  = functional.level(11, 900, vsa, block_size=1024)
        [a_random_vec] = functional.random(1, 900, vsa, block_size=1024)
        [one] = functional.identity(1, 900, vsa, block_size=1024)
        
    else:
        hv  = functional.level(11, 900, vsa)
        [a_random_vec] = functional.random(1, 900, vsa)
        [one] = functional.identity(1, 900, vsa)
    
    x = hv[0]
    xy_similarities = []
    identity_similarities = []
    for i in range(len(hv)):
        y = hv[len(hv)- 1 - i]
        sim1 = x.cosine_similarity(y)
        sim2 = x.bind(y.inverse()).cosine_similarity(one)
        xy_similarities.append(sim1)
        identity_similarities.append(sim2)
        
    # plt.plot()
    plt.plot(xy_similarities,identity_similarities,'-*')
    
plt.legend(vsa_tensors)
plt.xlabel("x similarity to y")
plt.ylabel("x bind inv(y) similarity to identitiy ")
plt.show()