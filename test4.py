from torchhd import functional
import torch
from matplotlib import pyplot as plt

vsa_tensors = ["BSC","MAP","HRR","FHRR",
               "BSBC","VTB","MCR"]

generator = torch.Generator()
generator.manual_seed(123)

for vsa in vsa_tensors:
    if vsa == "BSBC" or vsa == "MCR":
        hv  = functional.level(21, 10000, vsa,generator=generator, block_size=1024)
        [rand_vec_1,rand_vec_2] = functional.random(2, 10000, vsa,generator=generator, block_size=1024)
        [one] = functional.identity(1, 10000, vsa, block_size=1024)
        
    else:
        hv  = functional.level(21, 10000, vsa,generator=generator,)
        [rand_vec_1,rand_vec_2] = functional.random(2, 10000, vsa,generator=generator,)
        [one] = functional.identity(1, 10000, vsa,)
    
    c1 = hv[0]
    c3 = hv[-1]
    
    print(f'{vsa}: Similarity c1 to c3 ==> {c1.cosine_similarity(c3)}')
    
    ti = []
    t1t3_similarities = []
    for i in range(len(hv)):
        t1 = hv[i]
    
        # sim1 = c1.cosine_similarity(t1),c3.cosine_similarity(t1)
        
        sim = c1.bind(c3).bind(t1.inverse()).cosine_similarity(t1)
        
        ti.append(i)
        t1t3_similarities.append(sim)
        
    # plt.plot()
    plt.plot(ti,t1t3_similarities,'-*')

plt.legend(vsa_tensors)
plt.xlabel("i")
plt.ylabel("L(i) to L'(i) similarity ")
ax = plt.gca()
ax.set_ylim([-.2,1.2])
plt.show()
            