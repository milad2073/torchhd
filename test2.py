from torchhd import functional

vsa_tensors = ["BSC","MAP","HRR","FHRR",
               "BSBC","VTB","MCR"]

for vsa in vsa_tensors:
    
    if vsa == "BSBC" or vsa == "MCR":
        [a,b] = functional.random(2, 1000000, vsa, block_size=1024)
    else:
        [a,b] = functional.random(2, 1000000, vsa)

    one = a.inverse().bind(a)
    
    sim = b.bind(one).cosine_similarity(b)
    print(f'{vsa}: Similarity ==> {sim}')