import torchhd
from torchhd import functional
import torch

generator = torch.Generator()
generator.manual_seed(2)

hv = functional.circular(8, 1000000, 'MCR', generator=generator, dtype=torch.float32,block_size=64)
# hv = functional.circular(8, 1000000, 'HRR', generator=generator, dtype=torch.float32,)
sims = functional.cosine_similarity(hv[0], hv)
sims_diff = sims[:-1] - sims[1:]

print(sims_diff.sign())
print(sims_diff.abs())

print(sims)
