from correlation_package_pytorch1_0.correlation import Correlation
from correlation_torch import CorrTorch
import torch
import torch.cuda
import time


# tested parameter set
mds = [2, 3, 4, 5, 6]
kszs = [1, 1, 1, 1, 1]
st1s = [1, 1, 1, 1, 1]
st2s = [1, 1, 1, 1, 1]

torch.set_default_dtype(torch.float32)
for md, ksz, st1, st2 in zip(mds, kszs, st1s, st2s):
    print(f"-----test for: maxdisp={md}, kernelsize={ksz}, stride1={st1}, stride2={st2}")
    input1 = torch.rand(32, 64, 80, 100).cuda()
    input2 = torch.rand(32, 64, 80, 100).cuda()

    torch.cuda.synchronize()
    startmem = torch.cuda.memory_allocated()
    start = time.time()
    corr_cuda = Correlation(pad_size=md, kernel_size=ksz, max_displacement=md, stride1=st1, stride2=st2, corr_multiply=1)
    out_cuda = corr_cuda(input1, input2)
    torch.cuda.synchronize()
    end = time.time()
    endmem = torch.cuda.memory_allocated()

    corr_my = CorrTorch(pad_size=md, kernel_size=ksz, max_displacement=md, stride1=st1, stride2=st2, corr_multiply=1)
    out_my = corr_my(input1, input2)
    end2 = time.time()
    torch.cuda.synchronize()
    end2mem = torch.cuda.memory_allocated()

    print(f"\tofficial corr time: {end - start}s, torch corr time: {end2 - end}s")
    print(f"\tofficial corr memory: {(endmem - startmem) / 1024 / 1024}MB, "
          f"torch corr cost: {(end2mem - startmem) / 1024 / 1024}MB")
    print(f"\tsum of abs err: {torch.sum(torch.abs(out_cuda - out_my))}")
    print(f"\tmean of abs err: {torch.mean(torch.abs(out_cuda - out_my))}")
