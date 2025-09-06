import torch
import random
import timeit

def getP1(p:torch.Tensor,prefix:int,bitIdx:int)->float:
    v=p.shape[-1]; b=(v-1).bit_length()
    if not v or bitIdx>=b: return 0.0
    shift=b-bitIdx; start=prefix<<shift
    if start>=v: return 0.0
    cs=torch.nn.functional.pad(p.cumsum(0),(1,0))
    s0,s1,s2=cs[start],cs[min(start+(1<<(shift-1)),v)],cs[min(start+(1<<shift),v)]
    if(total:=s2-s0)<1e-9: return 0.0
    return((s2-s1)/total).item()

def getP1_cumsum(cs:torch.Tensor,prefix:int,bitIdx:int)->float:
    v=cs.shape[-1]-1
    if v<=0: return 0.0
    b=(v-1).bit_length()
    if not v or bitIdx>=b: return 0.0
    shift=b-bitIdx; start=prefix<<shift
    if start>=v: return 0.0
    s0,s1,s2=cs[start],cs[min(start+(1<<(shift-1)),v)],cs[min(start+(1<<shift),v)]
    if(total:=s2-s0)<1e-9: return 0.0
    return((s2-s1)/total).item()

def run_comparison(tensor_size:int=32000, num_runs:int=100000):
    p=torch.rand(tensor_size,dtype=torch.float32)
    p/=p.sum()
    cs=torch.nn.functional.pad(p.cumsum(0),(1,0))
    
    time_original=0
    time_cumsum=0
    
    for _ in range(num_runs):
        bitIdx=random.randint(1,14)
        prefix=random.randint(0,(1<<bitIdx)-1)
        
        start_orig=timeit.default_timer()
        getP1(p,prefix,bitIdx)
        time_original+=timeit.default_timer()-start_orig
        
        start_cumsum=timeit.default_timer()
        getP1_cumsum(cs,prefix,bitIdx)
        time_cumsum+=timeit.default_timer()-start_cumsum
        
    avg_orig = time_original / num_runs * 1e6
    avg_cumsum = time_cumsum / num_runs * 1e6
    
    print(f"Comparing functions over {num_runs} runs with tensor size {tensor_size}:")
    print(f"Original (getP1)      : {avg_orig:.3f} µs per call | {time_original:.2f}s")
    print(f"Refactored (getP1_cumsum): {avg_cumsum:.3f} µs per call | {time_cumsum:.2f}s")
    if avg_cumsum > 0:
        print(f"Speedup                : {avg_orig / avg_cumsum:.2f}x")

if __name__ == "__main__":
    run_comparison()
