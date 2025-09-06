import msgpack
from concurrent.futures import ProcessPoolExecutor
import timeit
import math
import functools
from typing import List, Any
import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

N_CALLS = 1875000
BLEN = 15
N_BENCHMARK_RUNS = 1

def getYs(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
    L=blen*8
    max_l=255*hashes.SHA256.digest_size
    if L > max_l:
        raise ValueError(f"Requested bytes {L} exceeds HKDF-SHA256 limit of {max_l}")
    info=msgpack.packb(context, use_bin_type=True) if context else b""
    hkdf=HKDF(algorithm=hashes.SHA256(),length=L,salt=salt,info=info)
    output_bytes=hkdf.derive(ikm)
    divisor=2**64-1
    chunk_size=8
    return [
        int.from_bytes(output_bytes[i*chunk_size:(i+1)*chunk_size],'big')/divisor
        for i in range(blen)
    ]

def st_test(n_calls: int, blen: int, key_bytes: bytes, context: List[Any]) -> List[float]:
    all_ys = []
    for salt_int in range(1, n_calls + 1):
        all_ys.extend(getYs(salt_int.to_bytes(4, 'big'), key_bytes, context, blen))
    return all_ys

def mp_work(executor: ProcessPoolExecutor, salts_list: List[bytes], key_bytes: bytes, context: List[Any], blen: int) -> List[float]:
    getYs_partial=functools.partial(getYs, ikm=key_bytes, context=context, blen=blen)
    chunksize=math.ceil(len(salts_list) // executor._max_workers)
    results_nested=list(executor.map(getYs_partial, salts_list, chunksize=chunksize))
    return [y for sublist in results_nested for y in sublist]

def main():
    print(f"Generating {N_CALLS * BLEN:,} values via {N_CALLS:,} calls of blen={BLEN}...")
    key=42
    context=[90, 87, 21, 23, 24, 90, 87, 21, 23, 24]
    key_bytes=key.to_bytes(4, 'big')
    salts_list=[s.to_bytes(4, 'big') for s in range(1, N_CALLS + 1)]
    
    print(f"Running benchmarks ({N_BENCHMARK_RUNS} run(s) each)...")
    st_wrapper=lambda: st_test(N_CALLS, BLEN, key_bytes, context)
    t_st=timeit.timeit(st_wrapper, number=N_BENCHMARK_RUNS)

    with ProcessPoolExecutor() as executor:
        mp_wrapper=lambda: mp_work(executor, salts_list, key_bytes, context, BLEN)
        t_mp=timeit.timeit(mp_wrapper, number=N_BENCHMARK_RUNS)

    print(f"stTest (HKDF single-thread): {t_st/N_BENCHMARK_RUNS:.6f}s per run")
    print(f"mpTest (HKDF multi-process): {t_mp/N_BENCHMARK_RUNS:.6f}s per run")
    print(f"Speedup (ST / MP): {t_st / t_mp:.2f}x")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total script time (including pool startup/shutdown): {time.time()-t0:.2f}s")


# import msgpack
# import math
# from concurrent.futures import ProcessPoolExecutor
# from typing import List, Any, Tuple, Optional
# from cryptography.hazmat.primitives import hashes
# from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# def getYs(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
#     L=blen*8
#     max_l=255*hashes.SHA256.digest_size
#     if L > max_l:
#         raise ValueError(f"Requested bytes {L} exceeds HKDF-SHA256 limit of {max_l}")
#     info=msgpack.packb(context, use_bin_type=True) if context else b""
#     hkdf=HKDF(algorithm=hashes.SHA256(),length=L,salt=salt,info=info)
#     output_bytes=hkdf.derive(ikm)
#     divisor=2**64-1
#     chunk_size=8
#     return [
#         int.from_bytes(output_bytes[i*chunk_size:(i+1)*chunk_size],'big')/divisor
#         for i in range(blen)
#     ]

# def getYs_batch_mp(
#     jobs: List[Tuple[bytes, bytes, List[Any], int]], 
#     max_workers: Optional[int] = None
# ) -> List[List[float]]:
#     with ProcessPoolExecutor(max_workers=max_workers) as ex:
#         nw = ex._max_workers if ex._max_workers else 1
#         cs = max(1, math.ceil(len(jobs) / nw))
#         return list(ex.map(lambda j_args: getYs(*j_args), jobs, chunksize=cs))
