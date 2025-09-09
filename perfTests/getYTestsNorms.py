import os
import math
import msgpack
import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional, Any, Callable

BLEN_PER_JOB=1020
N_JOBS=3922
CHUNK_SIZE=8
DIVISOR=2**64-1
INV_POW64=2**-64
INV_POW53=2**-53
MAX_L=255*hashes.SHA256.digest_size

#--- Three Core Worker Functions ---
def getYs_v1_div(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
    L=blen*8
    if L > MAX_L: raise ValueError("Limit")
    info=msgpack.packb(context, use_bin_type=True) if context else b""
    hkdf=HKDF(algorithm=hashes.SHA256(),length=L,salt=salt,info=info)
    output_bytes=hkdf.derive(ikm)
    return [
        int.from_bytes(output_bytes[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE],'big')/DIVISOR
        for i in range(blen)
    ]

def getYs_v2_mul(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
    L=blen*8
    if L > MAX_L: raise ValueError("Limit")
    info=msgpack.packb(context, use_bin_type=True) if context else b""
    hkdf=HKDF(algorithm=hashes.SHA256(),length=L,salt=salt,info=info)
    output_bytes=hkdf.derive(ikm)
    return [
        int.from_bytes(output_bytes[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE],'big')*2**-64
        for i in range(blen)
    ]

def getYs_v3_man(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
    L=blen*8
    if L > MAX_L: raise ValueError("Limit")
    info=msgpack.packb(context, use_bin_type=True) if context else b""
    hkdf=HKDF(algorithm=hashes.SHA256(),length=L,salt=salt,info=info)
    output_bytes=hkdf.derive(ikm)
    return [
        (int.from_bytes(output_bytes[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE],'big')>>11)*INV_POW53
        for i in range(blen)
    ]

#--- Three Picklable Helper Functions for mapping ---
def _helper_v1(j_args):
    return getYs_v1_div(*j_args)

def _helper_v2(j_args):
    return getYs_v2_mul(*j_args)

def _helper_v3(j_args):
    return getYs_v3_man(*j_args)

#--- Refactored Batch Function ---
def getYsBatched(
    map_func: Callable, 
    jobs: List[Tuple[bytes, bytes, List[Any], int]], 
    max_workers: Optional[int] = None
) -> List[List[float]]:
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        nw = ex._max_workers if ex._max_workers else os.cpu_count() or 1
        cs = max(1, math.ceil(len(jobs) / nw))
        return list(ex.map(map_func, jobs, chunksize=cs))

#--- Benchmark Execution ---
if __name__ == "__main__":
    total_ys = N_JOBS * BLEN_PER_JOB
    print(f"Generating one job list for {N_JOBS} jobs ({total_ys:,} total values)...")
    CONTEXT=[b"benchmark_run", 999]
    JOBS=[
        (os.urandom(16), os.urandom(32), CONTEXT, BLEN_PER_JOB)
        for _ in range(N_JOBS)
    ]
    print("Job list created. Starting benchmarks...\n")

    # --- Test V1 ---
    start_t1 = time.monotonic()
    results_v1 = getYsBatched(_helper_v1, JOBS)
    end_t1 = time.monotonic()
    t1 = end_t1 - start_t1
    print(f"V1 (Div 2**64-1):   {t1:.6f} sec")
    del results_v1 # Clear memory

    # --- Test V2 ---
    start_t2 = time.monotonic()
    results_v2 = getYsBatched(_helper_v2, JOBS)
    end_t2 = time.monotonic()
    t2 = end_t2 - start_t2
    print(f"V2 (Mult 2**-64):   {t2:.6f} sec")
    del results_v2 # Clear memory

    # --- Test V3 ---
    start_t3 = time.monotonic()
    results_v3 = getYsBatched(_helper_v3, JOBS)
    end_t3 = time.monotonic()
    t3 = end_t3 - start_t3
    print(f"V3 (Mantissa 53bit):{t3:.6f} sec")
    del results_v3 # Clear memory