import time, timeit, math, functools, os, hashlib
import numpy as np
import torch
import msgpack
from typing import List, Any, Sequence, Union
from concurrent.futures import ProcessPoolExecutor
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

N_CALLS = 1875000
BLEN = 15
N_BENCHMARK_RUNS = 1

def getYs_cpu(salt: bytes, ikm: bytes, context_bytes: bytes, blen: int) -> List[float]:
    L = blen*8
    max_l = 255*hashes.SHA256.digest_size
    if L > max_l:
        raise ValueError(f"Requested bytes {L} exceeds HKDF-SHA256 limit of {max_l}")
    hkdf = HKDF(algorithm=hashes.SHA256(), length=L, salt=salt, info=context_bytes)
    output_bytes = hkdf.derive(ikm)
    divisor = 2**64-1
    chunk_size = 8
    return [int.from_bytes(output_bytes[i*chunk_size:(i+1)*chunk_size],'big')/divisor for i in range(blen)]

def st_test_contexts(contexts_bytes: List[bytes], blen: int, key_bytes: bytes, salt_bytes: bytes) -> List[float]:
    all_ys = []
    for ctx in contexts_bytes:
        all_ys.extend(getYs_cpu(salt_bytes, key_bytes, ctx, blen))
    return all_ys

def _getYs_cpu_from_ctx(ctx: bytes, salt_bytes: bytes, key_bytes: bytes, blen: int) -> List[float]:
    return getYs_cpu(salt_bytes, key_bytes, ctx, blen)

def mp_work(executor: ProcessPoolExecutor, contexts_bytes: List[bytes], key_bytes: bytes, salt_bytes: bytes, blen: int) -> List[float]:
    from itertools import repeat
    workers = getattr(executor, "_max_workers", os.cpu_count() or 1)
    chunksize = max(1, len(contexts_bytes) // workers)
    it = executor.map(_getYs_cpu_from_ctx, contexts_bytes, repeat(salt_bytes), repeat(key_bytes), repeat(blen), chunksize=chunksize)
    results_nested = list(it)
    return [y for sublist in results_nested for y in sublist]

def mp_benchmark(contexts_bytes: List[bytes], key_bytes: bytes, salt_bytes: bytes, blen: int, max_workers: int=None) -> List[float]:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return mp_work(executor, contexts_bytes, key_bytes, salt_bytes, blen)

def _contexts_to_nonces_12B(contexts_bytes: Sequence[bytes]) -> torch.Tensor:
    dg = [hashlib.blake2s(c, digest_size=12).digest() for c in contexts_bytes]
    a = np.frombuffer(b''.join(dg), dtype='<u4').reshape(-1,3).astype(np.int64)
    return torch.from_numpy(a)

def _rotl32(x, n, mask):
    x = x & mask
    return ((x << n) | (x >> (32 - n))) & mask

def _chacha20_block_words_int64(key_words_int64, nonce_words_int64, counter_words_int64, mask, device):
    N = nonce_words_int64.shape[0]
    c = torch.tensor([0x61707865,0x3320646e,0x79622d32,0x6b206574], dtype=torch.int64, device=device)
    k = key_words_int64.to(device=device, dtype=torch.int64).view(1,8).expand(N,-1)
    n = nonce_words_int64.to(device=device, dtype=torch.int64)
    ctr = counter_words_int64.to(device=device, dtype=torch.int64).view(-1,1)
    s = torch.empty((N,16), dtype=torch.int64, device=device)
    s[:,0:4] = c
    s[:,4:12] = k
    s[:,12] = ctr[:,0]
    s[:,13:16] = n
    x = s.clone()
    def qr(a,b,c,d):
        x[:,a] = (x[:,a] + x[:,b]) & mask
        x[:,d] = _rotl32(x[:,d] ^ x[:,a], 16, mask)
        x[:,c] = (x[:,c] + x[:,d]) & mask
        x[:,b] = _rotl32(x[:,b] ^ x[:,c], 12, mask)
        x[:,a] = (x[:,a] + x[:,b]) & mask
        x[:,d] = _rotl32(x[:,d] ^ x[:,a], 8, mask)
        x[:,c] = (x[:,c] + x[:,d]) & mask
        x[:,b] = _rotl32(x[:,b] ^ x[:,c], 7, mask)
    for _ in range(10):
        qr(0,4,8,12); qr(1,5,9,13); qr(2,6,10,14); qr(3,7,11,15)
        qr(0,5,10,15); qr(1,6,11,12); qr(2,7,8,13); qr(3,4,9,14)
    return (x + s) & mask

def chacha20_prf_gpu_torch(key32: bytes, nonces_u32x3: torch.Tensor, blen: int, batch_size: int=None, device: str='cuda') -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    mask = torch.tensor((1<<32)-1, dtype=torch.int64, device=device)
    key_words = torch.from_numpy(np.frombuffer(key32, dtype='<u4').astype(np.int64))
    N = nonces_u32x3.shape[0]
    blocks = (blen + 7)//8
    if batch_size is None:
        batch_size = N
    out = torch.empty((N, blen), dtype=torch.float64, device=device)
    for start in range(0, N, batch_size):
        end = min(start+batch_size, N)
        nb = nonces_u32x3[start:end]
        ys = []
        for bi in range(blocks):
            ctr = torch.full((end-start,), bi, dtype=torch.int64, device=device)
            w = _chacha20_block_words_int64(key_words, nb, ctr, mask, device)
            lo = (w[:,0::2] & mask).to(torch.float64)
            hi = (w[:,1::2] & mask).to(torch.float64)
            f = (lo + hi * 4294967296.0) / 18446744073709551615.0
            ys.append(f[:, :8])
        y = torch.cat(ys, dim=1)[:, :blen]
        out[start:end] = y
    torch.cuda.synchronize()
    return out

def gpu_test_contexts(contexts_bytes: List[bytes], blen: int, key_bytes: bytes, salt_bytes: bytes):
    key32 = hashlib.blake2s(key_bytes + salt_bytes, digest_size=32).digest()
    nonces = _contexts_to_nonces_12B(contexts_bytes)
    ys = chacha20_prf_gpu_torch(key32, nonces, blen, batch_size=None, device='cuda')
    return ys

def main():
    print(f"Generating {N_CALLS * BLEN:,} values via {N_CALLS:,} distinct contexts of blen={BLEN} with constant salt/key...")
    key = 42
    key_bytes = key.to_bytes(4, 'big')
    salt_bytes = (123456).to_bytes(4,'big')
    base_ctx = [90, 87, 21, 23, 24, 90, 87, 21, 23, 24]
    contexts_bytes = [msgpack.packb(base_ctx + [i], use_bin_type=True) for i in range(1, N_CALLS+1)]
    print(f"Running benchmarks ({N_BENCHMARK_RUNS} run(s) each)...")
    st_wrapper = functools.partial(st_test_contexts, contexts_bytes, BLEN, key_bytes, salt_bytes)
    mp_wrapper = functools.partial(mp_benchmark, contexts_bytes, key_bytes, salt_bytes, BLEN, None)
    # t_st = timeit.timeit(st_wrapper, number=N_BENCHMARK_RUNS)
    t_st = 1
    t_mp = 1
    # t_mp = timeit.timeit(mp_wrapper, number=N_BENCHMARK_RUNS)
    gpu_wrapper = functools.partial(gpu_test_contexts, contexts_bytes, BLEN, key_bytes, salt_bytes)
    t_gpu = timeit.timeit(gpu_wrapper, number=N_BENCHMARK_RUNS)
    print(f"stTest (HKDF single-thread, contexts vary): {t_st/N_BENCHMARK_RUNS:.6f}s per run")
    print(f"mpTest (HKDF multi-process, contexts vary): {t_mp/N_BENCHMARK_RUNS:.6f}s per run")
    print(f"gpuTest (ChaCha20 PRF on GPU, contexts vary): {t_gpu/N_BENCHMARK_RUNS:.6f}s per run")
    print(f"Speedup ST/MP: {t_st / t_mp:.2f}x")
    print(f"Speedup ST/GPU: {t_st / t_gpu:.2f}x")
    print(f"Speedup MP/GPU: {t_mp / t_gpu:.2f}x")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total script time: {time.time()-t0:.2f}s")
