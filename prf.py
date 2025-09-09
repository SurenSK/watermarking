import math, hashlib
from typing import List, Any, Tuple, Optional, Sequence
import msgpack
import numpy as np
import torch


from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

def _pack_ctx(context: List[Any]) -> bytes:
    return msgpack.packb(context, use_bin_type=True) if context else b""

def _cpu_hkdf_getYs(salt: bytes, ikm: bytes, context_b: bytes, blen: int) -> List[float]:
    bytes_per = 3
    L = blen * bytes_per
    max_l = 255 * hashes.SHA256.digest_size
    if L > max_l:
        raise ValueError(f"Requested bytes {L} exceeds HKDF-SHA256 limit of {max_l}")
    hkdf = HKDF(algorithm=hashes.SHA256(), length=L, salt=salt, info=context_b)
    out = hkdf.derive(ikm)
    max_int = (1 << (8 * bytes_per)) - 1
    return [int.from_bytes(out[i*bytes_per:(i+1)*bytes_per], 'big')/max_int for i in range(blen)]

def getYs(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
    key32 = hashlib.blake2s(ikm + salt, digest_size=32).digest()
    ctx_b = _pack_ctx(context)
    nonce = hashlib.blake2s(ctx_b, digest_size=12).digest()
    nonces = torch.from_numpy(np.frombuffer(nonce, dtype='<u4').astype(np.int64).reshape(1,3))
    ys = _chacha20_prf_gpu_groups([(key32, nonces, blen)], device='cuda')[0]
    return ys

def getYsBatched(jobs: List[Tuple[bytes, bytes, List[Any], int]], max_workers: Optional[int] = None) -> List[List[float]]:
    keys = []
    ctx_bytes = []
    blens = []
    for salt, ikm, ctx, blen in jobs:
        keys.append(hashlib.blake2s(ikm + salt, digest_size=32).digest())
        ctx_bytes.append(_pack_ctx(ctx))
        blens.append(blen)
    nonces = [hashlib.blake2s(c, digest_size=12).digest() for c in ctx_bytes]
    nonces_arr = np.frombuffer(b''.join(nonces), dtype='<u4').astype(np.int64).reshape(len(jobs),3)
    nonces_tensor = torch.from_numpy(nonces_arr)
    groups = {}
    for i,(k,b) in enumerate(zip(keys, blens)):
        groups.setdefault((k,b), []).append(i)
    group_inputs = []
    order = []
    for (k,b), idxs in groups.items():
        group_inputs.append((k, nonces_tensor[idxs], b))
        order.append(idxs)
    group_outputs = _chacha20_prf_gpu_groups(group_inputs, device='cuda')
    out = [None]*len(jobs)
    for idxs, ys in zip(order, group_outputs):
        for j,v in zip(idxs, ys):
            out[j] = v
    return out

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

def _chacha20_words_to_unit_interval(w):
    f = w.to(torch.float64)
    f = f / 4294967295.0
    return f

def _chacha20_stream_to_y(key32: bytes, nonces_u32x3: torch.Tensor, blen: int, device: str):
    mask = torch.tensor((1<<32)-1, dtype=torch.int64, device=device)
    key_words = torch.from_numpy(np.frombuffer(key32, dtype='<u4').astype(np.int64))
    N = nonces_u32x3.shape[0]
    blocks = (blen + 15)//16
    out = torch.empty((N, blen), dtype=torch.float64, device=device)
    pos = 0
    for bi in range(blocks):
        ctr = torch.full((N,), bi, dtype=torch.int64, device=device)
        w = _chacha20_block_words_int64(key_words, nonces_u32x3, ctr, mask, device)
        y = _chacha20_words_to_unit_interval(w)
        take = min(16, blen - pos)
        out[:, pos:pos+take] = y[:, :take]
        pos += take
    torch.cuda.synchronize()
    return out

def _chacha20_prf_gpu_groups(group_inputs: Sequence[Tuple[bytes, torch.Tensor, int]], device: str) -> List[List[float]]:
    results = []
    for key32, nonces, blen in group_inputs:
        y = _chacha20_stream_to_y(key32, nonces.to(device=device), blen, device)
        results.append(y.cpu().numpy().tolist())
    return results
