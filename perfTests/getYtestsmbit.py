import hmac, hashlib
import msgpack
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import timeit
import math
import functools
from typing import List, Any
import time

import torch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from typing import List,Optional


def getY(salt: bytes, ikm: bytes, context: List[Any]) -> float:
    info = msgpack.packb(context, use_bin_type=True) if context else b""
    msg = len(salt).to_bytes(4, 'big') + salt + len(info).to_bytes(4, 'big') + info
    digest = hmac.new(ikm, msg, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], 'big') # make seed bigger by (blen x)
    return seed / (2**64 - 1) # then we'd get blen seed / (2**64 - 1)

def serialTest():
    key = 42
    key_bytes = key.to_bytes(4, 'big')
    context = [90, 87, 21, 23, 24, 90, 87, 21, 23, 24]
    Ys = []
    for salt in range(15):
        Ys.append(getY(salt.to_bytes(4, 'big'), key_bytes, context))
    return Ys

def getYs(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
    L=blen*8
    max_l=255*hashes.SHA256.digest_size
    if L > max_l:
        raise ValueError(f"Requested bytes {L} exceeds HKDF-SHA256 limit of {max_l}")

    info = msgpack.packb(context, use_bin_type=True) if context else b""
    hkdf=HKDF(algorithm=hashes.SHA256(),length=L,salt=salt,info=info)
    output_bytes=hkdf.derive(ikm)
    
    divisor=2**64-1
    chunk_size=8
    return [
        int.from_bytes(output_bytes[i*chunk_size:(i+1)*chunk_size],'big')/divisor
        for i in range(blen)
    ]

def parallelTest():
    key = 42
    key_bytes = key.to_bytes(4, 'big')
    salt = 32
    salt_bytes = salt.to_bytes(4, 'big')
    blen = 15
    context = [90, 87, 21, 23, 24, 90, 87, 21, 23, 24]
    return getYs(salt_bytes, key_bytes, context, blen)

# print(serialTest())
# print(parallelTest())
# print(parallelTest()) # should be deterministic (same)
# print(parallelTest()) # it is, which is good

# setup_code="""
# from __main__ import serialTest, parallelTest
# """

# n_runs=100000

# t_serial=timeit.timeit(stmt="serialTest()",setup=setup_code,number=n_runs)
# t_parallel=timeit.timeit(stmt="parallelTest()",setup=setup_code,number=n_runs)

# print(f"--- Results (over {n_runs} runs) ---")
# print(f"serialTest (15 HMAC calls):   {t_serial:.6f} seconds")
# print(f"parallelTest (1 HKDF call): {t_parallel:.6f} seconds")
# print(f"Ratio (Serial / Parallel):  {t_serial / t_parallel:.2f}x faster")






