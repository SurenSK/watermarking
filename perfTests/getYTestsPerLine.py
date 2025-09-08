import cProfile, pstats, msgpack, os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from typing import List

def getYs(salt: bytes, ikm: bytes, context, blen: int) -> List[float]:
    L=blen*8
    max_l=255*hashes.SHA256.digest_size
    if L > max_l:
        raise ValueError(f"Requested bytes {L} exceeds HKDF-SHA256 limit of {max_l}")
    info=msgpack.packb(context, use_bin_type=True) if context else b""
    hkdf=HKDF(algorithm=hashes.SHA256(),length=L,salt=salt,info=info)
    output_bytes=hkdf.derive(ikm)
    bytesPer=8
    return [
        int.from_bytes(output_bytes[i*bytesPer:(i+1)*bytesPer],'big')/(2**64-1)
        for i in range(blen)
    ]

SALT=os.urandom(16)
IKM=os.urandom(32)
CONTEXT=["user:some_id", "purpose:derivation"]
BLEN=15
NUM_CALLS=1000000

def run_benchmark():
    for _ in range(NUM_CALLS):
        getYs(SALT, IKM, CONTEXT, BLEN)

print(f"Running {NUM_CALLS} iterations with blen={BLEN}...")

profiler=cProfile.Profile()
profiler.enable()
run_benchmark()
profiler.disable()

print("Benchmark complete. Profiling Results (sorted by cumulative time):\n")
stats=pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()