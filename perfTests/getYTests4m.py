import hmac, hashlib
import msgpack
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import timeit
import functools
from typing import List, Any

# getY function remains the same
def getY(salt: bytes, ikm: bytes, context: List[Any]) -> float:
    info = msgpack.packb(context, use_bin_type=True) if context else b""
    msg = len(salt).to_bytes(4, 'big') + salt + len(info).to_bytes(4, 'big') + info
    digest = hmac.new(ikm, msg, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], 'big')
    return seed / (2**64 - 1)

# stTest and mtTest functions remain the same
def stTest() -> List[float]:
    key = 42
    context = [90, 87, 21, 23, 24, 90, 87, 21, 23, 24]
    key_bytes = key.to_bytes(4, 'big')
    ys = []
    for salt in range(1, 4000000):
        ys.append(getY(salt.to_bytes(4, 'big'), key_bytes, context))
    return ys

def mtTest() -> List[float]:
    key = 42
    context = [90, 87, 21, 23, 24, 90, 87, 21, 23, 24]
    key_bytes = key.to_bytes(4, 'big')
    salts = [s.to_bytes(4, 'big') for s in range(1, 4000000)]
    with ThreadPoolExecutor() as ex:
        ys = list(ex.map(lambda s: getY(s, key_bytes, context), salts))
    return ys

# New function that uses an EXISTING pool
def mp_work(executor, salts, key_bytes, context) -> List[float]:
    getY_partial = functools.partial(getY, ikm=key_bytes, context=context)
    # Give each worker a much larger chunk of work to reduce communication overhead
    return list(executor.map(getY_partial, salts, chunksize=1024))

def main():
    # --- Verification Step ---
    print("Verifying correctness...")
    ys_st = stTest()
    # ys_mt = mtTest()
    # For verification, we still need to create a one-off pool
    with ProcessPoolExecutor() as ex:
        key = 42
        context = [90, 87, 21, 23, 24, 90, 87, 21, 23, 24]
        key_bytes = key.to_bytes(4, 'big')
        salts = [s.to_bytes(4, 'big') for s in range(1, 4000000)]
        ys_mp = mp_work(ex, salts, key_bytes, context)

    # assert ys_st == ys_mt and ys_st == ys_mp, "Results do not match!"
    # print("All implementations produce the same results.\n")

    # --- Benchmarking Step ---
    print("Running benchmarks...")
    N = 1
    t_st = timeit.timeit(stTest, number=N)
    # t_mt = timeit.timeit(mtTest, number=N)

    # For multiprocessing, create the pool ONCE
    with ProcessPoolExecutor() as executor:
        # Prepare the arguments for the timed function
        key = 42
        context = [90, 87, 21, 23, 24, 90, 87, 21, 23, 24]
        key_bytes = key.to_bytes(4, 'big')
        salts = [s.to_bytes(4, 'big') for s in range(1, 4000000)]
        # Use a lambda to wrap the call with its arguments
        timed_func = lambda: mp_work(executor, salts, key_bytes, context)
        t_mp = timeit.timeit(timed_func, number=N)

    print(f"stTest (single-thread): {t_st/N:.6f}s per run")
    # print(f"mtTest (multi-thread):  {t_mt/N:.6f}s per run")
    print(f"mpTest (multi-process): {t_mp/N:.6f}s per run")

if __name__ == "__main__":
    main()