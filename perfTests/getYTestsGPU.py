import numpy as np
import msgpack
import timeit
import math
import functools
import time
import hmac, hashlib
import os
from typing import List, Any, Sequence, Union
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from concurrent.futures import ProcessPoolExecutor
try:
    import cupy as cp
    _cuda_src = r'''
    extern "C"{
    __device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n){return (x>>n)|(x<<(32-n));}
    __device__ void sha256_transform(unsigned int state[8], const unsigned char block[64]){
        const unsigned int K[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        unsigned int w[64];
        #pragma unroll
        for(int i=0;i<16;i++){
            int j=i*4;
            w[i]=((unsigned int)block[j]<<24)|((unsigned int)block[j+1]<<16)|((unsigned int)block[j+2]<<8)|((unsigned int)block[j+3]);
        }
        for(int i=16;i<64;i++){
            unsigned int s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            unsigned int s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        unsigned int a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
        for(int i=0;i<64;i++){
            unsigned int S1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            unsigned int ch=(e&f)^((~e)&g);
            unsigned int temp1=h+S1+ch+K[i]+w[i];
            unsigned int S0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            unsigned int maj=(a&b)^(a&c)^(b&c);
            unsigned int temp2=S0+maj;
            h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d; state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    struct Sha256Ctx{unsigned int state[8]; unsigned long long bitlen; int datalen; unsigned char data[64];};
    __device__ void sha256_init(Sha256Ctx* ctx){
        ctx->state[0]=0x6a09e667; ctx->state[1]=0xbb67ae85; ctx->state[2]=0x3c6ef372; ctx->state[3]=0xa54ff53a;
        ctx->state[4]=0x510e527f; ctx->state[5]=0x9b05688c; ctx->state[6]=0x1f83d9ab; ctx->state[7]=0x5be0cd19;
        ctx->bitlen=0ULL; ctx->datalen=0;
    }
    __device__ void sha256_update(Sha256Ctx* ctx, const unsigned char* data, unsigned long long len){
        for(unsigned long long i=0;i<len;i++){
            ctx->data[ctx->datalen++]=data[i];
            if(ctx->datalen==64){
                sha256_transform(ctx->state, ctx->data);
                ctx->bitlen+=512ULL;
                ctx->datalen=0;
            }
        }
    }
    __device__ void sha256_final(Sha256Ctx* ctx, unsigned char hash[32]){
        unsigned int i=ctx->datalen;
        ctx->data[i++]=0x80;
        if(i>56){
            for(;i<64;i++) ctx->data[i]=0;
            sha256_transform(ctx->state, ctx->data);
            i=0;
        }
        for(;i<56;i++) ctx->data[i]=0;
        ctx->bitlen+= (unsigned long long)ctx->datalen*8ULL;
        unsigned long long bl=ctx->bitlen;
        for(int j=0;j<8;j++) ctx->data[63-j]=(unsigned char)(bl>>(8*j));
        sha256_transform(ctx->state, ctx->data);
        for(i=0;i<4;i++){
            hash[i]=(ctx->state[0]>>(24-8*i))&0xff;
            hash[i+4]=(ctx->state[1]>>(24-8*i))&0xff;
            hash[i+8]=(ctx->state[2]>>(24-8*i))&0xff;
            hash[i+12]=(ctx->state[3]>>(24-8*i))&0xff;
            hash[i+16]=(ctx->state[4]>>(24-8*i))&0xff;
            hash[i+20]=(ctx->state[5]>>(24-8*i))&0xff;
            hash[i+24]=(ctx->state[6]>>(24-8*i))&0xff;
            hash[i+28]=(ctx->state[7]>>(24-8*i))&0xff;
        }
    }
    __device__ void hmac_sha256_three(const unsigned char key32[32],
                                      const unsigned char* seg1, unsigned int len1,
                                      const unsigned char* seg2, unsigned int len2,
                                      unsigned char c3,
                                      unsigned char out[32]){
        unsigned char k_ipad[64]; unsigned char k_opad[64];
        for(int i=0;i<64;i++){unsigned char kb=(i<32)?key32[i]:0; k_ipad[i]=kb^0x36; k_opad[i]=kb^0x5c;}
        Sha256Ctx ctx1; sha256_init(&ctx1); sha256_update(&ctx1,k_ipad,64);
        if(len1>0) sha256_update(&ctx1,seg1,len1);
        if(len2>0) sha256_update(&ctx1,seg2,len2);
        sha256_update(&ctx1,&c3,1);
        unsigned char inner[32]; sha256_final(&ctx1,inner);
        Sha256Ctx ctx2; sha256_init(&ctx2); sha256_update(&ctx2,k_opad,64); sha256_update(&ctx2,inner,32);
        sha256_final(&ctx2,out);
    }
    __global__ void hkdf_expand_many(const unsigned char* prk32,
                                     const unsigned char* flat_infos,
                                     const unsigned long long* offsets,
                                     const unsigned int* lengths,
                                     const int blen,
                                     const int n,
                                     double* out){
        int i=blockDim.x*blockIdx.x+threadIdx.x;
        if(i>=n) return;
        const unsigned char* info=flat_infos+offsets[i];
        unsigned int info_len=lengths[i];
        unsigned char Tprev[32];
        unsigned int Tprev_len=0;
        unsigned int blocks=((unsigned int)(blen*8)+31)/32;
        int y_idx=0; int byte_in_word=0;
        unsigned long long word=0ULL;
        const double denom=18446744073709551615.0;
        for(unsigned int bi=1; bi<=blocks; ++bi){
            unsigned char T[32];
            hmac_sha256_three(prk32, Tprev, Tprev_len, info, info_len, (unsigned char)bi, T);
            for(int b=0;b<32;b++){
                word=(word<<8)|(unsigned long long)T[b];
                byte_in_word++;
                if(byte_in_word==8){
                    out[(long long)i*blen + y_idx]=((double)word)/denom;
                    y_idx++; byte_in_word=0; word=0ULL;
                    if(y_idx==blen) return;
                }
            }
            for(int k=0;k<32;k++) Tprev[k]=T[k];
            Tprev_len=32;
        }
    }
    }
    '''
    _mod = cp.RawModule(code=_cuda_src, options=('-std=c++11',))
    _kernel = _mod.get_function('hkdf_expand_many')
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False
    cp = None

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

def _pack_contexts(contexts: Sequence[Union[bytes, bytearray, memoryview, object]]):
    if isinstance(contexts, np.ndarray) and contexts.dtype==np.uint8 and contexts.ndim==1:
        lengths = np.array([len(contexts)], dtype=np.uint32)
        offsets = np.array([0], dtype=np.uint64)
        flat = contexts
        return flat, offsets, lengths
    flat_parts=[]
    lengths=[]
    offsets=[]
    offset=0
    for ctx in contexts:
        if isinstance(ctx, (bytes, bytearray, memoryview)):
            b = bytes(ctx)
        else:
            b = msgpack.packb(ctx, use_bin_type=True)
        flat_parts.append(b)
        l = len(b)
        lengths.append(l)
        offsets.append(offset)
        offset+=l
    flat = np.frombuffer(b''.join(flat_parts), dtype=np.uint8)
    return flat, np.array(offsets, dtype=np.uint64), np.array(lengths, dtype=np.uint32)

def _hkdf_prf_gpu_core(prk: bytes, flat_infos_np: np.ndarray, offsets_np: np.ndarray, lengths_np: np.ndarray, blen: int, stream=None):
    n = len(lengths_np)
    d_prk = cp.asarray(np.frombuffer(prk, dtype=np.uint8))
    d_flat = cp.asarray(flat_infos_np)
    d_off = cp.asarray(offsets_np)
    d_len = cp.asarray(lengths_np)
    d_out = cp.empty((n, blen), dtype=cp.float64)
    threads=256
    blocks=(n+threads-1)//threads
    if stream is None:
        _kernel((blocks,), (threads,), (d_prk, d_flat, d_off, d_len, np.int32(blen), np.int32(n), d_out))
        cp.cuda.runtime.deviceSynchronize()
    else:
        with stream:
            _kernel((blocks,), (threads,), (d_prk, d_flat, d_off, d_len, np.int32(blen), np.int32(n), d_out))
    return d_out

def hkdf_prf_gpu(salt: bytes, ikm: bytes, contexts: Sequence[Union[bytes, bytearray, memoryview, object]], blen: int, batch_size: int=None, stream=None):
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    n_total = len(contexts)
    if batch_size is None:
        flat, offsets, lengths = _pack_contexts(contexts)
        return _hkdf_prf_gpu_core(prk, flat, offsets, lengths, blen, stream)
    out_all = np.empty((n_total, blen), dtype=np.float64)
    start=0
    while start<n_total:
        end=min(start+batch_size, n_total)
        flat, offsets, lengths = _pack_contexts(contexts[start:end])
        out_batch = _hkdf_prf_gpu_core(prk, flat, offsets, lengths, blen, stream)
        out_all[start:end,:]=cp.asnumpy(out_batch)
        start=end
    return out_all

def gpu_test_contexts(contexts_bytes: List[bytes], blen: int, key_bytes: bytes, salt_bytes: bytes):
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy/CUDA not available")
    ys = hkdf_prf_gpu(salt_bytes, key_bytes, contexts_bytes, blen, batch_size=None)
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
    t_st = timeit.timeit(st_wrapper, number=N_BENCHMARK_RUNS)
    t_mp = timeit.timeit(mp_wrapper, number=N_BENCHMARK_RUNS)
    if GPU_AVAILABLE:
        gpu_wrapper = functools.partial(gpu_test_contexts, contexts_bytes, BLEN, key_bytes, salt_bytes)
        t_gpu = timeit.timeit(gpu_wrapper, number=N_BENCHMARK_RUNS)
    else:
        t_gpu = float('nan')
    print(f"stTest (HKDF single-thread, contexts vary): {t_st/N_BENCHMARK_RUNS:.6f}s per run")
    print(f"mpTest (HKDF multi-process, contexts vary): {t_mp/N_BENCHMARK_RUNS:.6f}s per run")
    if GPU_AVAILABLE:
        print(f"gpuTest (HKDF-Expand on GPU, contexts vary): {t_gpu/N_BENCHMARK_RUNS:.6f}s per run")
        print(f"Speedup ST/MP: {t_st / t_mp:.2f}x")
        print(f"Speedup ST/GPU: {t_st / t_gpu:.2f}x")
        print(f"Speedup MP/GPU: {t_mp / t_gpu:.2f}x")
    else:
        print("gpuTest skipped (CuPy/CUDA not available)")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total script time: {time.time()-t0:.2f}s")
