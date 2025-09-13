import numpy as np
import cupy as cp
import torch
import hmac, hashlib
from typing import Any, List, Optional, Tuple, Sequence

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
__device__ void hmac_sha256_init(const unsigned char* key32, unsigned char k_ipad[64], unsigned char k_opad[64]){
    for(int i=0;i<64;i++){unsigned char kb=(i<32)?key32[i]:0; k_ipad[i]=kb^0x36; k_opad[i]=kb^0x5c;}
}
__device__ void hmac_sha256_digest(const unsigned char k_ipad[64], const unsigned char k_opad[64],
                                  const unsigned char* m1, unsigned int l1,
                                  const unsigned char* m2, unsigned int l2,
                                  unsigned char c3,
                                  unsigned char out[32]){
    Sha256Ctx ctx1; sha256_init(&ctx1); sha256_update(&ctx1,k_ipad,64);
    if(l1) sha256_update(&ctx1,m1,l1);
    if(l2) sha256_update(&ctx1,m2,l2);
    sha256_update(&ctx1,&c3,1);
    unsigned char inner[32]; sha256_final(&ctx1,inner);
    Sha256Ctx ctx2; sha256_init(&ctx2); sha256_update(&ctx2,k_opad,64); sha256_update(&ctx2,inner,32);
    sha256_final(&ctx2,out);
}
__global__ void sha256_key_init(const unsigned char* key, int key_len, const unsigned char* salt, int salt_len, unsigned char* out32){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>0) return;
    Sha256Ctx ctx; sha256_init(&ctx);
    if(key_len>0) sha256_update(&ctx,key,key_len);
    if(salt_len>0) sha256_update(&ctx,salt,salt_len);
    sha256_final(&ctx,out32);
}
__global__ void prf_many(const unsigned char* key32,
                         const unsigned long long* payloads,
                         const unsigned long long* offsets,
                         const unsigned long long* tkidxs,
                         int blen,
                         int n,
                         float* out){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i>=n) return;
    unsigned char k_ipad[64], k_opad[64];
    hmac_sha256_init(key32, k_ipad, k_opad);
    unsigned char msg[24];
    unsigned long long p = payloads[i];
    unsigned long long o = offsets[i];
    unsigned long long t = tkidxs[i];
    for(int b=0;b<8;b++){msg[b]=(unsigned char)(p>>(56-8*b));}
    for(int b=0;b<8;b++){msg[8+b]=(unsigned char)(o>>(56-8*b));}
    for(int b=0;b<8;b++){msg[16+b]=(unsigned char)(t>>(56-8*b));}
    int blocks=((int)(blen*4)+31)/32;
    int y_idx=0; int byte_in_word=0;
    unsigned int word=0U;
    const float denom=4294967295.0f;
    for(int bi=1; bi<=blocks; ++bi){
        unsigned char T[32];
        hmac_sha256_digest(k_ipad,k_opad,msg,24,nullptr,0,(unsigned char)bi,T);
        for(int b=0;b<32;b++){
            word=(word<<8)|(unsigned int)T[b];
            byte_in_word++;
            if(byte_in_word==4){
                out[(long long)i*blen + y_idx]=((float)word)/denom;
                y_idx++; byte_in_word=0; word=0U;
                if(y_idx==blen) return;
            }
        }
    }
}
}
'''

_mod = cp.RawModule(code=_cuda_src, options=('-std=c++11',))
_key_init = _mod.get_function('sha256_key_init')
_prf_many = _mod.get_function('prf_many')

def _build_key32_dev(ikm: bytes, salt: bytes) -> cp.ndarray:
    d_key = cp.asarray(np.frombuffer(ikm, dtype=np.uint8))
    d_salt = cp.asarray(np.frombuffer(salt, dtype=np.uint8))
    d_out = cp.empty((32,), dtype=cp.uint8)
    _key_init((1,), (1,), (d_key, np.int32(d_key.size), d_salt, np.int32(d_salt.size), d_out))
    cp.cuda.runtime.deviceSynchronize()
    return d_out

def _prf_batched_dev(key32_dev: cp.ndarray, payloads: np.ndarray, offsets: np.ndarray, tkidxs: np.ndarray, blen: int, device: str = 'cuda') -> torch.Tensor:
    n = payloads.size
    d_p = cp.asarray(payloads)
    d_o = cp.asarray(offsets)
    d_t = cp.asarray(tkidxs)
    d_out = cp.empty((n, blen), dtype=cp.float32)
    threads = 256
    blocks = (n + threads - 1)//threads
    _prf_many((blocks,), (threads,), (key32_dev, d_p, d_o, d_t, np.int32(blen), np.int32(n), d_out))
    cp.cuda.runtime.deviceSynchronize()
    return torch.utils.dlpack.from_dlpack(d_out.toDlpack()).to(device)

def getYs(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
    key32_dev = _build_key32_dev(ikm, salt)
    if context is None:
        payload = np.array([0], dtype=np.uint64)
        offset = np.array([0], dtype=np.uint64)
        tkidx = np.array([0], dtype=np.uint64)
    else:
        payload_int, r_prefix, tkIdx = context
        payload = np.array([int(payload_int)], dtype=np.uint64)
        # offset = np.array([int(len(r_prefix))], dtype=np.uint64)
        offset = np.array([len(r_prefix)], dtype=np.uint64)
        tkidx = np.array([int(tkIdx)], dtype=np.uint64)
    y = _prf_batched_dev(key32_dev, payload, offset, tkidx, blen, device='cuda')
    return y[0].tolist()

def getYsBatchedTorch(jobs: List[Tuple[bytes, bytes, List[Any], int]], device: str = 'cuda') -> torch.Tensor:
    if not jobs:
        return torch.empty((0,0), dtype=torch.float32, device=device)
    groups: dict[Tuple[bytes, bytes, int], List[int]] = {}
    for i, (s, k, c, b) in enumerate(jobs):
        groups.setdefault((s, k, b), []).append(i)
    max_b = max(b for (_, _, b) in groups.keys())
    out = torch.empty((len(jobs), max_b), dtype=torch.float32, device=device)
    for (s, k, b), idxs in groups.items():
        key32_dev = _build_key32_dev(k, s)
        payloads = np.empty((len(idxs),), dtype=np.uint64)
        offsets = np.empty((len(idxs),), dtype=np.uint64)
        tkidxs = np.empty((len(idxs),), dtype=np.uint64)
        for j, idx in enumerate(idxs):
            _, _, c, _ = jobs[idx]
            if c is None:
                payloads[j] = 0
                offsets[j] = 0
                tkidxs[j] = 0
            else:
                payload_int, r_prefix, tkIdx = c
                payloads[j] = int(payload_int)
                offsets[j] = int(len(r_prefix))
                tkidxs[j] = int(tkIdx)
        y = _prf_batched_dev(key32_dev, payloads, offsets, tkidxs, b, device=device)
        out[idxs, :b] = y
    return out[:, : (jobs[0][3] if len({j[3] for j in jobs}) == 1 else max_b)]