import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import time
import random
import cProfile, pstats
from dotenv import load_dotenv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import multiprocessing as mp

import torch
import msgpack
from tqdm import tqdm
import torch.nn.functional as F
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from transformers import AutoModelForCausalLM, AutoTokenizer

N_PROMPTS = 1000
MAX_NEW_TOKENS = 500
MODEL_ID = "meta-llama/Llama-2-7b-hf"
def nested_dd_list():
    return defaultdict(list)
# --- Experiment Parameters (pending clarification) ---
WM_PARAMS = {
    'key': 40,
    'salt': 41,
    'rLambda': 4.0,
    'random_seed': 42,
    't': 1.0,
    'payload': "1",
    'isGeneral': True
}
NWM_PARAMS = {**WM_PARAMS, 'rLambda': float('inf')}
PAYLOAD_LEN_DETECT = 1

def setup():
    HF_TOKEN = os.getenv("HF")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,token=HF_TOKEN,torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer

model, tokenizer, bitLen = None, None, None
device = "cuda" if torch.cuda.is_available() else "cpu"

def getEntropy(probs: torch.Tensor) -> float: return -torch.sum(probs * torch.log2(probs + 1e-9)).item()
def getEmpiricalEntropy(probs: torch.Tensor, selIdx: int) -> float: return -torch.log2(probs[selIdx] + 1e-9).item()
def getPDF(logits: torch.Tensor, t: float) -> torch.Tensor: return torch.softmax(logits/t, dim=-1)
def getBinaryEntropy(p: float) -> float: return -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0<p<1 else 0.0
def getBinaryEmpiricalEntropy(p: float, selIdx: int) -> float: return -math.log2(p+1e-9) if selIdx==1 else -math.log2(1-p+1e-9)

def getP1(cs:torch.Tensor,prefix:int,bitIdx:int)->float:
    v=cs.shape[-1]-1
    if v<=0: return 0.0
    b=(v-1).bit_length()
    if not v or bitIdx>=b: return 0.0
    shift=b-bitIdx; start=prefix<<shift
    if start>=v: return 0.0
    s0,s1,s2=cs[start],cs[min(start+(1<<(shift-1)),v)],cs[min(start+(1<<shift),v)]
    if(total:=s2-s0)<1e-9: return 0.0
    return((s2-s1)/total).item()

def getYs(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
    bytes_per = 3
    L = blen * bytes_per
    max_l = 255 * hashes.SHA256.digest_size
    if L > max_l:
        raise ValueError(f"Requested bytes {L} exceeds HKDF-SHA256 limit of {max_l}")
    info = msgpack.packb(context, use_bin_type=True) if context else b""
    hkdf = HKDF(algorithm=hashes.SHA256(), length=L, salt=salt, info=info)
    output_bytes = hkdf.derive(ikm)
    max_int = (1 << (8 * bytes_per)) - 1
    return [
        int.from_bytes(output_bytes[i*bytes_per:(i+1)*bytes_per], 'big') / max_int
        for i in range(blen)
    ]
import numpy as np
from multiprocessing import shared_memory

_BITLEN = None
_SALT_BYTES = None
_KEY_BYTES = None
_BITS_SHM = None
_BITS_ARR = None
_OUT_SHM = None
_OUT_ARR = None
_OUT_DTYPE = None

def _ys_worker_init(bits_name, bits_len, salt_bytes, key_bytes, out_name, out_shape0, out_shape1, bitlen, out_dtype_str):
    global _BITLEN, _SALT_BYTES, _KEY_BYTES, _BITS_SHM, _BITS_ARR, _OUT_SHM, _OUT_ARR, _OUT_DTYPE
    _BITLEN = bitlen
    _SALT_BYTES = salt_bytes
    _KEY_BYTES = key_bytes
    _BITS_SHM = shared_memory.SharedMemory(name=bits_name)
    _BITS_ARR = np.ndarray((bits_len,), dtype=np.uint8, buffer=_BITS_SHM.buf)
    _OUT_SHM = shared_memory.SharedMemory(name=out_name)
    _OUT_DTYPE = np.float32 if out_dtype_str == "float32" else np.float64
    _OUT_ARR = np.ndarray((out_shape0, out_shape1), dtype=_OUT_DTYPE, buffer=_OUT_SHM.buf)

def _ys_worker(job_tuple):
    j_idx, m, offset, tkIdx = job_tuple
    r_prefix = _BITS_ARR[:offset].tolist()
    y = getYs(_SALT_BYTES, _KEY_BYTES, [m, r_prefix, tkIdx], _BITLEN)
    _OUT_ARR[j_idx, :len(y)] = y
    return j_idx

def getYsBatched_shm(fullBinary, payloads, offsets, numTokens, salt_bytes, key_bytes, bitLen, max_workers=None, dtype="float32"):
    n_jobs = len(payloads) * len(offsets) * numTokens
    if n_jobs == 0:
        return np.empty((0, bitLen), dtype=np.float32 if dtype == "float32" else np.float64)
    bits_np = np.asarray(fullBinary, dtype=np.uint8)
    shm_bits = shared_memory.SharedMemory(create=True, size=bits_np.nbytes)
    bits_view = np.ndarray(bits_np.shape, dtype=bits_np.dtype, buffer=shm_bits.buf)
    bits_view[:] = bits_np
    out_dtype = np.float32 if dtype == "float32" else np.float64
    out_shape = (n_jobs, bitLen)
    shm_out = shared_memory.SharedMemory(create=True, size=int(np.prod(out_shape)) * np.dtype(out_dtype).itemsize)
    out_view = np.ndarray(out_shape, dtype=out_dtype, buffer=shm_out.buf)
    out_view.fill(0)
    jobs_iter = ((j, m, o, k) for j, (m, o, k) in enumerate((m_, o_, k_) for m_ in payloads for o_ in offsets for k_ in range(numTokens)))
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_ys_worker_init, initargs=(shm_bits.name, bits_np.shape[0], salt_bytes, key_bytes, shm_out.name, out_shape[0], out_shape[1], bitLen, "float32" if out_dtype == np.float32 else "float64")) as ex:
        nw = ex._max_workers if ex._max_workers else 1
        cs = max(1, math.ceil(n_jobs / (nw * 32)))
        list(ex.map(_ys_worker, jobs_iter, chunksize=cs))
    result = out_view.copy()
    del out_view
    shm_out.close()
    shm_out.unlink()
    del bits_view
    shm_bits.close()
    shm_bits.unlink()
    return result

class Christ:
    def __init__(self, key: int, salt: int, rLambda: float, random_seed: int, t: float = 1.0, payload: Optional[str] = None, scoreThreshold: Optional[float] = None, isGeneral: bool = True):
        self.h = 0.0
        self.inH = True
        self.r = []
        self.rLambda = rLambda
        self.scoreThreshold = rLambda if scoreThreshold is None else scoreThreshold
        self.t = t
        self.isGeneral = isGeneral
        self.tkIdx = 0
        self.payload_int = int(payload, 2) if payload is not None else 0
        self.key_bytes = key.to_bytes(8, 'big', signed=True)
        self.salt_bytes = salt.to_bytes(8, 'big', signed=True)
        self.seed_bytes = random_seed.to_bytes(8, 'big', signed=True)
        self.log = defaultdict(nested_dd_list)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / self.t, dim=-1)
        cs = F.pad(probs.cumsum(0), (1, 0))
        self.inH = self.h < self.rLambda
        Ys = getYs(self.seed_bytes, self.seed_bytes, [self.tkIdx], bitLen) if self.inH else getYs(self.salt_bytes, self.key_bytes, [self.payload_int, self.r, self.tkIdx], bitLen)
        newTokenId = 0
        for bitIdx in range(bitLen):
            p1 = getP1(cs, newTokenId, bitIdx)
            newTokenId = (newTokenId << 1) | (1 if Ys[bitIdx] < p1 else 0)
            self.h += getBinaryEntropy(p1 if newTokenId & 1 == 1 else 1 - p1)
            self.log['encoder']['y'].append(Ys[bitIdx])
            self.log['encoder']['p1'].append(p1)
            self.log['encoder']['binaryEntropy'].append(getBinaryEntropy(p1 if newTokenId & 1 == 1 else 1 - p1))
            self.log['encoder']['binaryEmpiricalEntropy'].append(getBinaryEmpiricalEntropy(p1, newTokenId & 1))
            if self.inH:
                self.r.append(newTokenId & 1)
            if self.isGeneral and self.h >= self.rLambda:
                self.inH = False
                Ys = getYs(self.salt_bytes, self.key_bytes, [self.payload_int, self.r, self.tkIdx], bitLen)
        self.tkIdx += 1
        self.log['encoder']['vocabEntropy'].append(getEntropy(probs))
        self.log['encoder']['vocabEmpiricalEntropy'].append(getEmpiricalEntropy(probs, newTokenId))
        if not self.inH:
            self.log['encoder']['r'] = self.r
        return torch.tensor(newTokenId, dtype=torch.long, device=device)

    def decode(self, tokenIds: List[int], payloadLen: Optional[int] = 0) -> Dict:
        fullBinary = [int(b) for t in tokenIds for b in format(t, f'0{bitLen}b')]
        totalBits = len(fullBinary)
        numTokens = len(tokenIds)
        offsets = list(range(0, totalBits + 1, 1 if self.isGeneral else bitLen))
        nMessages = 2 ** payloadLen
        payloads = list(range(nMessages))
        Ylist_np = getYsBatched_shm(fullBinary, payloads, offsets, numTokens, self.salt_bytes, self.key_bytes, bitLen, max_workers=None, dtype="float32")
        Ys = torch.from_numpy(Ylist_np).to(device=device, dtype=torch.float64).reshape(nMessages, len(offsets), numTokens, bitLen).flatten(2, 3)
        B = torch.tensor(fullBinary, dtype=torch.int64, device=device).view(1, 1, totalBits)
        p = Ys.clamp(min=1e-9, max=1 - 1e-9)
        v = torch.where(B == 1, p, 1.0 - p)
        scores = -torch.log(v)
        masked_scores = torch.cumsum(scores.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        offset_vals_t = torch.tensor(offsets, device=device)
        if self.isGeneral:
            total_nll = torch.diagonal(masked_scores, dim1=-2, dim2=-1)
            zero_pad = torch.zeros((nMessages, 1), device=device, dtype=torch.float64)
            total_nll = torch.cat([total_nll, zero_pad], dim=1)
        else:
            msg_indices = torch.arange(nMessages, device=device).view(-1, 1)
            offset_indices = torch.arange(len(offsets), device=device).view(1, -1)
            bit_start_indices = offset_vals_t.view(1, -1)
            total_nll = masked_scores[msg_indices, offset_indices, bit_start_indices]
        wm_len = (totalBits - offset_vals_t).unsqueeze(0).float()
        norm_scores = (total_nll - wm_len) / (wm_len + 1e-9).sqrt()
        max_val, max_idx = torch.max(norm_scores.view(-1), dim=0)
        best_message = (max_idx // len(offsets)).item()
        best_offset = offsets[(max_idx % len(offsets)).item()]
        best_score = max_val.item()
        detected = best_score > self.scoreThreshold
        message = format(best_message, f'0{payloadLen}b') if detected and payloadLen else ''
        self.log['decoder']['y'] = Ys.detach().cpu()
        self.log['decoder']['scores'] = scores.detach().cpu()
        self.log['decoder']['normScores'] = norm_scores.detach().cpu()
        return {'detected': detected, 'score': best_score, 'n_star': best_offset, 'message': message}


@torch.no_grad()
def generateSequence(model, tokenizer, prompt: str, algo, maxLen: int):
    tokInput = tokenizer(prompt, return_tensors='pt').to(device)
    inputIds = tokInput.input_ids; initLen = inputIds.shape[1]
    cache, lastToken = None, inputIds
    for _ in range(maxLen):
        outputs = model(input_ids=lastToken, past_key_values=cache, use_cache=True)
        logits, cache = outputs.logits[0, -1, :], outputs.past_key_values
        logits[tokenizer.eos_token_id] = -float('inf') 
        newToken = algo(logits).unsqueeze(0)
        if newToken.item() == tokenizer.eos_token_id: break
        inputIds = torch.cat([inputIds, newToken.unsqueeze(0)], dim=1)
        lastToken = newToken.unsqueeze(0)
    return inputIds.squeeze(0)[initLen:].tolist()

def main(idxStart, idxEnd):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    global model, tokenizer, bitLen, counter
    model, tokenizer = setup()
    print(f"Running at Lambda={WM_PARAMS['rLambda']:.2f}")
    with open("prompts.txt", "r", encoding="utf-8") as f:
        dataset=[line.strip() for line in f]
    dataset=dataset[idxStart:idxEnd]
    bitLen = math.ceil(math.log2(len(tokenizer)))
    counter = defaultdict(int)
    for i,prompt_text in tqdm(enumerate(dataset), desc="Processing Prompts"):
        i+=idxStart
        fp = f"results/experiment0_results_wm_{i}.pt"
        if not os.path.exists(fp):
            t0 = time.time()
            wmEncoder = Christ(**WM_PARAMS)
            wmIds = generateSequence(model, tokenizer, prompt_text, wmEncoder, maxLen=MAX_NEW_TOKENS)
            tWM = time.time()-t0
            t0 = time.time()
            wmRes = wmEncoder.decode(wmIds, PAYLOAD_LEN_DETECT)
            tWMDecode = time.time()-t0
            print(wmRes)
            data = {"idx": i, "tEncode": tWM, "tDecode": tWMDecode, "isWM": True, "ids": wmIds, "data": wmEncoder.log, "decodeRes":wmRes, "params": WM_PARAMS}
            pass
            counter[wmRes['message']]+=1
            # torch.save(data, fp)
            print(counter)
        else:
            print(f"idx {i} wm exists")
        
        # fp = f"results/experiment0_results_nwm_{i}.pt"
        # if not os.path.exists(fp):
        #     t0 = time.time()
        #     nwmEncoder = Christ(**NWM_PARAMS)
        #     nwmIds = generateSequence(model, tokenizer, prompt_text, nwmEncoder, maxLen=MAX_NEW_TOKENS)
        #     tNWM = time.time()-t0
        #     t0 = time.time()
        #     nwmRes = nwmEncoder.decode(nwmIds)
        #     tNWMDecode = time.time()-t0
        #     print(nwmRes)
        #     data = {"idx": i, "tEncode": tNWM, "tDecode": tNWMDecode, "isWM": False, "ids": nwmIds, "data": nwmEncoder.log, "decodeRes":nwmRes, "params": NWM_PARAMS}
        #     torch.save(data, f"results/experiment0_results_nwm_{i}.pt")
        # else:
        #     print(f"idx {i} nwm exists")
    print('final')
    print(counter)
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import sys
    if len(sys.argv)>1:
        a = int(sys.argv[1])
        b = int(sys.argv[2])
    else:
        a = 0
        b = 100
    print(f"Starting prompts#{a}-{b}")
    main(a,b)
