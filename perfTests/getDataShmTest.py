import math
import time
import random
import cProfile, pstats
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import msgpack
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from transformers import AutoModelForCausalLM, AutoTokenizer

N_PROMPTS = 5
MAX_NEW_TOKENS = 500
MODEL_ID = "meta-llama/Llama-2-7b-hf"
DATASET_ID = "allenai/c4"
DATASET_CONFIG = "en.noblocklist"

# --- Experiment Parameters (pending clarification) ---
WM_PARAMS = {
    'key': 40,
    'salt': 41,
    'rLambda': 4.0,
    'random_seed': 42,
    't': 1.0,
    'payload': None,
    'isGeneral': True
}
NWM_PARAMS = {**WM_PARAMS, 'rLambda': float('inf')}
PAYLOAD_LEN_DETECT = 0

def setup():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)
    return model, tokenizer, dataset

model, tokenizer, dataset, bitLen = None, None, None, None
device = "cuda" if torch.cuda.is_available() else "cpu"

def getEntropy(probs: torch.Tensor) -> float: return -torch.sum(probs * torch.log(probs + 1e-9)).item()
def getEmpiricalEntropy(probs: torch.Tensor, selIdx: int) -> float: return -torch.log(probs[selIdx] + 1e-9).item()
def getPDF(logits: torch.Tensor, t: float) -> torch.Tensor: return torch.softmax(logits/t, dim=-1)
def getBinaryEntropy(p: float) -> float: return -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0<p<1 else 0.0

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
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from cryptography.hazmat.primitives import hmac as crypto_hmac
from multiprocessing import shared_memory
import numpy as np

def extract_prk(salt: bytes, ikm: bytes) -> bytes:
    h = crypto_hmac.HMAC(salt, hashes.SHA256())
    h.update(ikm)
    return h.finalize()

def getYs_from_prk(prk: bytes, context: List[Any], blen: int) -> List[float]:
    L = blen * 8
    max_l = 255 * hashes.SHA256.digest_size
    if L > max_l:
        raise ValueError(f"Requested bytes {L} exceeds HKDF-SHA256 limit of {max_l}")
    info = msgpack.packb(context, use_bin_type=True) if context else b""
    hkdf_expand = HKDFExpand(algorithm=hashes.SHA256(), length=L, info=info)
    output_bytes = hkdf_expand.derive(prk)
    bytesPer = 8
    return [
        int.from_bytes(output_bytes[i*bytesPer:(i+1)*bytesPer], 'big') / (2**64 - 1)
        for i in range(blen)
    ]

_PRK = None
_BITLEN = None
_SHM_NAME = None
_SHM_LEN = None

def _init_worker(prk_bytes: bytes, blen: int, shm_name: Optional[str] = None, shm_len: Optional[int] = None):
    global _PRK, _BITLEN, _SHM_NAME, _SHM_LEN
    _PRK = prk_bytes
    _BITLEN = blen
    _SHM_NAME = shm_name
    _SHM_LEN = shm_len

def _worker_getYs_context(context: List[Any]) -> List[float]:
    return getYs_from_prk(_PRK, context, _BITLEN)

def _worker_getYs_from_shm(job: Tuple[int, int, int]) -> List[float]:
    m_, offset, tkIdx = job
    shm = shared_memory.SharedMemory(name=_SHM_NAME)
    arr = np.ndarray((_SHM_LEN,), dtype=np.uint8, buffer=shm.buf)
    r_list = arr[:offset].tolist()
    context = [m_, r_list, tkIdx]
    y = getYs_from_prk(_PRK, context, _BITLEN)
    shm.close()
    return y

def getYsBatched_prk(context_jobs: List[List[Any]], prk: bytes, blen: int, max_workers: Optional[int] = None) -> List[List[float]]:
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(prk, blen)) as ex:
        nw = ex._max_workers if ex._max_workers else 1
        cs = max(1, math.ceil(len(context_jobs) / (nw * 32)))
        return list(ex.map(_worker_getYs_context, context_jobs, chunksize=cs))

def getYsBatched_prk_shm(jobs: List[Tuple[int, int, int]], prk: bytes, blen: int, shm_name: str, shm_len: int, max_workers: Optional[int] = None) -> List[List[float]]:
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(prk, blen, shm_name, shm_len)) as ex:
        nw = ex._max_workers if ex._max_workers else 1
        cs = max(1, math.ceil(len(jobs) / (nw * 32)))
        return list(ex.map(_worker_getYs_from_shm, jobs, chunksize=cs))

class Christ:
    def __init__(self, key: int, salt: int, rLambda: float, random_seed: int, t: float = 1.0, payload: Optional[str] = None, scoreThreshold: Optional[float] = None, isGeneral: bool = True):
        self.h = 0.0
        self.inH = True
        self.r: List[int] = []
        self.rLambda = rLambda
        self.scoreThreshold = rLambda if scoreThreshold is None else scoreThreshold
        self.t = t
        self.isGeneral = isGeneral
        self.tkIdx = 0
        self.payload_int = int(payload, 2) if payload is not None else 0
        self.key_bytes = key.to_bytes(8, 'big', signed=True)
        self.salt_bytes = salt.to_bytes(8, 'big', signed=True)
        self.seed_bytes = random_seed.to_bytes(8, 'big', signed=True)
        self.prk_seed = extract_prk(self.seed_bytes, self.seed_bytes)
        self.prk_main = extract_prk(self.salt_bytes, self.key_bytes)
        self.log = defaultdict(lambda: defaultdict(list))

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / self.t, dim=-1)
        cs = F.pad(probs.cumsum(0), (1, 0))
        self.inH = self.h < self.rLambda
        Ys = getYs_from_prk(self.prk_seed, None, bitLen) if self.inH else getYs_from_prk(self.prk_main, [self.payload_int, self.r, self.tkIdx], bitLen)
        newTokenId = 0
        for bitIdx in range(bitLen):
            p1 = getP1(cs, newTokenId, bitIdx)
            newTokenId = (newTokenId << 1) | (1 if Ys[bitIdx] < p1 else 0)
            self.h += getBinaryEntropy(p1 if newTokenId & 1 == 1 else 1 - p1)
            self.log['encoder']['y'].append(Ys[bitIdx])
            self.log['encoder']['p1'].append(p1)
            self.log['encoder']['binaryEntropy'].append(getBinaryEntropy(p1 if newTokenId & 1 == 1 else 1 - p1))
            if self.inH:
                self.r.append(newTokenId & 1)
            if self.isGeneral and self.h >= self.rLambda:
                self.inH = False
                Ys = getYs_from_prk(self.prk_main, [self.payload_int, self.r, self.tkIdx], bitLen)
        self.tkIdx += 1
        self.log['encoder']['vocabEntropy'].append(getEntropy(probs))
        self.log['encoder']['vocabEmpiricalEntropy'].append(getEmpiricalEntropy(probs, newTokenId))
        return torch.tensor(newTokenId, dtype=torch.long, device=device)

    def decode(self, tokenIds: List[int], payloadLen: Optional[int] = 0) -> Dict:
        fullBinary = [int(b) for t in tokenIds for b in format(t, f'0{bitLen}b')]
        totalBits = len(fullBinary)
        numTokens = len(tokenIds)
        offsets = list(range(0, totalBits + 1, 1 if self.isGeneral else bitLen))
        nMessages = 2 ** payloadLen
        payloads = list(range(nMessages))
        prk = extract_prk(self.salt_bytes, self.key_bytes)
        arr = np.array(fullBinary, dtype=np.uint8)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shm_arr[:] = arr[:]
        jobs = []
        for m_ in payloads:
            for offset in offsets:
                for tkIdx in range(numTokens):
                    jobs.append((m_, offset, tkIdx))
        t0 = time.time()
        Ylist = getYsBatched_prk_shm(jobs, prk, bitLen, shm.name, arr.shape[0])
        shm.close()
        shm.unlink()
        Ys = torch.tensor(Ylist, dtype=torch.float64, device=device).reshape(nMessages, len(offsets), numTokens, bitLen).flatten(2, 3)
        print(f"{time.time() - t0:.2f}s")
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

def main():
    global model, tokenizer, dataset, bitLen
    model, tokenizer, dataset = setup()
    dataset = iter(dataset)
    bitLen = math.ceil(math.log2(len(tokenizer)))
    data = []
    for i in tqdm(range(N_PROMPTS), desc="Processing Prompts"):
        t0 = time.time()
        prompt_text = next(dataset)['text'][:256] # Truncate long prompts
        pass

        wmEncoder = Christ(**WM_PARAMS)
        wmIds = generateSequence(model, tokenizer, prompt_text, wmEncoder, maxLen=MAX_NEW_TOKENS)
        pass
        # print(wmEncoder.log['encoder']['y'][:20])
        # print(len(wmEncoder.r))
        # print(wmEncoder.r)
        # pass
        wmDecoder = Christ(**WM_PARAMS)
        wmRes = wmDecoder.decode(wmIds, payloadLen=PAYLOAD_LEN_DETECT)
        print(f"{time.time()-t0:.2f}s/prompt {wmRes}")
        pass
        # nwmEncoder = Christ(**NWM_PARAMS)
        # nwmIds = generateSequence(model, tokenizer, prompt_text, nwmEncoder, maxLen=MAX_NEW_TOKENS)
        # nwmDecoder = Christ(**WM_PARAMS) # Detector uses WM key
        # nwmRes = nwmDecoder.decode(nwmIds, payloadLen=PAYLOAD_LEN_DETECT)

        # data.append({'prompt_id':i,'encoder_log':wmEncoder.log,'decoder_log':wmDecoder.log,'is_wm':True,'detected':wmRes['detected']})
        # data.append({'prompt_id':i,'encoder_log':nwmEncoder.log,'decoder_log':nwmDecoder.log,'is_wm':False,'detected':nwmRes['detected']})
        pass

    # torch.save(data, "experiment_results.pt")

if __name__ == "__main__":
    cProfile.run('main()', 'mBitmProcHKDF.prof')

    stats = pstats.Stats('mBitmProcHKDF.prof')
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20) # Print top 20 cumulative time functions
