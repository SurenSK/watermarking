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
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from transformers import AutoModelForCausalLM, AutoTokenizer

N_PROMPTS = 1000
MAX_NEW_TOKENS = 500
MODEL_ID = "meta-llama/Llama-2-7b-hf"

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer, bitLen = None, None, None
rng = torch.Generator(device=torch.device("cpu"))
rng.manual_seed(2971215073)
table_size = 1_000_003
fixed_table = torch.randperm(table_size, generator=rng, device=rng.device).to(torch.float32) / table_size
pass


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
    c = sum(sum(i) if isinstance(i, list) else i for i in context) if context else 0
    base = int.from_bytes(salt, 'big') * int.from_bytes(ikm, 'big') * (c)
    return [float(fixed_table[(base + i) % table_size].item()) for i in range(blen)]
    # return [fixed_table[(base + i) % 1_000_003] for i in range(blen)]

def _unpack_and_call_getYs(j_args):
    """Unpacks arguments and calls getYs."""
    return getYs(*j_args)
def getYsBatched(
    jobs: List[Tuple[bytes, bytes, List[Any], int]], 
    max_workers: Optional[int] = None
) -> List[List[float]]:
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        nw = ex._max_workers if ex._max_workers else os.cpu_count() or 1
        num_chunks = nw * 1
        cs = max(1, math.ceil(len(jobs) / num_chunks))
        return list(ex.map(_unpack_and_call_getYs, jobs, chunksize=cs))

def nested_dd_list():
    return defaultdict(list)

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

        self.log = defaultdict(nested_dd_list)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / self.t, dim=-1)
        cs = F.pad(probs.cumsum(0), (1, 0))
        self.inH = self.h<self.rLambda # always invalidate self.inH at token boundary regardless of isGeneral status
        Ys = getYs(self.seed_bytes, self.seed_bytes, [self.tkIdx], bitLen) if self.inH else getYs(self.salt_bytes, self.key_bytes, [self.payload_int, self.r, self.tkIdx], bitLen)
        newTokenId=0
        for bitIdx in range(bitLen):
            p1=getP1(cs,newTokenId,bitIdx)
            newTokenId = (newTokenId<<1) | (1 if Ys[bitIdx]<p1 else 0)
            self.h += getBinaryEntropy(p1 if newTokenId&1==1 else 1-p1)

            self.log['encoder']['y'].append(Ys[bitIdx])
            self.log['encoder']['p1'].append(p1)
            self.log['encoder']['binaryEntropy'].append(getBinaryEntropy(p1 if newTokenId&1==1 else 1-p1))
            self.log['encoder']['binaryEmpiricalEntropy'].append(getBinaryEmpiricalEntropy(p1,newTokenId&1))

            if self.inH: self.r.append(newTokenId&1)
            if self.isGeneral and self.h>=self.rLambda: # sometimes invalidate self.inH at bit boundary if isGeneral status, update Ys
                self.inH = False
                Ys = getYs(self.salt_bytes, self.key_bytes, [self.payload_int, self.r, self.tkIdx], bitLen)
        # if newTokenId>len(tokenizer): newTokenId=len(tokenizer)-1
        pass
        self.tkIdx += 1
        self.log['encoder']['vocabEntropy'].append(getEntropy(probs))
        self.log['encoder']['vocabEmpiricalEntropy'].append(getEmpiricalEntropy(probs, newTokenId))
        if not self.inH: self.log['encoder']['r']=self.r
        return torch.tensor(newTokenId,dtype=torch.long,device=device)

    def decode(self, tokenIds: List[int], payloadLen: Optional[int] = 0) -> Dict:
        fullBinary = [int(b) for t in tokenIds for b in format(t, f'0{bitLen}b')]
        totalBits = len(fullBinary)
        numTokens = len(tokenIds)
        offsets = list(range(0, totalBits + 1, 1 if self.isGeneral else bitLen))

        # Payload hypotheses
        nMessages = 2**payloadLen
        payloads = list(range(nMessages))
        pass
        # Build all jobs: (salt, key, [payload, r_prefix, tkIdx], bitLen)
        jobs = []
        for m_ in payloads:
            for offset in offsets:
                r_ = fullBinary[:offset]
                for tkIdx in range(numTokens):
                    jobs.append((self.salt_bytes, self.key_bytes, [m_, r_, tkIdx], bitLen))
        print("Calling batched Y")
        Ylist = getYsBatched(jobs)
        print("Got Y")
        Ys = (
            torch.tensor(Ylist, dtype=torch.float64, device=device)
            .reshape(nMessages, len(offsets), numTokens, bitLen)
            .flatten(2, 3)  # -> (nMessages, nOffsets, totalBits)
        )
        pass
    
        B = torch.tensor(fullBinary, dtype=torch.int64, device=device).view(1, 1, totalBits)

        # Compute log-likelihood scores
        p = Ys.clamp(min=1e-9, max=1 - 1e-9)
        v = torch.where(B == 1, p, 1.0 - p)
        scores = -torch.log(v)  # (nMessages, nOffsets, totalBits)

        # Mask out acausal prefix (only bits from offset onward count)
        # tri = torch.triu(torch.ones(totalBits, totalBits, device=device, dtype=torch.float64))
        # mask = tri.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        # # MS[m, o_idx, k] = score for hypo (m, offsets[o_idx]) starting at bit k
        # masked_scores = (scores.unsqueeze(-2) * mask).sum(dim=-1) # (M, nOffsets, T)
        # Efficiently compute the suffix sum: masked_scores[m, o, k] = SUM_{j=k}^{T-1} scores[m, o, j]
        # This avoids materializing the O(T^3) intermediate tensor.
        masked_scores = torch.cumsum(scores.flip(dims=[-1]), dim=-1).flip(dims=[-1])
        offset_vals_t = torch.tensor(offsets, device=device)

        if self.isGeneral:
            # Efficiently extract the diagonal for the isGeneral=True case
            # This works because offsets[o_idx] == o_idx
            total_nll = torch.diagonal(masked_scores, dim1=-2, dim2=-1)
            
            # Pad for the final offset (offset=T, L=0, score=0)
            zero_pad = torch.zeros((nMessages, 1), device=device, dtype=torch.float64)
            total_nll = torch.cat([total_nll, zero_pad], dim=1) # Shape (M, nOffsets)
        else:
            # For isGeneral=False, offsets are sparse. We must use advanced indexing.
            # We need to extract masked_scores[m, o_idx, offsets[o_idx]] for each hypothesis.
            msg_indices = torch.arange(nMessages, device=device).view(-1, 1)
            offset_indices = torch.arange(len(offsets), device=device).view(1, -1)
            
            # offset_vals_t contains the actual bit positions [0, 15, 30, ...]
            bit_start_indices = offset_vals_t.view(1, -1)

            total_nll = masked_scores[msg_indices, offset_indices, bit_start_indices]

        # Effective watermark length L for EACH offset hypothesis
        wm_len = (totalBits - offset_vals_t).unsqueeze(0).float() # Shape (1, nOffsets)

        # Normalize: (Score - L) / sqrt(L). Add epsilon to avoid div by zero for L=0.
        norm_scores = (total_nll - wm_len) / (wm_len + 1e-9).sqrt()
        # The L=0 case (last offset) results in 0/eps = 0, which is correct.

        # Best hypothesis
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

count = defaultdict(int)
def main(idxStart, idxEnd):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    global model, tokenizer, bitLen, count

    model, tokenizer = setup()
    print(f"Running at Lambda={WM_PARAMS['rLambda']:.2f}")
    with open("prompts.txt", "r", encoding="utf-8") as f:
        dataset=[line.strip() for line in f]
    dataset=dataset[idxStart:idxEnd]
    bitLen = math.ceil(math.log2(len(tokenizer)))
    
    for i,prompt_text in tqdm(enumerate(dataset), desc="Processing Prompts"):
        i+=idxStart
        fp = f"results/experiment0_results_wm_{i}.npz"
        if not os.path.exists(fp):
            t0 = time.time()
            wmEncoder = Christ(**WM_PARAMS)
            wmIds = generateSequence(model, tokenizer, prompt_text, wmEncoder, maxLen=MAX_NEW_TOKENS)
            tWM = time.time()-t0
            t0 = time.time()
            wmRes = wmEncoder.decode(wmIds, payloadLen=PAYLOAD_LEN_DETECT)
            tWMDecode = time.time()-t0
            
            log_to_save = {}
            for key, tensor in wmEncoder.log['decoder'].items():
                log_to_save[f'decoder_{key}'] = tensor.to(torch.float32).numpy()
            for key, lst in wmEncoder.log['encoder'].items():
                log_to_save[f'encoder_{key}'] = np.array(lst, dtype=object if key == 'r' else np.float32)

            data = {
                "idx": i, "tEncode": tWM, "tDecode": tWMDecode, "isWM": True, 
                "ids": np.array(wmIds, dtype=np.int32), "params": str(WM_PARAMS),
                **wmRes, **log_to_save
            }
            count[wmRes['message']]+=1
            print(count)
            print({'tEncode': data['tEncode'], 'tDecode': data['tDecode'], 'detected': data['detected'], 'score': data['score']})
            np.savez_compressed(fp, **data)
        else:
            print(f"idx {i} wm exists")
        
        # fp = f"results/experiment0_results_nwm_{i}.npz"
        # if not os.path.exists(fp):
        #     t0 = time.time()
        #     nwmEncoder = Christ(**NWM_PARAMS)
        #     nwmIds = generateSequence(model, tokenizer, prompt_text, nwmEncoder, maxLen=MAX_NEW_TOKENS)
        #     tNWM = time.time()-t0
        #     t0 = time.time()
        #     nwmRes = nwmEncoder.decode(nwmIds, payloadLen=PAYLOAD_LEN_DETECT)
        #     tNWMDecode = time.time()-t0
            
        #     log_to_save = {}
        #     for key, tensor in nwmEncoder.log['decoder'].items():
        #         log_to_save[f'decoder_{key}'] = tensor.to(torch.float32).numpy()
        #     for key, lst in nwmEncoder.log['encoder'].items():
        #         log_to_save[f'encoder_{key}'] = np.array(lst, dtype=object if key == 'r' else np.float32)

        #     data = {
        #         "idx": i, "tEncode": tNWM, "tDecode": tNWMDecode, "isWM": False, 
        #         "ids": np.array(nwmIds, dtype=np.int32), "params": str(NWM_PARAMS),
        #         **nwmRes, **log_to_save
        #     }
        #     print({'tEncode': data['tEncode'], 'tDecode': data['tDecode'], 'detected': data['detected'], 'score': data['score']})
        #     np.savez_compressed(fp, **data)
        # else:
        #     print(f"idx {i} nwm exists")
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
