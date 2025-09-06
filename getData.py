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

N_PROMPTS = 2
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
    'isGeneral': False
}
NWM_PARAMS = {**WM_PARAMS, 'rLambda': float('inf')}
PAYLOAD_LEN_DETECT = 0

def setup():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)
    return model, tokenizer, dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer, dataset = setup()
dataset = iter(dataset)
bitLen = math.ceil(math.log2(len(tokenizer)))

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

def getYs(salt: bytes, ikm: bytes, context: List[Any], blen: int) -> List[float]:
    L=blen*8
    max_l=255*hashes.SHA256.digest_size
    if L > max_l:
        raise ValueError(f"Requested bytes {L} exceeds HKDF-SHA256 limit of {max_l}")
    info=msgpack.packb(context, use_bin_type=True) if context else b""
    hkdf=HKDF(algorithm=hashes.SHA256(),length=L,salt=salt,info=info)
    output_bytes=hkdf.derive(ikm)
    divisor=2**64-1
    chunk_size=8
    return [
        int.from_bytes(output_bytes[i*chunk_size:(i+1)*chunk_size],'big')/divisor
        for i in range(blen)
    ]

def getYs_batch_mp(
    jobs: List[Tuple[bytes, bytes, List[Any], int]], 
    max_workers: Optional[int] = None
) -> List[List[float]]:
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        nw = ex._max_workers if ex._max_workers else 1
        cs = max(1, math.ceil(len(jobs) / nw))
        return list(ex.map(lambda j_args: getYs(*j_args), jobs, chunksize=cs))

class Christ:
    def __init__(self, key: int, salt: int, rLambda: float, random_seed: int, t: float = 1.0, payload: Optional[str] = None, scoreThreshold: Optional[float] = None, isGeneral: bool = False):
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

        self.log = defaultdict(lambda: defaultdict(list))

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / self.t, dim=-1)
        cs = F.pad(probs.cumsum(0), (1, 0))
        self.inH = self.h<self.rLambda # always invalidate self.inH at token boundary regardless of isGeneral status
        Ys = getYs(self.seed_bytes, self.seed_bytes, None, bitLen) if self.inH else getYs(self.salt_bytes, self.key_bytes, [self.r, self.tkIdx], bitLen)
        newTokenId=0
        for bitIdx in range(bitLen):
            p1=getP1(cs,newTokenId,bitIdx)
            newTokenId = (newTokenId<<1) | Ys[bitIdx]<p1
            self.h += getBinaryEntropy(p1 if newTokenId&1==1 else 1-p1)
            if self.isGeneral and self.h>self.rLambda: # sometimes invalidate self.inH at bit boundary if isGeneral status, update Ys
                self.inH = False
                Ys = getYs(self.salt_bytes, self.key_bytes, [self.r, self.tkIdx], bitLen)
        self.tkIdx += 1
        return torch.tensor(newTokenId,dtype=torch.long,device=device)

    def decode(self, tokenIds: List[int], payloadLen: Optional[int] = None) -> Dict:
        """
        Detects the watermark in a sequence of token IDs.

        Args:
            tokenIds (List[int]): The list of token IDs to analyze.
            payloadLen (Optional[int]): The length of the message to search for. If None or 0,
                                        assumes no message is embedded.

        Returns:
            Dict: A dictionary containing detection results: 'detected' (bool),
                  'score' (float), 'n_star' (int), and 'message' (str).
        """
        startTime=time.time()
        if not tokenIds: return {'detected': False, 'score': -math.inf, 'n_star': 0, 'message': ''}
        
        # Convert all tokens to a single sequence of bits
        fullBinary=[int(b) for t in tokenIds for b in format(t,f'0{bitLen}b')]
        totalBits=len(fullBinary)
        
        # Define the search space for n*, the number of initial bits assumed to be random (not watermarked)
        n_star_step=bitLen if self.isGeneral else 1
        search_range=range(0,totalBits+1,n_star_step)
        num_offsets=len(search_range)
        
        # Define the search space for the embedded message
        num_messages=2**payloadLen if payloadLen is not None and payloadLen>0 else 1
        
        # Tensors for logging and analysis
        y_tensor=torch.zeros((totalBits,num_messages,num_offsets))
        scores_tensor=torch.zeros((totalBits,num_messages,num_offsets))
        norm_scores_matrix=torch.full((num_messages,num_offsets),-math.inf)
        
        best_overall_score,best_message_int,best_n_star=-math.inf,-1,0
        
        # Exhaustive search over all possible payloads and n* offsets
        for p_idx,payload_hyp_int in enumerate(range(num_messages)):
            for o_idx,offset in enumerate(tqdm(search_range, desc=f"Sweeping n* for payload {payload_hyp_int}", leave=False)):
                currentScore,rSeq=0.0,fullBinary[:offset]
                
                # Calculate the log-likelihood score for the watermarked portion of the bit sequence
                for bitPos in range(offset,totalBits):
                    tokenIdx,bitIdx=bitPos//bitLen,bitPos%bitLen
                    
                    # Reconstruct the PRF context for this bit position
                    # context=torch.tensor(rSeq+[tokenIdx,bitIdx,payload_hyp_int],dtype=torch.float64)
                    context=[self.r,self.tokenIdx,bitIdx,self.payload_int]
                    # yPrf=getYTGEN(self.salt,self.key,context)
                    yPrf=getY(self.salt_bytes,self.key_bytes,context) # Re-generate the PRF value
                    
                    obsBit=fullBinary[bitPos]
                    v=yPrf if obsBit==1 else(1-yPrf) # Probability of observing this bit under the watermark hypothesis
                    bit_score=-math.log(v+1e-9)
                    currentScore+=bit_score
                    
                    y_tensor[bitPos,p_idx,o_idx]=yPrf
                    scores_tensor[bitPos,p_idx,o_idx]=bit_score
                
                # Normalize the score to get a z-score like statistic
                wmLen=totalBits-offset
                normScore=(currentScore-wmLen)/math.sqrt(wmLen)if wmLen>0 else 0.0
                norm_scores_matrix[p_idx,o_idx]=normScore
                
                # Keep track of the best hypothesis found so far
                if normScore>best_overall_score:
                    best_overall_score,best_message_int,best_n_star=normScore,payload_hyp_int,offset
        
        # Make a detection decision based on the threshold
        detected=best_overall_score>self.scoreThreshold
        message=""
        if detected and payloadLen is not None and payloadLen>0:
            message=format(best_message_int,f'0{payloadLen}b')
            
        self.log['decode']['time']=time.time()-startTime
        self.log['decode']['y_tensor']=y_tensor
        self.log['decode']['scores_tensor']=scores_tensor
        self.log['decode']['norm_scores_matrix']=norm_scores_matrix
        return {'detected': detected, 'score': best_overall_score, 'n_star': best_n_star, 'message': message}

@torch.no_grad()
def generateSequence(model, tokenizer, prompt: str, algo, maxLen: int):
    tokInput = tokenizer(prompt, return_tensors='pt').to(device)
    inputIds = tokInput.input_ids; initLen = inputIds.shape[1]
    cache, lastToken = None, inputIds
    for _ in range(maxLen):
        outputs = model(input_ids=lastToken, past_key_values=cache, use_cache=True)
        logits, cache = outputs.logits[0, -1, :], outputs.past_key_values
        newToken = algo(logits).unsqueeze(0)
        if newToken.item() == tokenizer.eos_token_id: break
        inputIds = torch.cat([inputIds, newToken.unsqueeze(0)], dim=1)
        lastToken = newToken.unsqueeze(0)
    return inputIds.squeeze(0)[initLen:].tolist()

def main():
    data = []
    for i in tqdm(range(N_PROMPTS), desc="Processing Prompts"):
        prompt_text = next(dataset)['text'][:256] # Truncate long prompts

        wmEncoder = Christ(**WM_PARAMS)
        wmIds = generateSequence(model, tokenizer, prompt_text, wmEncoder, maxLen=MAX_NEW_TOKENS)
        wmDecoder = Christ(**WM_PARAMS)
        wmRes = wmDecoder.decode(wmIds, payloadLen=PAYLOAD_LEN_DETECT)

        nwmEncoder = Christ(**NWM_PARAMS)
        nwmIds = generateSequence(model, tokenizer, prompt_text, nwmEncoder, maxLen=MAX_NEW_TOKENS)
        nwmDecoder = Christ(**WM_PARAMS) # Detector uses WM key
        nwmRes = nwmDecoder.decode(nwmIds, payloadLen=PAYLOAD_LEN_DETECT)

        data.append({'prompt_id':i,'encoder_log':wmEncoder.log,'decoder_log':wmDecoder.log,'is_wm':True,'detected':wmRes['detected']})
        data.append({'prompt_id':i,'encoder_log':nwmEncoder.log,'decoder_log':nwmDecoder.log,'is_wm':False,'detected':nwmRes['detected']})

    # torch.save(data, "experiment_results.pt")

if __name__ == "__main__":
    cProfile.run('main()', 'output.prof')

    stats = pstats.Stats('output.prof')
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20) # Print top 20 cumulative time functions
