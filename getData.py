import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List, Union, Optional, Callable, Optional, Any
import math
import time
from collections import defaultdict
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import cProfile, pstats
import hmac, hashlib

N_PROMPTS = 2
MAX_NEW_TOKENS = 50
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

def getP1(p:torch.Tensor,prefix:int,bitIdx:int)->float:
    v=p.shape[-1]; b=(v-1).bit_length()
    if not v or bitIdx>=b: return 0.0
    shift=b-bitIdx; start=prefix<<shift
    if start>=v: return 0.0
    cs=torch.nn.functional.pad(p.cumsum(0),(1,0))
    s0,s1,s2=cs[start],cs[min(start+(1<<(shift-1)),v)],cs[min(start+(1<<shift),v)]
    if(total:=s2-s0)<1e-9: return 0.0
    return((s2-s1)/total).item()

# def getY(salt: bytes, ikm: bytes, context: List[Any]) -> float:
#     info=json.dumps(context,sort_keys=True,separators=(',',':')).encode('utf-8') if context is not None else None
#     hkdf=HKDF(algorithm=hashes.SHA256(),length=8,salt=salt,info=info)
#     seed=int.from_bytes(hkdf.derive(ikm),'big')
#     return seed/(2**64-1)

def getY(salt: bytes, ikm: bytes, context: List[Any]) -> float:
    info = json.dumps(context, sort_keys=True, separators=(',', ':')).encode('utf-8') if context is not None else b""
    msg = len(salt).to_bytes(4, 'big') + salt + len(info).to_bytes(4, 'big') + info
    digest = hmac.new(ikm, msg, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], 'big')
    return seed / (2**64 - 1)

class Christ:
    """
    Implements the watermarking algorithm from "Undetectable Watermarks for Language Models"
    by Christ et al. (2023). This method embeds a watermark by pseudorandomly selecting
    token bits during generation, conditioned on a secret key.
    """
    def __init__(self, key: int, salt: int, rLambda: float, random_seed: int, t: float = 1.0, payload: Optional[str] = None, scoreThreshold: Optional[float] = None, isGeneral: bool = False):
        """
        Initializes the watermarking processor.

        Args:
            key (int): The secret key for the watermark PRF.
            rLambda (float): The target entropy threshold to accumulate before watermarking begins.
            random_seed (int): A public random seed for the entropy accumulation phase.
            t (float): The temperature for softmax scaling of logits.
            payload (Optional[str]): A binary string to embed as a message.
            scoreThreshold (Optional[float]): The z-score threshold for detection. Defaults to rLambda.
            isGeneral (bool): If True, uses the "general scheme" where entropy is checked per-token.
                              If False, uses the original scheme with a one-time switch to watermarking.
        """
        self.key=key
        self.rLambda=rLambda
        self.random_seed=random_seed
        self.t=t
        self.isGeneral=isGeneral
        self.payload_int=int(payload,2)if payload is not None else 0
        self.scoreThreshold=rLambda if scoreThreshold is None else scoreThreshold
        
        # State variables
        self.h,self.r,self.tokenIdx=0.0,[],0 # h: accumulated entropy, r: public randomness sequence, tokenIdx: token counter
        self.in_entropy_phase=True # Flag indicating if we are in the initial entropy accumulation phase

        # Pre-compute byte representations of keys/seeds for the PRF
        self.key = key
        self.salt = salt
        self.seed = random_seed
        self.key_bytes=key.to_bytes(8,'big',signed=True)
        self.salt_bytes=salt.to_bytes(8,'big',signed=True)
        self.seed_bytes=random_seed.to_bytes(8,'big',signed=True)
        
        # Separate RNG for any stochastic sampling (not used in this deterministic implementation)
        self.sampling_rng=torch.Generator(device=device)
        self.sampling_rng.manual_seed(random_seed+1)
        
        # Logging dictionary
        self.log=defaultdict(lambda:defaultdict(list))
        self.log['params']={'key':key,'rLambda':rLambda,'random_seed':random_seed,'t':t,'payload':payload,'scoreThreshold':scoreThreshold,'isGeneral':isGeneral}

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        The encoding function. Given logits, it determines the next token ID
        by selecting its bits one by one according to the watermarking scheme.
        This is designed to be used as a logits processor in a generation loop.

        Args:
            logits (torch.Tensor): The raw logits from the language model for the next token.

        Returns:
            torch.Tensor: A tensor containing the selected watermarked token ID.
        """
        startTime=time.time()
        probs=torch.softmax(logits/self.t,dim=-1)
        newTokenId=0
        self.log['encode']['vocab_entropy'].append(getEntropy(probs))
        
        if self.isGeneral:
            # General Scheme: Check entropy threshold at the start of each token generation.
            generalFlag = self.h < self.rLambda
            for bitIdx in range(bitLen):
                p1=getP1(probs,newTokenId,bitIdx)
                self.log['encode']['p1'].append(p1)
                self.log['encode']['binary_entropy'].append(getBinaryEntropy(p1))
                if generalFlag:
                    # Entropy Accumulation Phase for this token: Use public seed.
                    y=getY(len(self.r).to_bytes(8,'big',signed=True),self.seed_bytes,None)
                    nextBit=1 if y<p1 else 0
                    probChosen=p1 if nextBit==1 else(1-p1)
                    score=-math.log(probChosen+1e-9)
                    self.h+=score # Accumulate entropy
                    self.r.append(nextBit) # Append bit to public randomness list
                    self.log['encode']['y'].append(y)
                    self.log['encode']['binary_empirical_entropy'].append(score)
                else:
                    # Watermarking Phase for this token: Use secret key.
                    context=[self.r,self.tokenIdx,bitIdx,self.payload_int]
                    y=getY(self.salt_bytes,self.key_bytes,context)
                    nextBit=1 if y<p1 else 0
                    probChosen=p1 if nextBit==1 else(1-p1)
                    self.log['encode']['y'].append(y)
                    self.log['encode']['binary_empirical_entropy'].append(-math.log(probChosen+1e-9))
                newTokenId=(newTokenId<<1)+nextBit
        else:
            # Original Scheme: A single switch from entropy accumulation to watermarking.
            for bitIdx in range(bitLen):
                p1=getP1(probs,newTokenId,bitIdx)
                self.log['encode']['p1'].append(p1)
                self.log['encode']['binary_entropy'].append(getBinaryEntropy(p1))
                if self.in_entropy_phase:
                    y=getY(len(self.r).to_bytes(8,'big',signed=True),self.seed_bytes,None)
                    nextBit=1 if y<p1 else 0
                    probChosen=p1 if nextBit==1 else(1-p1)
                    score=-math.log(probChosen+1e-9)
                    self.h+=score # Accumulate entropy
                    self.r.append(nextBit) # Append bit to public randomness list
                    self.log['encode']['y'].append(y)
                    self.log['encode']['binary_empirical_entropy'].append(score)
                    if self.h>=self.rLambda: self.in_entropy_phase=False # Check if we have enough entropy to switch
                else:
                    # Watermarking Phase: Use secret key.
                    context=[self.r,self.tokenIdx,bitIdx,self.payload_int]
                    y=getY(self.salt_bytes,self.key_bytes,context)
                    nextBit=1 if y<p1 else 0
                    probChosen=p1 if nextBit==1 else(1-p1)
                    self.log['encode']['y'].append(y)
                    self.log['encode']['binary_empirical_entropy'].append(-math.log(probChosen+1e-9))
                newTokenId=(newTokenId<<1)+nextBit
        
        self.tokenIdx+=1
        self.log['encode']['vocab_entropy'].append(getEntropy(probs))
        self.log['encode']['vocab_empirical_entropy'].append(getEmpiricalEntropy(probs,newTokenId))
        self.log['encode']['time'].append(time.time()-startTime)
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
