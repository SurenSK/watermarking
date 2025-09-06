from typing import Dict, List, Union, Optional, Callable, Optional
import random
import os
import sys
import time
import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import hashlib
from scipy.special import gammainc, gammaincc
from tqdm import tqdm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from collections import defaultdict

print("Setting up model and tokenizer...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
modelID = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(modelID)
tokenizer.padding_side = 'left'
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(modelID, torch_dtype=torch.float16).to(device)
vocabSize = len(tokenizer)
bitLen = math.ceil(math.log2(vocabSize))
print(f"Setup complete. Using device: {device}\n")

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

def getY(salt: bytes, ikm: bytes, context: torch.Tensor) -> float:
    info = context.cpu().numpy().tobytes() if context is not None else None
    hkdf = HKDF(algorithm=hashes.SHA256(), length=8, salt=salt, info=info)
    seed = int.from_bytes(hkdf.derive(ikm), 'big')
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
                    context=torch.tensor(self.r+[self.tokenIdx,bitIdx,self.payload_int],dtype=torch.float64)
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
                    # Entropy Accumulation Phase: Use public seed.
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
                    context=torch.tensor(self.r+[self.tokenIdx,bitIdx,self.payload_int],dtype=torch.float64)
                    y=getY(self.salt_bytes,self.key_bytes,context)
                    nextBit=1 if y<p1 else 0
                    probChosen=p1 if nextBit==1 else(1-p1)
                    self.log['encode']['y'].append(y)
                    self.log['encode']['binary_empirical_entropy'].append(-math.log(probChosen+1e-9))
                newTokenId=(newTokenId<<1)+nextBit
        
        self.tokenIdx+=1
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
                    context=torch.tensor(rSeq+[tokenIdx,bitIdx,payload_hyp_int],dtype=torch.float64)
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

class _OZ_DynamicEcc:
    def decode(self, y: List[int]) -> str:
        d = []
        for s in y:
            if s in [0, 1]: d.append(str(s))
            elif s == 2 and d: d.pop()
        return "".join(d)

    def getNextSymbol(self, p: str, rc: List[int]) -> int:
        dsf = self.decode(rc); last = 0
        while last < len(p) and last < len(dsf) and p[last] == dsf[last]: last += 1
        if len(dsf) - last > 0: return 2
        elif last < len(p): return int(p[last])
        else: return 0

class OZ:
    def __init__(self, key: int, rLambda: float, random_seed: int, payload: str = "", threshold: float = 2.0, t: float=1.0):
        self.key, self.rLambda, self.random_seed = key, rLambda, random_seed
        self.payload, self.threshold, self.t = payload, threshold, t
        self.h, self.r, self.tokenIdx = 0.0, [], 0
        self.ecc, self.code = _OZ_DynamicEcc(), []
        self.nextSymbol = self.ecc.getNextSymbol(self.payload, [])
        self.scores, self.scoreLen = torch.zeros(3, device=device), 0.0
        self.key_bytes = key.to_bytes(8, 'big', signed=True)
        self.seed_bytes = random_seed.to_bytes(8, 'big', signed=True)
        self.salt_bytes = b"OZ-WATERMARK-SALT"
        self.log = defaultdict(List)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits/self.t, dim=-1); newTokenId = 0
        for bitIdx in range(bitLen):
            p1 = getP1(probs, newTokenId, bitIdx)
            if self.h < self.rLambda:
                context = torch.tensor([len(self.r)], dtype=torch.float64)
                y = getY(self.seed_bytes, self.seed_bytes, context)
                nextBit = 1 if y < p1 else 0
                probChosen = p1 if nextBit == 1 else (1 - p1)
                self.h += -math.log(probChosen + 1e-9)
                self.r.append(nextBit)
            else:
                yPrfVals = torch.tensor([
                    getY(self.salt_bytes, self.key_bytes, torch.tensor(self.r + [self.tokenIdx, bitIdx, s])) 
                    for s in range(3)
                ], device=device)
                ySample = yPrfVals[self.nextSymbol]
                nextBit = 1 if ySample < p1 else 0
                self.scoreLen += 1.0
                v = torch.where(torch.tensor(nextBit, device=device) == 1, yPrfVals, 1 - yPrfVals)
                self.scores += -torch.log(v + 1e-9)
                if self.scoreLen > 0:
                    normScores = (self.scores - self.scoreLen) / torch.sqrt(torch.tensor(self.scoreLen, device=device))
                    if (passed := torch.where(normScores > self.threshold)[0]).size(0) > 0:
                        self.code.append(passed[0].item())
                        self.scores.zero_(); self.scoreLen = 0.0
                        self.nextSymbol = self.ecc.getNextSymbol(self.payload, self.code)
            newTokenId = (newTokenId << 1) + nextBit
        self.tokenIdx += 1
        return torch.tensor(newTokenId, dtype=torch.long, device=device)
    
    def decode(self, tokenIds: List[int]) -> str:
        if not tokenIds: return ""
        totalBits = len(tokenIds) * bitLen
        fullBinary = [int(b) for t in tokenIds for b in format(t, f'0{bitLen}b')]
        best_message = ""
        # Assign tqdm to a variable to update its display
        pbar = tqdm(range(totalBits + 1), desc="Sweeping OZ offsets")
        for n_star in pbar:
            rSeq = fullBinary[:n_star]
            scores, scoreLen, retrieved = torch.zeros(3, device=device), 0.0, []
            bitSeq = fullBinary[n_star:]
            for bitPos, bit in enumerate(bitSeq):
                scoreLen += 1.0
                absBitPos = bitPos + n_star
                tokenIdx, bitIdx = absBitPos // bitLen, absBitPos % bitLen
                yPrfs = torch.tensor([
                    getY(self.salt_bytes, self.key_bytes, torch.tensor(rSeq + [tokenIdx, bitIdx, s])) 
                    for s in range(3)
                ], device=device)
                v = torch.where(torch.tensor(bit, device=device) == 1, yPrfs, 1 - yPrfs)
                scores += -torch.log(v + 1e-9)
                if scoreLen > 0:
                    normScores = (scores - scoreLen) / math.sqrt(scoreLen)
                    if (passed := torch.where(normScores > self.threshold)[0]).size(0) > 0:
                        retrieved.append(passed[0].item())
                        scores.zero_(); scoreLen = 0.0
            decoded_message = self.ecc.decode(retrieved)
            
            if len(decoded_message) > len(best_message):
                best_message = decoded_message
                print(f"  -> New best message found at n*={n_star}: '{best_message}'")
            
            if best_message:
                pbar.set_postfix_str(f"Best='{best_message}'")
        return best_message

def b2g(n: int) -> int: return n ^ (n >> 1)
def g2b(n: int) -> int:
    mask = n >> 1
    while mask != 0: n = n ^ mask; mask >>= 1
    return n

blen = math.ceil(math.log2(len(tokenizer)))

class DISC:
    def __init__(self, key: int, random_seed: int, payloadBits: str = "", rLambda: float = 4.0, t: float=1.0, contextChecking: bool = False):
        self.key, self.rLambda, self.t = key, rLambda, t
        self.random_seed, self.contextChecking = random_seed, contextChecking
        self.h, self.r, self.tokenIdx = 0.0, [], 0
        self.payloadBits, self.msgLen = payloadBits, len(payloadBits)
        self.msgSpaceSz = 2**self.msgLen if self.msgLen > 0 else 1
        self.delta = 1.0 / self.msgSpaceSz
        self.deltaM = b2g(int(payloadBits, 2) if payloadBits else 0) * self.delta
        self.key_bytes = key.to_bytes(8, 'big', signed=True)
        self.seed_bytes = random_seed.to_bytes(8, 'big', signed=True)
        self.salt_bytes = b"DISC-WATERMARK-SALT"
        if self.contextChecking: self.seen_contexts = set()
        self.log = defaultdict(List)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits/self.t, dim=-1); newTokenId = 0
        for bitIdx in range(bitLen):
            p1 = getP1(probs, newTokenId, bitIdx)
            use_random_sampling = True
            if self.h < self.rLambda:
                pass # Use random sampling
            else:
                context_tuple = tuple(self.r + [self.tokenIdx, bitIdx]) # TODO replace token idx with last h tokens
                if self.contextChecking and context_tuple in self.seen_contexts:
                    pass # Context repeated, use random sampling
                else:
                    if self.contextChecking: self.seen_contexts.add(context_tuple)
                    use_random_sampling = False
            
            if use_random_sampling:
                context = torch.tensor([len(self.r)], dtype=torch.float64)
                y = getY(self.seed_bytes, self.seed_bytes, context)
                nextBit = 1 if y < p1 else 0
                if self.h < self.rLambda:
                    probChosen = p1 if nextBit == 1 else (1 - p1)
                    self.h += -math.log(probChosen + 1e-9)
                    self.r.append(nextBit)
            else:
                context = torch.tensor(list(context_tuple), dtype=torch.float64)
                y = getY(self.salt_bytes, self.key_bytes, context)
                start, end = self.deltaM, self.deltaM + p1
                nextBit = 1 if (start <= y < end if end <= 1.0 else y >= start or y < (end - 1.0)) else 0
            newTokenId = (newTokenId << 1) + nextBit
        self.tokenIdx += 1
        return torch.tensor(newTokenId, dtype=torch.long, device=device)
    
    def decode(self, tokenIds: List[int], msgLenHyp: int) -> Dict:
        if not tokenIds: return {'detected': False, 'message': ''}
        totalBits = len(tokenIds) * blen
        fullBinarySequence = [int(b) for t in tokenIds for b in format(t, f'0{blen}b')]
        msgSpaceHyp = 2 ** msgLenHyp

        def _pvalue_for_hyp(nStar, mPrime):
            if not (0 <= mPrime < msgSpaceHyp): return 1.0, 0.0
            rSeq, deltaMPrime = fullBinarySequence[:nStar], mPrime * (1.0 / msgSpaceHyp)
            currentScore, fresh_bit_count = 0.0, 0
            if self.contextChecking: seen_contexts_hyp = set()
            
            for bitPos in range(nStar, totalBits):
                tokenIdx, bitIdx = bitPos // blen, bitPos % blen
                context_tuple = tuple(rSeq + [tokenIdx, bitIdx])
                if self.contextChecking:
                    if context_tuple in seen_contexts_hyp: continue
                    seen_contexts_hyp.add(context_tuple)
                
                fresh_bit_count += 1
                context = torch.tensor(list(context_tuple), dtype=torch.float64)
                yPrf = getY(self.salt_bytes, self.key_bytes, context)
                w = fullBinarySequence[bitPos]
                v = (yPrf - deltaMPrime + 1.0) if w == 1 else (deltaMPrime - yPrf + 1.0)
                currentScore += -math.log(v % 1.0 + 1e-9)

            if fresh_bit_count <= 0: return 1.0, 0.0
            return gammaincc(fresh_bit_count, currentScore), currentScore

        def _process_nstar(nStar):
            numCoarseSteps, coarseStepSize = 32, max(1, msgSpaceHyp // 32)
            bestCoarseP, bestCoarseM, tested = 1.0, 0, 0
            coarseCandidates = list(range(0, msgSpaceHyp, coarseStepSize))
            if msgSpaceHyp - 1 not in coarseCandidates: coarseCandidates.append(msgSpaceHyp - 1)
            for mPrimeCoarse in coarseCandidates:
                pVal, _ = _pvalue_for_hyp(nStar, mPrimeCoarse); tested += 1
                if pVal < bestCoarseP: bestCoarseP, bestCoarseM = pVal, mPrimeCoarse
            windowRadius = coarseStepSize
            fineStart, fineEnd = max(0, bestCoarseM - windowRadius), min(msgSpaceHyp, bestCoarseM + windowRadius)
            bestPValHere, bestMHere = bestCoarseP, bestCoarseM
            for mPrimeFine in range(fineStart, fineEnd):
                if mPrimeFine in coarseCandidates and mPrimeFine != bestCoarseM: continue
                pVal, _ = _pvalue_for_hyp(nStar, mPrimeFine); tested += 1
                if pVal < bestPValHere: bestPValHere, bestMHere = pVal, mPrimeFine
            return bestPValHere, nStar, bestMHere, tested
        
        results = [_process_nstar(n) for n in range(totalBits + 1)]
        bestPValue, bestNStar, bestMStar, _ = min(results, key=lambda x: x[0]) if results else (1.0, 0, 0, 0)
        hypothesesTestedCount = sum(r[3] for r in results)
        globalPValue = 1.0 - (1.0 - bestPValue) ** hypothesesTestedCount
        detected = globalPValue < 1e-3
        decodedBits = format(g2b(bestMStar), f'0{msgLenHyp}b') if detected else ""
        return {'detected': detected, 'message': decodedBits, 'p_value': globalPValue, 'n_star': bestNStar}

class DISCP2:
    def __init__(self, key: int, watermarkMaskKey: int, random_seed: int, payloadBits: str = "", p2: float = 0.25, t: float = 1.0, windowSz: int = 8, contextChecking: bool = False):
        self.key, self.p2, self.windowSz, self.t = key, p2, windowSz, t
        self.watermarkMaskKey, self.random_seed = watermarkMaskKey, random_seed
        self.contextChecking = contextChecking
        self.token_history: List[int] = []; self.tokenIdx = 0
        self.payloadBits, self.msgLen = payloadBits, len(payloadBits)
        self.delta = 1.0 / (2**self.msgLen if self.msgLen > 0 else 1)
        self.deltaM = b2g(int(payloadBits, 2) if payloadBits else 0) * self.delta
        self.key_bytes = key.to_bytes(8, 'big', signed=True)
        self.mask_key_bytes = watermarkMaskKey.to_bytes(8, 'big', signed=True)
        self.seed_bytes = random_seed.to_bytes(8, 'big', signed=True)
        if self.contextChecking: self.seen_contexts = set()
        self.log = defaultdict(List)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits/self.t, dim=-1); newTokenId = 0
        padded_history = self._get_padded_history()
        for bitIdx in range(bitLen):
            p1 = getP1(probs, newTokenId, bitIdx)
            context_tuple = tuple(padded_history + [self.tokenIdx, bitIdx])
            
            is_fresh = (not self.contextChecking) or (context_tuple not in self.seen_contexts)
            mask_prf = self._getPrf(self.mask_key_bytes, self.mask_key_bytes, padded_history, self.tokenIdx, bitIdx)
            
            if is_fresh and (mask_prf <= self.p2):
                if self.contextChecking: self.seen_contexts.add(context_tuple)
                y = self._getPrf(self.key_bytes, self.key_bytes, padded_history, self.tokenIdx, bitIdx)
                start, end = self.deltaM, self.deltaM + p1
                nextBit = 1 if (start <= y < end if end <= 1.0 else (y >= start or y < (end - 1.0))) else 0
            else:
                if self.contextChecking and is_fresh: self.seen_contexts.add(context_tuple)
                context = torch.tensor(list(context_tuple), dtype=torch.float64)
                y = getY(self.seed_bytes, self.seed_bytes, context)
                nextBit = 1 if y < p1 else 0
            newTokenId = (newTokenId << 1) + nextBit
        self.token_history.append(newTokenId); self.tokenIdx += 1
        return torch.tensor(newTokenId, dtype=torch.long, device=device)
    # Add these methods inside the DISCP2 class
    def _get_padded_history(self) -> List[int]:
        start_idx = max(0, self.tokenIdx - self.windowSz)
        window = self.token_history[start_idx:self.tokenIdx]
        return ([0] * (self.windowSz - len(window))) + window

    @staticmethod
    def _get_padded_history_for_decode(full_history: List[int], current_token_idx: int, windowSz: int) -> List[int]:
        start_idx = max(0, current_token_idx - windowSz)
        window = full_history[start_idx:current_token_idx]
        return ([0] * (windowSz - len(window))) + window

    def _getPrf(self, salt:bytes, ikm:bytes, history: List[int], tokenIdx: int, bitIdx: int) -> float:
        context_tensor=torch.tensor(history + [tokenIdx, bitIdx], dtype=torch.float64)
        return getY(salt, ikm, context_tensor)
    
    def _calculatePValue(self, mPrime: int, msgLenHyp: int, watermarked_bits: Dict[int, int], all_token_ids: List[int]) -> tuple[float, float]:
        if not (0 <= mPrime < (2**msgLenHyp)): return 1.0, 0.0
        deltaMPrime = mPrime * (1.0 / (2**msgLenHyp))
        currentScore = 0.0
        for bitPos, w in watermarked_bits.items():
            tokenIdx, bitIdx = bitPos // blen, bitPos % blen
            padded_history = self._get_padded_history_for_decode(all_token_ids, tokenIdx, self.windowSz)
            yPrf = self._getPrf(self.key_bytes, self.key_bytes, padded_history, tokenIdx, bitIdx)
            v = (yPrf - deltaMPrime + 1.0) if w == 1 else (deltaMPrime - yPrf + 1.0)
            currentScore += -math.log(v % 1.0 + 1e-9)
        watermarkedLen = len(watermarked_bits)
        if watermarkedLen > 0: return gammaincc(watermarkedLen, currentScore), currentScore
        else: return 1.0, 0.0
    def decode(self, tokenIds: List[int], msgLenHyp: int) -> Dict:
        if not tokenIds: return {'detected': False, 'message': ''}
        totalBits = len(tokenIds) * blen
        fullBinarySequence = [int(b) for t in tokenIds for b in format(t, f'0{blen}b')]
        watermarked_bits = {}
        if self.contextChecking: seen_contexts = set()
        
        for bitPos in range(totalBits):
            tokenIdx, bitIdx = bitPos // blen, bitPos % blen
            padded_history = self._get_padded_history_for_decode(tokenIds, tokenIdx, self.windowSz)
            context_tuple = tuple(padded_history + [tokenIdx, bitIdx])
            
            is_fresh = (not self.contextChecking) or (context_tuple not in seen_contexts)
            if self.contextChecking: seen_contexts.add(context_tuple)

            if is_fresh:
                mask_prf = self._getPrf(self.mask_key_bytes, self.mask_key_bytes, padded_history, tokenIdx, bitIdx)
                if mask_prf <= self.p2:
                    watermarked_bits[bitPos] = fullBinarySequence[bitPos]
        
        msgSpaceHyp = 2**msgLenHyp
        bestPValue, bestMStar = 1.0, 0
        
        # Optional: Add Coarse/Fine search from DISC if msgSpaceHyp is large
        search_range = range(msgSpaceHyp)
        if msgSpaceHyp > 128:
            print(f"Large message space ({msgSpaceHyp}), consider adding coarse-to-fine search for speed.")

        for mPrime in search_range:
            pValue, _ = self._calculatePValue(mPrime, msgLenHyp, watermarked_bits, tokenIds)
            if pValue < bestPValue: bestPValue, bestMStar = pValue, mPrime
        hypothesesTestedCount = msgSpaceHyp
        globalPValue = 1.0 - (1.0 - bestPValue)**hypothesesTestedCount if hypothesesTestedCount > 0 else 1.0
        detected = globalPValue < 1e-2
        decodedBits = format(g2b(bestMStar), f'0{msgLenHyp}b') if detected else ""
        return {'detected': detected, 'message': decodedBits, 'p_value': globalPValue}
    
@torch.no_grad()
def generateSequence(prompt: str, algo: Callable, maxLen: int) -> List[int]:
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

def testChrist():
    print("--- 1. CHRIST DETECTION WATERMARK TEST ---")
    prompt,key,rLambda,maxLen,seed="Artificial intelligence is",42,4.0,40,0
    print(f"Generating watermarked text (RLambda = {rLambda})...")
    wm_encoder=Christ(salt=key,key=key,rLambda=rLambda,random_seed=seed)
    wmIds=generateSequence(prompt,wm_encoder,maxLen=maxLen)
    print(f"Generating normal text (RLambda = infinity)...")
    norm_encoder=Christ(salt=key,key=key,rLambda=float('inf'),random_seed=seed)
    normIds=generateSequence(prompt,norm_encoder,maxLen=maxLen)
    print("\nRunning detection...")
    det=Christ(salt=key,key=key,rLambda=rLambda,random_seed=seed)
    wmRes,normRes=det.decode(wmIds),det.decode(normIds)
    n_star_ok=wmRes['n_star']==len(wm_encoder.r)
    encoder_y=torch.tensor(wm_encoder.log['encode']['y'][wmRes['n_star']:])
    decoder_y=det.log['decode']['y_tensor'][wmRes['n_star']:,0,wmRes['n_star']]
    y_vals_ok=torch.allclose(encoder_y,decoder_y)if len(encoder_y)>0 else True
    ok,normOk=wmRes['detected']and n_star_ok and y_vals_ok,not normRes['detected']
    print(f"Watermarked text  -> {'DETECTED' if ok else 'NOT DETECTED'} (Score: {wmRes['score']:.2f})")
    print(f"  n* match: {'✅' if n_star_ok else '❌'} (Expected: {len(wm_encoder.r)}, Got: {wmRes['n_star']})")
    print(f"  Y-vals match: {'✅' if y_vals_ok else '❌'}")
    print(f"Normal text       -> {'NOT DETECTED' if normOk else 'DETECTED'} (Score: {normRes['score']:.2f})")
    print("\nResult: SUCCESS" if ok and normOk else "\nResult: FAILED")
    print("-" * 45 + "\n")
    return 1 if(ok and normOk)else 0

def testChristGeneral():
    print("--- 1c. CHRIST-GENERAL MULTI-BIT PAYLOAD TEST ---")
    prompt,payload,key,rLambda,maxLen,seed="The future of AI is",''.join(random.choice('01')for _ in range(2)),2077,5.0,60,42
    print(f"Embedding payload '{payload}' using binarized method...")
    cg_encoder=Christ(key=key,salt=key,rLambda=rLambda,payload=payload,random_seed=seed,isGeneral=True)
    ids=generateSequence(prompt,cg_encoder,maxLen=maxLen)
    print(f" -> Encoder finished with h={cg_encoder.h:.2f} after {len(cg_encoder.r)} prefix bits.")
    print("Decoding payload from generated text...")
    cg_decoder=Christ(key=key,salt=key,rLambda=rLambda,random_seed=seed,isGeneral=True)
    res=cg_decoder.decode(ids,payloadLen=len(payload))
    retrieved,p_idx=res['message'],int(payload,2)
    n_star_ok=res['n_star']==len(cg_encoder.r)
    o_idx=res['n_star']//bitLen
    encoder_y=torch.tensor(cg_encoder.log['encode']['y'][res['n_star']:])
    decoder_y=cg_decoder.log['decode']['y_tensor'][res['n_star']:,p_idx,o_idx]
    y_vals_ok=torch.allclose(encoder_y,decoder_y)if len(encoder_y)>0 else True
    ok=res['detected']and(retrieved==payload)and n_star_ok and y_vals_ok
    print(f"\nResult: {'SUCCESS' if ok else 'FAILED'}")
    print(f"  Detection Status:    {res['detected']}")
    print(f"  Payload Match:       {'✅' if retrieved == payload else '❌'}")
    print(f"    Original:          '{payload}'")
    print(f"    Retrieved:         '{retrieved}'")
    print(f"  n* Match:            {'✅' if n_star_ok else '❌'} (Expected: {len(cg_encoder.r)}, Got: {res['n_star']})")
    print(f"  Y-values Match:      {'✅' if y_vals_ok else '❌'}")
    print(f"  (Best Score: {res['score']:.2f})")
    print("-" * 45 + "\n")
    return 1 if ok else 0

def testOZ():
    print("--- 2. OZ PAYLOAD WATERMARK TEST (WITH ENTROPY GATHERING) ---")
    prompt,key,rLambda,seed,threshold,maxLen="The secret ingredient is",1337,4.0,0,3.0,150
    payload=''.join(random.choice('01')for _ in range(5))
    print(f"Embedding payload '{payload}' with rLambda={rLambda}...")
    # Note: The OZ constructor now requires rLambda and random_seed
    oz_encoder = OZ(key=key, rLambda=rLambda, random_seed=seed, payload=payload, threshold=threshold)
    ids=generateSequence(prompt, oz_encoder, maxLen=maxLen)
    print(f'true n {len(oz_encoder.r)}')
    print("Decoding payload...")
    oz_decoder = OZ(key=key, rLambda=rLambda, random_seed=seed, payload="", threshold=threshold)
    retrieved = oz_decoder.decode(ids)
    ok=retrieved.startswith(payload)
    print(f"\nResult: {'SUCCESS' if ok else 'FAILED'}")
    print(f"    Original:  '{payload}'\n    Retrieved: '{retrieved}'");print("-" * 45 + "\n")
    return 1 if ok else 0

def testDISC():
    print("--- 3. DISC PAYLOAD WATERMARK TEST ---")
    prompt,key,rLambda,maxLen,seed="The launch code is",2024,4.0,60,0
    payload=''.join(random.choice('01')for _ in range(2))
    print(f"Embedding payload '{payload}' (Message {int(payload,2)})...")
    enc=DISC(key=key,payloadBits=payload,rLambda=rLambda,random_seed=seed)
    ids=generateSequence(prompt,enc,maxLen=maxLen)
    txt=tokenizer.decode(ids,skip_special_tokens=True)
    print(f" -> Encoder state: H={enc.h:.2f}, n={len(enc.r)}")
    print(f" -> Generated text sample: '{txt[:80]}...'")
    print("\nDecoding payload from generated text...")
    res=DISC(key=key,rLambda=rLambda,random_seed=seed).decode(ids,msgLenHyp=len(payload))
    retrieved=res['message']
    ok=(retrieved==payload)and res['detected']
    print(f"\nResult: {'SUCCESS' if ok else 'FAILED'}")
    print(f"    Detection Status: {res['detected']}\n    Original Payload:   '{payload}'\n    Retrieved Payload:  '{retrieved}'")
    print(f"    (Best hypothesis: n*={res['n_star']}, M*={int(retrieved,2) if retrieved else 'N/A'}, p-val={res['p_value']:.2e})")
    print("-" * 45 + "\n")
    return 1 if ok else 0

def testDISCP2(payload=None):
    print("--- DISCP2 PAYLOAD WATERMARK TEST ---")
    prompt,payloadSz,key,maskKey,seed,p2,win,maxLen="Once upon a time ",2,2024,2025,0,0.5,4,200
    payload=''.join(random.choice('01')for _ in range(payloadSz))if payload is None else payload
    print(f"Embedding payload '{payload}' (Message {int(payload,2)}) with p2={p2}...")
    enc=DISCP2(key=key,watermarkMaskKey=maskKey,random_seed=seed,payloadBits=payload,p2=p2,windowSz=win)
    ids=generateSequence(prompt,enc,maxLen=maxLen)
    txt=tokenizer.decode(ids,skip_special_tokens=True)
    print(f" -> Generated text sample: '{txt[:80]}...'")
    print("\nDecoding payload from generated text...")
    res=DISCP2(key=key,watermarkMaskKey=maskKey,random_seed=seed,p2=p2,windowSz=win).decode(ids,msgLenHyp=len(payload))
    retrieved=res['message']
    ok=(retrieved==payload)and res['detected']
    print(f"\nResult: {'SUCCESS' if ok else 'FAILED'}")
    print(f"    Detection Status: {res['detected']}\n    Original Payload:   '{payload}'\n    Retrieved Payload:  '{retrieved}'")
    retrieved_int_str=str(int(retrieved,2))if retrieved else 'N/A'
    print(f"    (Best hypothesis: M*={retrieved_int_str}, p-val={res['p_value']:.2e})")
    print("-" * 45 + "\n")
    return 1 if ok else 0

if __name__ == '__main__':
    tests_to_run = [testChrist, testChristGeneral]*5
    success_count = sum(test() for test in tests_to_run)
    print(f"\n{'='*20}\nTest Passes: {success_count}/{len(tests_to_run)}\n{'='*20}")