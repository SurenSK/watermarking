import argparse
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
    info = context.cpu().numpy().tobytes()
    hkdf = HKDF(algorithm=hashes.SHA256(), length=8, salt=salt, info=info)
    seed = int.from_bytes(hkdf.derive(ikm), 'big')
    return seed / (2**64 - 1)

class Christ:
    def __init__(self, key: int, rLambda: float, random_seed: int, t: float = 1.0, payload: Optional[str] = None, scoreThreshold: Optional[float] = None):
        self.key=key
        self.rLambda=rLambda
        self.random_seed=random_seed
        self.t=t
        self.payload_int=int(payload,2)if payload is not None else 0
        self.scoreThreshold=rLambda if scoreThreshold is None else scoreThreshold
        self.h,self.r,self.tokenIdx=0.0,[],0
        self.key_bytes=key.to_bytes(8,'big',signed=True)
        self.seed_bytes=random_seed.to_bytes(8,'big',signed=True)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        newTokenId=0
        probs=torch.softmax(logits/self.t,dim=-1)
        for bitIdx in range(bitLen):
            p1=getP1(probs,newTokenId,bitIdx)
            if self.h<self.rLambda:
                context=torch.tensor([len(self.r)],dtype=torch.float64)
                y=getY(self.seed_bytes,self.seed_bytes,context)
                nextBit=1 if y<p1 else 0
                probChosen=p1 if nextBit==1 else(1-p1)
                self.h+=-math.log(probChosen+1e-9)
                self.r.append(nextBit)
            else:
                context=torch.tensor(self.r+[self.tokenIdx,bitIdx,self.payload_int],dtype=torch.float64)
                y=getY(self.key_bytes,self.key_bytes,context)
                nextBit=1 if y<p1 else 0
            newTokenId=(newTokenId<<1)+nextBit
        self.tokenIdx+=1
        return torch.tensor(newTokenId,dtype=torch.long,device=device)

    def decode(self, tokenIds: List[int], payloadLen: Optional[int] = None) -> Dict:
        if not tokenIds: return {'detected': False, 'score': -math.inf, 'n_star': 0, 'message': ''}
        totalBits=len(tokenIds)*bitLen
        fullBinary=[int(b) for t in tokenIds for b in format(t,f'0{bitLen}b')]

        def _score_for_nstar(nStar,hyp):
            currentScore,rSeq=0.0,fullBinary[:nStar]
            for bitPos in range(nStar,totalBits):
                tokenIdx,bitIdx=bitPos//bitLen,bitPos%bitLen
                context=torch.tensor(rSeq+[tokenIdx,bitIdx,hyp],dtype=torch.float64)
                yPrf=getY(self.key_bytes,self.key_bytes,context)
                obsBit=fullBinary[bitPos]
                v=yPrf if obsBit==1 else(1-yPrf)
                currentScore+=-math.log(v+1e-9)
            wmLen=totalBits-nStar
            normScore=(currentScore-wmLen)/math.sqrt(wmLen)if wmLen>0 else 0.0
            return normScore,nStar

        num_messages=2**payloadLen if payloadLen is not None and payloadLen>0 else 1
        best_overall_score,best_message_int,best_n_star=-math.inf,-1,0
        for payload_hyp_int in range(num_messages):
            for n in range(totalBits+1):
                normScore,nStar=_score_for_nstar(n,payload_hyp_int)
                if normScore>best_overall_score:
                    best_overall_score,best_message_int,best_n_star=normScore,payload_hyp_int,nStar
        
        detected=best_overall_score>self.scoreThreshold
        message=""
        if detected and payloadLen is not None and payloadLen>0:
            message=format(best_message_int,f'0{payloadLen}b')
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
    def __init__(self, key: int, payload: str = "", threshold: float = 2.0, t: float=1.0):
        self.key, self.payload, self.threshold, self.t = key, payload, threshold, t
        self.tokenIdx, self.ecc, self.code = 0, _OZ_DynamicEcc(), []
        self.nextSymbol = self.ecc.getNextSymbol(self.payload, [])
        self.scores, self.scoreLen = torch.zeros(3, device=device), 0.0
        # --- Simplified State ---
        self.key_bytes = key.to_bytes(8, 'big', signed=True)
        # Salt can be a fixed byte string for this scheme
        self.salt_bytes = b"OZ-WATERMARK-SALT"

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits/self.t, dim=-1); newTokenId = 0
        for bitIdx in range(bitLen):
            p1 = getP1(probs, newTokenId, bitIdx)
            # Directly use getY with simplified state
            yPrfVals = torch.tensor([
                getY(self.salt_bytes, self.key_bytes, torch.tensor([self.tokenIdx, bitIdx, s])) 
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
        scores, scoreLen, retrieved = torch.zeros(3, device=device), 0.0, []
        bitSeq = [int(b) for t in tokenIds for b in format(t, f'0{bitLen}b')]
        for bitPos, bit in enumerate(bitSeq):
            scoreLen += 1.0
            tokenIdx, bitIdx = bitPos // bitLen, bitPos % bitLen
            yPrfs = torch.tensor([
                getY(self.salt_bytes, self.key_bytes, torch.tensor([tokenIdx, bitIdx, s])) 
                for s in range(3)
            ], device=device)
            v = torch.where(torch.tensor(bit, device=device) == 1, yPrfs, 1 - yPrfs)
            scores += -torch.log(v + 1e-9)
            if scoreLen > 0:
                normScores = (scores - scoreLen) / math.sqrt(scoreLen)
                if (passed := torch.where(normScores > self.threshold)[0]).size(0) > 0:
                    retrieved.append(passed[0].item())
                    scores.zero_(); scoreLen = 0.0
        return self.ecc.decode(retrieved)
    
def b2g(n: int) -> int: return n ^ (n >> 1)
def g2b(n: int) -> int:
    mask = n >> 1
    while mask != 0: n = n ^ mask; mask >>= 1
    return n

blen = math.ceil(math.log2(len(tokenizer)))

class DISC:
    def __init__(self, key: int, random_seed: int, payloadBits: str = "", rLambda: float = 4.0, t: float=1.0):
        self.key, self.rLambda, self.t = key, rLambda, t
        self.h, self.r, self.tokenIdx = 0.0, [], 0
        self.payloadBits, self.msgLen = payloadBits, len(payloadBits)
        self.msgSpaceSz = 2**self.msgLen if self.msgLen > 0 else 1
        self.delta = 1.0 / self.msgSpaceSz
        gMessage = b2g(int(payloadBits, 2) if payloadBits else 0)
        self.deltaM = gMessage * self.delta
        # --- Simplified State ---
        self.key_bytes = key.to_bytes(8, 'big', signed=True)
        self.seed_bytes = random_seed.to_bytes(8, 'big', signed=True)
        # Salt can be a fixed byte string
        self.salt_bytes = b"DISC-WATERMARK-SALT"

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits/self.t, dim=-1); newTokenId = 0
        for bitIdx in range(bitLen):
            p1 = getP1(probs, newTokenId, bitIdx)
            if self.h < self.rLambda:
                # Replace torch.rand with a deterministic KDF call
                context = torch.tensor([len(self.r)], dtype=torch.float64)
                y = getY(self.seed_bytes, self.seed_bytes, context)
                nextBit = 1 if y < p1 else 0
                probChosen = p1 if nextBit == 1 else (1 - p1)
                self.h += -math.log(probChosen + 1e-9)
                self.r.append(nextBit)
            else:
                context = torch.tensor(self.r + [self.tokenIdx, bitIdx], dtype=torch.float64)
                y = getY(self.salt_bytes, self.key_bytes, context)
                start, end = self.deltaM, self.deltaM + p1
                nextBit = 1 if (start <= y < end if end <= 1.0 else y >= start or y < (end - 1.0)) else 0
            newTokenId = (newTokenId << 1) + nextBit
        self.tokenIdx += 1
        return torch.tensor(newTokenId, dtype=torch.long, device=device)
    
    def _getPValueForHyp(self, nStar: int, mPrime: int, fullBinarySequence: List[int], totalBits: int, msgLenHyp: int) -> tuple[float, float, dict]:
        if not (0 <= mPrime < (2**msgLenHyp)): return 1.0, 0.0, {}
        rSequenceHyp, deltaMPrime = fullBinarySequence[:nStar], mPrime * (1.0 / (2**msgLenHyp))
        currentScore, bit_scores = 0.0, {}
        for bitPos in range(nStar, totalBits):
            tokenIdx, bitIdx = bitPos // blen, bitPos % blen
            yPrf = self._getYPrf(rSequenceHyp, tokenIdx, bitIdx)
            w = fullBinarySequence[bitPos]
            score_component = self._calculateScore(w, yPrf, deltaMPrime)
            currentScore += score_component; bit_scores[bitPos] = score_component
        watermarkedLen = totalBits - nStar
        if watermarkedLen <= 0: return 1.0, 0.0, {}
        p_value = gammaincc(watermarkedLen, currentScore)
        return p_value, currentScore, bit_scores
    def decode(self, tokenIds: List[int], msgLenHyp: int) -> Dict:
        if not tokenIds: return {'detected': False, 'message': ''}
        totalBits = len(tokenIds) * blen
        fullBinarySequence = [int(b) for t in tokenIds for b in format(t, f'0{blen}b')]
        msgSpaceHyp = 2 ** msgLenHyp

        def _pvalue_for_hyp(nStar, mPrime):
            if not (0 <= mPrime < msgSpaceHyp): return 1.0, 0.0
            rSeq, deltaMPrime = fullBinarySequence[:nStar], mPrime * (1.0 / msgSpaceHyp)
            currentScore = 0.0
            for bitPos in range(nStar, totalBits):
                tokenIdx, bitIdx = bitPos // blen, bitPos % blen
                context = torch.tensor(rSeq + [tokenIdx, bitIdx], dtype=torch.float64)
                yPrf = getY(self.salt_bytes, self.key_bytes, context)
                w = fullBinarySequence[bitPos]
                v = (yPrf - deltaMPrime + 1.0) if w == 1 else (deltaMPrime - yPrf + 1.0)
                currentScore += -math.log(v % 1.0 + 1e-9)
            watermarkedLen = totalBits - nStar
            if watermarkedLen <= 0: return 1.0, 0.0
            return gammaincc(watermarkedLen, currentScore), currentScore

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
    def __init__(self, key: int, watermarkMaskKey: int, random_seed: int, payloadBits: str = "", p2: float = 0.25, t: float = 1.0, windowSz: int = 8):
        self.key, self.p2, self.windowSz, self.t = key, p2, windowSz, t
        self.watermarkMaskKey = watermarkMaskKey
        self.token_history: List[int] = []; self.tokenIdx = 0
        self.payloadBits, self.msgLen = payloadBits, len(payloadBits)
        self.msgSpaceSz = 2**self.msgLen if self.msgLen > 0 else 1
        self.delta = 1.0 / self.msgSpaceSz
        gMessage = b2g(int(payloadBits, 2) if payloadBits else 0)
        self.deltaM = gMessage * self.delta
        # --- Simplified State ---
        self.key_bytes = key.to_bytes(8, 'big', signed=True)
        self.mask_key_bytes = watermarkMaskKey.to_bytes(8, 'big', signed=True)
        self.seed_bytes = random_seed.to_bytes(8, 'big', signed=True)
        

    def _getPrf(self, salt: bytes, ikm: bytes, history: List[int], tokenIdx: int, bitIdx: int) -> float:
        context=torch.tensor(history + [tokenIdx, bitIdx], dtype=torch.float64)
        return getY(salt, ikm, context)

    def _get_padded_history(self) -> List[int]:
        start_idx = max(0, self.tokenIdx - self.windowSz)
        window = self.token_history[start_idx:self.tokenIdx]
        return ([0] * (self.windowSz - len(window))) + window

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits/self.t, dim=-1); newTokenId = 0
        padded_history = self._get_padded_history()
        for bitIdx in range(bitLen):
            mask_prf = self._getPrf(self.mask_key_bytes, self.mask_key_bytes, padded_history, self.tokenIdx, bitIdx)
            should_watermark = (mask_prf <= self.p2)
            p1 = getP1(probs, newTokenId, bitIdx)
            if should_watermark:
                y = self._getPrf(self.key_bytes, self.key_bytes, padded_history, self.tokenIdx, bitIdx)
                start, end = self.deltaM, self.deltaM + p1
                nextBit = 1 if (start <= y < end if end <= 1.0 else (y >= start or y < (end - 1.0))) else 0
            else:
                # Replace torch.rand with a deterministic KDF call
                context = torch.tensor(padded_history + [self.tokenIdx, bitIdx], dtype=torch.float64)
                y = getY(self.seed_bytes, self.seed_bytes, context)
                nextBit = 1 if y < p1 else 0
            newTokenId = (newTokenId << 1) + nextBit
        self.token_history.append(newTokenId); self.tokenIdx += 1
        return torch.tensor(newTokenId, dtype=torch.long, device=device)
    @staticmethod
    def _get_padded_history_for_decode(full_history: List[int], current_token_idx: int, windowSz: int) -> List[int]:
        start_idx = max(0, current_token_idx - windowSz)
        window = full_history[start_idx:current_token_idx]
        return ([0] * (windowSz - len(window))) + window

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
        for bitPos in range(totalBits):
            tokenIdx, bitIdx = bitPos // blen, bitPos % blen
            padded_history = self._get_padded_history_for_decode(tokenIds, tokenIdx, self.windowSz)
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
    prompt, key, rLambda, maxLen, seed = "Artificial intelligence is", 42, 4.0, 40, 0
    print(f"Generating watermarked text (RLambda = {rLambda})...")
    wmIds=generateSequence(prompt,Christ(key=key,rLambda=rLambda,random_seed=seed),maxLen=maxLen)
    print(f"Generating normal text (RLambda = infinity)...")
    normIds=generateSequence(prompt,Christ(key=key,rLambda=float('inf'),random_seed=seed),maxLen=maxLen)
    print("\nRunning detection...")
    det=Christ(key=key,rLambda=rLambda,random_seed=seed)
    wmRes,normRes=det.decode(wmIds),det.decode(normIds)
    ok,normOk=wmRes['detected'],not normRes['detected']
    print(f"Watermarked text  -> {'DETECTED' if ok else 'NOT DETECTED'} (Score: {wmRes['score']:.2f})")
    print(f"Normal text       -> {'NOT DETECTED' if normOk else 'DETECTED'} (Score: {normRes['score']:.2f})")
    print("\nResult: SUCCESS" if ok and normOk else "\nResult: FAILED");print("-" * 45 + "\n")
    return 1 if(ok and normOk)else 0

def testChristMultiBit():
    print("--- 1b. CHRIST MULTI-BIT PAYLOAD TEST ---")
    prompt,payload,key,rLambda,maxLen,seed="The secret launch code is","10",1984,4.0,50,0
    print(f"Embedding payload '{payload}'...")
    ids=generateSequence(prompt,Christ(key=key,rLambda=rLambda,payload=payload,random_seed=seed),maxLen=maxLen)
    print("Decoding payload from generated text...")
    res=Christ(key=key,rLambda=rLambda,random_seed=seed).decode(ids,payloadLen=len(payload))
    retrieved=res['message']
    ok=res['detected']and(retrieved==payload)
    print(f"\nResult: {'SUCCESS' if ok else 'FAILED'}")
    print(f"  Detection Status:  {res['detected']}\n  Original Payload:    '{payload}'\n  Retrieved Payload:   '{retrieved}'")
    print(f"  (Score: {res['score']:.2f}, n*={res['n_star']})");print("-" * 45 + "\n")
    return 1 if ok else 0

def testOZ():
    # No changes needed for testOZ as its __init__ signature was not changed
    print("--- 2. OZ PAYLOAD WATERMARK TEST ---")
    prompt,key,threshold,maxLen="The secret ingredient is",1337,3.0,150
    payload=''.join(random.choice('01')for _ in range(5))
    print(f"Embedding payload '{payload}'...")
    ids=generateSequence(prompt,OZ(key=key,payload=payload,threshold=threshold),maxLen=maxLen)
    print("Decoding payload...")
    retrieved=OZ(key=key,payload="",threshold=threshold).decode(ids)
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
    # You can choose which tests to run
    tests_to_run = [testDISC]
    success_count = sum(test() for test in tests_to_run)
    print(f"\n{'='*20}\nTest Passes: {success_count}/{len(tests_to_run)}\n{'='*20}")