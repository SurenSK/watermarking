BER - Christ (General), DISC (context), DISCP2 (context), OZ

use C4 dataset, 1000 prompts, 1 completion per prompt (wm + nwm), max_new_tokens=500

for each binary token - bitLen P1 (binary), Entropy (binary), EmpiricalEntropy(binary), Y (binary)
for each vocabulary token Entropy + Empirical Entropy also
store keys, etc
timeEncode

scores tensor |len(text bits)-1| x |payloads| x |len(text bits)-1| 
normScore |len(text bits)-1| x |payloads| (normalized over offsets)
y tensor |len(text bits)-1|  x |payloads| x |len(text bits)-1|
timeDecode
isWM @ threshold=lambda

for each token len:
    FPR and FNR for threshold=lambda for Christ and Christ General
    plot for different thresholds

then check perf of Christ up to 8bit payloads

BER - Christ (General), DISC (context), DISCP2 (context), OZ

use C4 dataset, 1000 prompts, 1 completion per prompt (wm + nwm), max_new_tokens=500

for each binary token - bitLen P1 (binary), Entropy (binary), EmpiricalEntropy(binary), Y (binary)
for each V Entropy + Empirical Entropy also
store keys, etc
timeEncode

scores tensor |len(text bits)-1| x |payloads| x |len(text bits)-1|
timeDecode
isWM @ threshold=lambda
for each token len:
    FPR and FNR for threshold=lambda for Christ and Christ General
    plot for different thresholds