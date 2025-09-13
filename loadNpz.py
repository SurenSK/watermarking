import numpy as np
from pathlib import Path

results_dir = Path("results")

# Find all .npz files from the experiment
for p in sorted(results_dir.glob("experiment0_1bit_results_*.npz")):
    # allow_pickle=True is required to load the 'encoder_r' array
    data = np.load(p, allow_pickle=True)
    
    print(f"--- Loading: {p.name} ---")

    # --- Metadata and Parameters ---
    is_wm_str = "Watermarked" if data['isWM'] else "Not Watermarked"
    print(f"Index: {data['idx']} ({is_wm_str})")
    print(f"Encode Time: {data['tEncode']:.2f}s, Decode Time: {data['tDecode']:.2f}s")
    
    # --- Decode Results ---
    decodeRes = {
        'detected': data['detected'], 'score': data['score'], 
        'n_star': data['n_star'], 'message': data['message']
    }
    print(f"Decode Result: {decodeRes}")

    # --- Encoder Log Verification ---
    print("  Encoder Log:")
    if 'encoder_y' in data:
        print(f"    - encoder_y:                (length {len(data['encoder_y'])})")
        print(f"    - encoder_p1:               (length {len(data['encoder_p1'])})")
        print(f"    - encoder_binaryEntropy:    (length {len(data['encoder_binaryEntropy'])})")
        print(f"    - encoder_vocabEntropy:     (length {len(data['encoder_vocabEntropy'])})")
        if 'encoder_r' in data and data['encoder_r'].size > 0:
            print(f"    - encoder_r:                (length {len(data['encoder_r'])}, dtype={data['encoder_r'].dtype})")
        else:
            print("    - encoder_r:                (empty or not found)")
    else:
        print("    - No encoder data found.")

    # --- Decoder Log Verification ---
    print("  Decoder Log:")
    if 'decoder_y' in data:
        print(f"    - decoder_y:                (shape {data['decoder_y'].shape}, dtype {data['decoder_y'].dtype})")
        print(f"    - decoder_scores:           (shape {data['decoder_scores'].shape})")
        print(f"    - decoder_normScores:       (shape {data['decoder_normScores'].shape})")
    else:
        print("    - No decoder data found.")
    print("-" * 20 + "\n")
    pass