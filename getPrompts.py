from datasets import load_dataset
from itertools import islice

DATASET_ID = "allenai/c4"
DATASET_CONFIG = "realnewslike"
NUM_PROMPTS = 1000
OUTPUT_FILE = "prompts.txt"
WORD_LIMIT = 20

# --- Generation Step with Truncation ---
print(f"Generating '{OUTPUT_FILE}' with {NUM_PROMPTS} prompts, each truncated to {WORD_LIMIT} words...")
dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for example in islice(dataset, NUM_PROMPTS):
        words = example['text'].split()
        truncated_text = ' '.join(words[:WORD_LIMIT])
        f.write(truncated_text + "\n")
print("File generation complete.")

# --- Verification Step ---
print("Verifying file...")
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    line_count = sum(1 for _ in f)

print(f"✅ Verification successful: '{OUTPUT_FILE}' has {line_count} rows.")