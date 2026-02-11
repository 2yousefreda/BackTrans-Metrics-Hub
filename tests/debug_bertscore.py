
from bert_score import score
import torch

cands = ["تعتبر اللغة العربية من أقدم اللغات"]
refs = ["تعتبر اللغة العربية من أقدم اللغات"]

print("Testing BERTScore with aubmindlab/bert-base-arabertv2...")
try:
    P, R, F1 = score(cands, refs, model_type="aubmindlab/bert-base-arabertv2", lang="ar", verbose=True)
    print(f"F1: {F1}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting BERTScore with bert-base-multilingual-cased (as fallback)...")
try:
    P, R, F1 = score(cands, refs, model_type="bert-base-multilingual-cased", lang="ar", verbose=True)
    print(f"F1: {F1}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
