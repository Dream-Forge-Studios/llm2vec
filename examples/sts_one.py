import datasets
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr

import torch
from llm2vec import LLM2Vec

instruction = "Retrieve semantically similar text: "
sentences1, sentences2 = "남는 돈 운용 규정을 위반하여 사모펀드에 입금한 사건", "여유자금을 유동화계획 외의 다른 부동산 프로젝트에 투자한 유동화전문회사 관련 분쟁"

print("Loading model...")
model = LLM2Vec.from_pretrained(
    "yanolja/EEVE-Korean-10.8B-v1.0",
    peft_model_name_or_path="D:\\mlm\\EEVE-Korean-Instruct-10.8B-RoBERTa-mntp",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

embeddings1 = model.encode([instruction, sentences1, 0], batch_size=1)
embeddings2 = model.encode([instruction, sentences2, 0], batch_size=1)

# 코사인 유사도 계산
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

print("Cosine Similarity:", cosine_scores)
