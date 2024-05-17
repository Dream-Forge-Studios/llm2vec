# -*- coding: utf-8 -*-

import datasets
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer
import torch
from llm2vec import LLM2Vec

instruction = "Retrieve semantically similar text: "
sentences1 = """
나는 너무 많이 먹어서 배 불르다.
"""
sentences2 = "나는 아이가 생겨 배가 부르고 있다."

print("Loading model...")
model = LLM2Vec.from_pretrained(
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    peft_model_name_or_path="D:\mlm\EEVE-Korean-Instruct-10.8B-RoBERTa-mntp\checkpoint-1000",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    cache_dir="D:\\huggingface\\cache",
)

embeddings1 = model.encode([instruction, sentences1, 0], batch_size=1)
embeddings2 = model.encode([instruction, sentences2, 0], batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta').to(device)
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')  # or 'BM-K/KoSimCSE-bert-multitask'

def robertMeanPooling(attention_mask, embeddings):
    # attention_mask를 embeddings 차원에 맞게 조정
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    # 실제 토큰 위치는 1, 패딩 토큰 위치는 0의 값을 가진 마스크를 사용하여 유효한 토큰 값만을 합산
    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
    # 각 입력에 대해 유효한 토큰의 수 계산
    sum_mask = attention_mask.sum(1).unsqueeze(-1)  # 이 값이 0이 되지 않도록 주의, 차원을 맞춰주기 위해 unsqueeze 사용
    # 평균 임베딩 계산
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings

# 메모리 관리 및 모니터링
torch.cuda.empty_cache()

inputs1 = tokenizer(sentences1, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
embeddings11, _ = model(**inputs1, return_dict=False)

robert_embeddings1 = robertMeanPooling(inputs1['attention_mask'], embeddings11)

inputs2 = tokenizer(sentences2, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
embeddings22, _ = model(**inputs2, return_dict=False)
robert_embeddings2 = robertMeanPooling(inputs2['attention_mask'], embeddings22)

print("Evaluating...")
cosine_scoresRo = 1 - (paired_cosine_distances(robert_embeddings1.detach().cpu().numpy(), robert_embeddings2.detach().cpu().numpy()))
# 코사인 유사도 계산
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

print("Cosine Similarity:", cosine_scores)
print("Cosine Similarity roberta:", cosine_scoresRo)
