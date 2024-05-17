import datasets
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr

import torch
from llm2vec import LLM2Vec

from transformers import AutoModel, AutoTokenizer

cache_dir = "D:\\huggingface\\cache"

dataset = "mteb/sts17-crosslingual-sts"
instruction = "Retrieve semantically similar text: "

# dataset = datasets.load_dataset(dataset, "ko-ko", cache_dir=cache_dir)
dataset = datasets.load_dataset(dataset, "ko-ko")

min_score, max_score = 0, 5
normalize = lambda x: (x - min_score) / (max_score - min_score)
normalized_scores = list(map(normalize, dataset["test"]["score"]))
batch_size = 8

sentences1, sentences2 = dataset["test"]["sentence1"], dataset["test"]["sentence2"]

print("Loading model...")
# model = LLM2Vec.from_pretrained(
#     "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
#     peft_model_name_or_path="D:\mlm\EEVE-Korean-Instruct-10.8B-RoBERTa-mntp-simcse\checkpoint-1000",
#     device_map="cuda" if torch.cuda.is_available() else "cpu",
#     torch_dtype=torch.bfloat16,
#     cache_dir=cache_dir
# )
#
# def append_instruction(instruction, sentences):
#     new_sentences = []
#     for s in sentences:
#         new_sentences.append([instruction, s, 0])
#     return new_sentences
#
# print(f"Encoding {len(sentences1)} sentences1...")
# instruction_sentences1 = append_instruction(instruction, sentences1)
# embeddings1 = np.asarray(model.encode(instruction_sentences1, batch_size=batch_size))
#
# print(f"Encoding {len(sentences2)} sentences2...")
# instruction_sentences2 = append_instruction(instruction, sentences2)
# embeddings2 = np.asarray(model.encode(instruction_sentences2, batch_size=batch_size))
#

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
# cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
cosine_scores = 1 - (paired_cosine_distances(robert_embeddings1.detach().cpu().numpy(), robert_embeddings2.detach().cpu().numpy()))
cosine_spearman, _ = spearmanr(normalized_scores, cosine_scores)

results = {
    "cos_sim": {
        "spearman": cosine_spearman,
    }
}

print(results)
# {'cos_sim': {'spearman': 0.9021906216635642}}
