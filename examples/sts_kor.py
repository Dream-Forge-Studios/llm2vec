import datasets
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr

import torch
from llm2vec import LLM2Vec

from transformers import AutoModel, AutoTokenizer

cache_dir = "D:\\huggingface\\cache"

file_path = "D:/KorSTS/sts-train.tsv"
instruction = "Retrieve semantically similar text: "

dataset = {
    "test": {
        "score": [],
        "sentence1": [],
        "sentence2": [],
    },
}
with open(file_path, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        temps = line.strip().split('\t')
        dataset["test"]["score"].append(float(temps[4]))
        dataset["test"]["sentence1"].append(temps[5])
        dataset["test"]["sentence2"].append(temps[6])
min_score, max_score = 0, 5
normalize = lambda x: (x - min_score) / (max_score - min_score)
normalized_scores = list(map(normalize, dataset["test"]["score"]))
batch_size = 8

sentences1, sentences2 = dataset["test"]["sentence1"], dataset["test"]["sentence2"]

print("Loading model...")
model = LLM2Vec.from_pretrained(
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    # "maywell/Synatra-7B-v0.3-dpo",
    # "maywell/Synatra-7B-v0.3-RP",
    # peft_model_name_or_path="D:\mlm\EEVE-Korean-Instruct-10.8B-RoBERTa-mntp-qa-supervised\ko_wikidata_QA_train_m-EEVE-Korean-Instruct-10.8B-v1.0_p-mean_b-12_l-512_bidirectional-True_e-3_s-42_w-300_lr-3e-05_lora_r-16\checkpoint-1000",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)

def append_instruction(instruction, sentences):
    new_sentences = []
    for s in sentences:
        new_sentences.append([instruction, s, 0])
    return new_sentences

print(f"Encoding {len(sentences1)} sentences1...")
instruction_sentences1 = append_instruction(instruction, sentences1)
embeddings1 = np.asarray(model.encode(instruction_sentences1, batch_size=batch_size))

print(f"Encoding {len(sentences2)} sentences2...")
instruction_sentences2 = append_instruction(instruction, sentences2)
embeddings2 = np.asarray(model.encode(instruction_sentences2, batch_size=batch_size))
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# def get_embeddings(model, tokenizer, sentences, device):
#     inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
#     with torch.no_grad():  # 그라디언트 계산 비활성화
#         outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state[:, 0, :]  # 첫번째 토큰(보통 CLS 토큰) 사용
#     return embeddings
#
# def paired_cosine(embeddings1, embeddings2):
#     embeddings1_np = embeddings1.detach().cpu().numpy()
#     embeddings2_np = embeddings2.detach().cpu().numpy()
#     distances = paired_cosine_distances(embeddings1_np, embeddings2_np)
#     return 1 - distances
#
# # 모델과 토크나이저 초기화
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask').to(device)
# tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')
#
# # 임베딩 계산
# embeddings1 = get_embeddings(model, tokenizer, sentences1, device)
# embeddings2 = get_embeddings(model, tokenizer, sentences2, device)
#
# # 코사인 유사도 계산
# cosine_scores = paired_cosine(embeddings1, embeddings2)

print("Evaluating...")
cosine_spearman, _ = spearmanr(normalized_scores, cosine_scores)

results = {
    "cos_sim": {
        "spearman": cosine_spearman,
        # "spearmanRo": cosine_spearmanRo,
    }
}

print(results)
# {'cos_sim': {'spearman': 0.9021906216635642}}
