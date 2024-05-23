# -*- coding: utf-8 -*-

import datasets
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer
import torch
from llm2vec import LLM2Vec
from sklearn.decomposition import PCA

instruction = "Retrieve semantically similar text: "
# sentences1 = """
# 건물 주차장 천장 화재 사고 책임, 누구에게 있을까?
# 상황
# A 회사는 투자 목적으로 어떤 건물을 사들였고, 그 건물의 소유권은 안전하게 관리하기 위해 B 은행에 맡겨졌습니다. A 회사는 C 회사에게 건물 관리를 맡겼고, C 회사는 A 회사의 지시에 따라 건물을 관리했습니다. 그런데 어느 날, 이 건물 주차장 천장에서 화재가 발생하여 임차인 D 회사가 큰 피해를 입었습니다.
# 판결 요지
# 건물 점유자는 누구? D 회사는 A 회사, B 은행, 그리고 C 회사를 상대로 손해배상을 청구했습니다. 법원은 A 회사와 B 은행이 공동으로 건물을 실질적으로 지배하고 관리할 책임이 있다고 판단하여, 이들이 공동으로 손해를 배상해야 한다고 판결했습니다. 반면 C 회사는 A 회사의 지시를 받아 건물을 관리하는 점유보조자에 불과하므로 책임이 없다고 보았습니다.
# 손해배상 책임의 범위는? 법원은 A 회사와 B 은행이 투자신탁재산을 넘어서 고유재산으로도 손해를 배상해야 한다고 판결했습니다.
# 실화책임법 적용은? 법원은 실화책임법은 직접 화재로 인한 손해가 아닌, 화재가 번져서 발생한 손해에 대해서만 책임을 제한한다고 보았습니다.
# 쉽게 이해하기
# 이번 판결은 투자 목적으로 소유권을 가진 회사와 실제 관리를 맡은 회사가 다른 경우, 누가 건물 관리 책임을 지는지 명확히 보여줍니다. 비록 투자 목적으로 건물을 소유했더라도, 실질적인 관리 책임을 져야 한다는 것입니다. 또한, 화재로 인한 손해배상 책임 범위를 명확히 하여 피해자 보호를 강화했습니다.
# 결론
# 투자 목적으로 건물을 소유한 경우에도 실질적인 관리 책임을 져야 하며, 화재 발생 시 투자금을 넘어 고유재산으로도 손해를 배상해야 할 수 있습니다.
# 핵심 법률
# 민법 제195조 (점유보조자) 타인의 점유를 보조하는 자는 그 타인의 지시를 받아서 물건에 대한 사실상의 지배를 하는 때에는 점유의 의사가 없어도 점유자로 본다.
# 민법 제758조 (공작물등의 점유자, 소유자의 책임) ① 공작물의 점유자 또는 소유자는 공작물의 설치 또는 보존의 하자로 인하여 타인에게 손해를 가한 때에는 책임을 면하지 못한다.
# 실화책임에 관한 법률 실화로 인하여 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다. 다만, 중대한 과실 없이 실화를 한 자는 법원은 재산상태 기타 모든 사정을 참작하여 그 책임을 경감하거나 면제할 수 있다.
# 신탁법 제114조 (유한책임신탁) ① 신탁행위로 특별히 정한 경우 외에는 수탁자의 책임은 신탁재산에 한한다. 다만, 수탁자가 고의 또는 중과실로 인하여 신탁재산의 관리 또는 처분상의 의무를 위반하여 신탁채권자에게 손해를 발생하게 한 때에는 그러하지 아니하다.
# """
sentences1 = """
친구에게 빌려준 자전거를 친구의 동생이 타다가 망가뜨렸어요. 누구에게 책임을 물어야 하나요?
"""
# sentences2 = """
# 친구에게 빌려준 자전거를 친구의 동생이 타다가 망가뜨렸어요. 누구에게 책임을 물어야 하나요?
# """
sentences2 = """
나는 친구를 때렸어 어떻게 해야할까?
"""

print("Loading model...")
model = LLM2Vec.from_pretrained(
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    # "maywell/Synatra-7B-v0.3-dpo",
    # peft_model_name_or_path="D:\mlm\EEVE-Korean-Instruct-10.8B-RoBERTa-mntp\checkpoint-1000",
    # peft_model_name_or_path="D:\mlm\EEVE-Korean-Instruct-10.8B-RoBERTa-mntp-qa-supervised\ko_wikidata_QA_train_m-EEVE-Korean-Instruct-10.8B-v1.0_p-mean_b-12_l-512_bidirectional-True_e-3_s-42_w-300_lr-3e-05_lora_r-16\checkpoint-1000",
    # peft_model_name_or_path="D:\mlm\Synatra-7B-v0.3-dpo-RoBERTa-mntp\checkpoint-1000",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    cache_dir="D:\\huggingface\\cache",
    # pooling_mode="eos_token",
    # enable_bidirectional=False
)

embeddings1 = model.encode([instruction, sentences1, 0], batch_size=1)
embeddings2 = model.encode([instruction, sentences2, 0], batch_size=1)

# PCA 인스턴스 생성, 축소하고자 하는 차원 수 설정 (예: 10차원으로 축소)
pca = PCA(n_components=2048)

# 데이터에 PCA 적용
embeddings1 = pca.fit_transform(embeddings1)
embeddings2 = pca.fit_transform(embeddings2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta', cache_dir="D:\\huggingface\\cache").to(device)
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta', cache_dir="D:\\huggingface\\cache")

# 저장된 모델 상태 로드
# checkpoint = torch.load(r'D:\llm2vec\KoSimCSE_my\test.pt', map_location=device)
# new_state_dict = {key.replace('bert.', ''): value for key, value in checkpoint['model'].items()}
#
# # 모델 상태 사전 로드 (모델 파라미터)
# model.load_state_dict(new_state_dict)
# model.eval()  # 모델을 평가 모드로 설정

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
