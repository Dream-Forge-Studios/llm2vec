import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather
import torch.nn.functional as F
from sklearn.metrics.pairwise import paired_cosine_distances

class DPOLoss():
    def __init__(
        self,
        beta: float = 0.1,
        similarity_fct = cos_sim,
    ):
        self.beta = beta
        self.similarity_fct = similarity_fct

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        reference_score: list,
    ):
        reference_score_tensor = torch.tensor(reference_score, device=q_reps.device)
        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_reference_score_tensor = mismatched_sizes_all_gather(reference_score_tensor)
            full_reference_score_tensor = torch.cat(full_reference_score_tensor)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_reference_score_tensor = reference_score_tensor

        q_reps_cpu = full_q_reps.cpu().to(torch.float32).detach().numpy()
        d_reps_pos_cpu = full_d_reps_pos.cpu().to(torch.float32).detach().numpy()
        policy_scores = 1 - (paired_cosine_distances(q_reps_cpu, d_reps_pos_cpu))

        policy_scores_tensor = torch.tensor(policy_scores).to(q_reps.device)
        reference_scores_tensor = full_reference_score_tensor.flatten()
        full_reference_score_tensor.requires_grad = False

        # RLHF 손실 함수 계산
        # 정책 로그 확률 계산
        ratios = policy_scores_tensor / full_reference_score_tensor
        log_ratios = torch.log(ratios + 1e-8)

        # 보상 신호 기반 손실 계산
        rewards = self.beta * log_ratios
        loss = -torch.mean(rewards)
        return loss
