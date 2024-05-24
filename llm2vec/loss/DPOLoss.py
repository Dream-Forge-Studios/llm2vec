import torch
from torch import nn, Tensor
from .loss_utils import cos_sim_single_pairs, mismatched_sizes_all_gather
import torch.nn.functional as F
class DPOLoss():
    def __init__(
        self,
        beta: float = 0.1,
        similarity_fct = cos_sim_single_pairs,
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


        policy_scores = self.similarity_fct(full_q_reps, full_d_reps_pos)

        full_reference_score_tensor = full_reference_score_tensor.flatten()
        full_reference_score_tensor.requires_grad = False

        # RLHF 손실 함수 계산
        # 정책 로그 확률 계산
        loss = F.logsigmoid(policy_scores - full_reference_score_tensor).mean()
        # loss = F.mse_loss(policy_scores, full_reference_score_tensor)
        return loss
