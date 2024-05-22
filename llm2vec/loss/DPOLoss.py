import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather
import torch.nn.functional as F
class DPOLoss():
    def __init__(
        self,
        beta: float = 0.1,
        similarity_fct = cos_sim,
    ):
        self.beta = beta
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor,
    ):
        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        policy_scores = self.similarity_fct(full_q_reps, full_d_reps_pos)
        reference_scores = full_d_reps_neg

        ratios = policy_scores / reference_scores
        log_ratios = torch.log(ratios + 1e-8)

        loss = -F.logsigmoid(self.beta * log_ratios)
        return loss
