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
        reference_q_reps: Tensor,
        reference_d_reps_pos: Tensor,
        reference_d_reps_neg: Tensor,
    ):
        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)

            reference_full_d_reps_pos = mismatched_sizes_all_gather(reference_d_reps_pos)
            reference_full_d_reps_pos = torch.cat(reference_full_d_reps_pos)

            reference_full_q_reps = mismatched_sizes_all_gather(reference_q_reps)
            reference_full_q_reps = torch.cat(reference_full_q_reps)

            reference_full_d_reps_neg = mismatched_sizes_all_gather(reference_d_reps_neg)
            reference_full_d_reps_neg = torch.cat(reference_full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

            reference_full_d_reps_pos = reference_d_reps_pos
            reference_full_q_reps = reference_q_reps
            reference_full_d_reps_neg = reference_d_reps_neg

        pos_scores = self.similarity_fct(full_q_reps, full_d_reps_pos)
        neg_scores = self.similarity_fct(full_q_reps, full_d_reps_neg)
        reference_pos_scores = self.similarity_fct(reference_full_q_reps, reference_full_d_reps_pos)
        reference_neg_scores = self.similarity_fct(reference_full_q_reps, reference_full_d_reps_neg)

        pos_ratios = pos_scores / reference_pos_scores
        pos_log_ratios = torch.log(pos_ratios + 1e-8)

        neg_ratios = neg_scores / reference_neg_scores
        neg_log_ratios = torch.log(neg_ratios + 1e-8)
        logits = pos_log_ratios - neg_log_ratios

        loss = -F.logsigmoid(self.beta * logits)
        return loss
