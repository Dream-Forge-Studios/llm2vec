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
        d_reps_neg: list,
    ):
        d_reps_neg = torch.tensor(d_reps_neg, device=q_reps.device)
        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_reps = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_reps)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        q_reps_cpu = full_q_reps.cpu().to(torch.float32).detach().numpy()
        d_reps_pos_cpu = full_d_reps_pos.cpu().to(torch.float32).detach().numpy()
        policy_scores = 1 - (paired_cosine_distances(q_reps_cpu, d_reps_pos_cpu))

        policy_scores_tensor = torch.tensor(policy_scores)
        policy_scores_tensor = policy_scores_tensor.to(q_reps.device)

        reference_scores_tensor = full_d_reps_neg.flatten().requires_grad_()

        ratios = policy_scores_tensor / reference_scores_tensor
        log_ratios = torch.log(ratios + 1e-8)

        loss = -F.logsigmoid(self.beta * log_ratios)
        return loss
