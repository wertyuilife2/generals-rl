"""Shared categorical grid distribution and WAIT/MOVE greedy decision."""
import torch
from torch.distributions import Categorical
from torch.nn import functional as F


def make_distribution(logits, mask, *, factorized=True, validate_mask=True):
    if mask is None:
        raise ValueError("categorical policies require action_mask")
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape or (validate_mask and not mask.any(dim=-1).all()):
        raise ValueError("action_mask must match logits and contain a legal action in every row")
    if factorized:
        has_move = mask[:, 1:].any(dim=-1, keepdim=True)
        moves = logits[:, 1:].masked_fill(~mask[:, 1:], -torch.inf)
        moves = torch.where(has_move, moves, torch.zeros_like(moves))
        log_moves = F.log_softmax(moves, dim=-1)
        wait = logits[:, :1]
        log_wait = torch.where(has_move, F.logsigmoid(wait), torch.zeros_like(wait))
        log_move_gate = torch.where(mask[:, :1], F.logsigmoid(-wait), torch.zeros_like(wait))
        logits = torch.cat((log_wait, log_move_gate + log_moves), dim=-1)
    return Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min), validate_args=False)


def action_probabilities(logits, mask, *, factorized=True, validate_mask=True):
    return make_distribution(logits, mask, factorized=factorized, validate_mask=validate_mask).probs


def greedy_actions(probabilities, *, factorized=True):
    if not factorized:
        return probabilities.argmax(dim=-1, keepdim=True)
    choose_wait = probabilities[:, :1] >= probabilities[:, 1:].sum(dim=-1, keepdim=True)
    move = probabilities[:, 1:].argmax(dim=-1, keepdim=True) + 1
    return torch.where(choose_wait, torch.zeros_like(move), move)


def select_action(logits, mask, *, factorized=True):
    return greedy_actions(action_probabilities(logits, mask, factorized=factorized), factorized=factorized)
