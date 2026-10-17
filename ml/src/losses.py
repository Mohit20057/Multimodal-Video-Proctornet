"""
Multiclass Focal Loss implementation for imbalanced behavioral anomaly classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class MulticlassFocalLoss(nn.Module):
    """
    Multiclass Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Designed to handle severe class imbalance between normal student/candidate behavior
    and rare behavioral anomalies (secondary screen, external voice assistance, absence).
    """
    def __init__(
        self,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) Raw unnormalized model predictions
            targets: (B,) Ground truth class indices [0 .. C-1]
        Returns:
            focal_loss: scalar tensor if reduction is 'mean' or 'sum'
        """
        # Log-softmax for numeric stability
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Gather target probabilities: (B, 1)
        target_log_p = log_p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_p = p.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - target_p) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = -alpha_t * modulating_factor * target_log_p
        else:
            focal_loss = -modulating_factor * target_log_p

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")
