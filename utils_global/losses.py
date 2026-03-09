import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def softmax_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    """
    """
    inputs = inputs.softmax(dim=0)
    inputs = inputs.flatten(-2, -1)
    targets = targets.flatten(-2, -1)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    # reduction = "sum" if num_items_in_batch is not None else "mean"
    reduction = "none"
    loss = F.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction).mean(dim=-1)  # shape (batch_size, seq_len)
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss

