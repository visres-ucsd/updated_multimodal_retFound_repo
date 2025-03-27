import torch
import torch.nn.functional as F



def sigmoid_focal_loss(inputs: torch.Tensor,
                       targets: torch.Tensor,
                       alpha_neg: float = 0.25,
                       alpha_pos: float = 0.25,
                       gamma: float = 2,
                       reduction: str = "none",
                       logits_flag = False,
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha_neg (float): Weighting factor for negative classes.
        alpha_pos (float): Weighting factor for positive classes.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    if logits_flag:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = "none")
    else:
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

    if logits_flag:
        inputs = torch.sigmoid(inputs)
    
    # Applying labels...
    p_t = inputs * targets + (1 - inputs) * (1 - targets)

    # Focal loss weight params to each component of BCE...
    loss = ce_loss * ((1 - p_t) ** gamma)
    # Additional class weight parameters.....
    if (alpha_neg >=0) & (alpha_pos >=0):
        alpha_t = alpha_pos * targets + alpha_neg * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

