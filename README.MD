# Reduced Focal Loss for Multi-class Classification

PyTorch 实现

论文原文: https://arxiv.org/abs/1903.01347

```python
def reduced_focal_loss(logits,
                       labels,
                       gamma=2,
                       threshold=0.5,
                       reduction="mean"):
    """Reduced Focal Loss: https://arxiv.org/pdf/1903.01347v2.pdf"""

    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    log_pt = -ce_loss
    pt = torch.exp(log_pt)

    low_th_weight = torch.ones_like(pt)
    high_th_weight = (1 - pt) ** gamma / (threshold) ** gamma
    weights = torch.where(pt < threshold, low_th_weight, high_th_weight)

    rfl = weights * ce_loss

    if reduction == "sum":
        rfl = rfl.sum()
    elif reduction == "mean":
        rfl = rfl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return rfl
```

