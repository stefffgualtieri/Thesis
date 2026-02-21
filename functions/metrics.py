import torch

@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    # logits: (N, C), y_true: (N,)
    y_pred = torch.argmax(logits, dim=1)
    return (y_pred == y_true).float().mean().item()


def precision_recall_f1_binary(y_pred: torch.Tensor, y_true: torch.Tensor):
    # y_pred/y_true: (N,) con 0/1 (int o bool)
    y_pred = y_pred.int()
    y_true = y_true.int()

    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    fn = ((y_pred == 0) & (y_true == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision.item(), recall.item(), f1.item()

def macro_precision_recall_f1(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int):
    precisions = []
    recalls = []

    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fp = ((y_pred == c) & (y_true != c)).sum().float()
        fn = ((y_pred != c) & (y_true == c)).sum().float()

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)

        precisions.append(p)
        recalls.append(r)

    precision = torch.mean(torch.stack(precisions))
    recall = torch.mean(torch.stack(recalls))
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1.item()