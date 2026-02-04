import torch

def balanced_indices(dataset, per_class: int, num_classes: int = 10, seed: int = 42):
    g = torch.Generator().manual_seed(seed)

    targets = torch.tensor(dataset.targets)  # MNIST ha .targets
    idxs = []

    for c in range(num_classes):
        class_idx = torch.where(targets == c)[0]
        # shuffle deterministico dentro la classe
        perm = class_idx[torch.randperm(len(class_idx), generator=g)]
        idxs.append(perm[:per_class])

    idxs = torch.cat(idxs)
    # shuffle globale (sempre deterministico)
    idxs = idxs[torch.randperm(len(idxs), generator=g)]
    return idxs.tolist()