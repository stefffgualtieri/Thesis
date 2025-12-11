import torch

from optimizers.utils import vector_to_weights

class SNNProblem_CE:
    def __init__(self, model, X, y, layers):
        self.model = model
        self.X = X
        self.y = y
        self.layers = layers
        self.ce = torch.nn.CrossEntropyLoss()

    def fitness(self, w_flat):
        w_flat = torch.tensor(w_flat, dtype=torch.float32)
        vector_to_weights(w_flat, self.layers)

        self.model.eval()
        with torch.no_grad():
            spk, _ = self.model(self.X)          # [T,B,C]
            logits = spk.sum(dim=0)              # [B,C]
            loss = self.ce(logits, self.y)       # CE loss

        return [float(loss)]
