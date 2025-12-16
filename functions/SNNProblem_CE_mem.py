import torch

from functions.utils.utils import vector_to_weights

class SNNProblem_CE_mem:
    def __init__(
        self,
        model,
        X,
        y,
        layers,
        lb,
        ub,
        ce,
        device = "cpu"
    ):
        self.model = model
        self.X = X
        self.y = y
        self.layers = layers
        self.lb = lb
        self.ub = ub
        self.device = device
        self.ce = ce
        self.dim = len(lb)
        

    def get_bounds(self):
        return (self.lb, self.ub)
    
    def get_nx(self):
        return (self.dim)
    
    def get_nobj(self):
        return 1
    
    def has_gradient(self):
        return False

    def fitness(self, w_flat):
        if not isinstance(w_flat, torch.Tensor):
            w_flat = torch.as_tensor(w_flat, dtype=torch.float32, device=self.device)
        else:
            w_flat = w_flat.to(self.device)
             
        vector_to_weights(w_flat, self.layers)

        self.model.eval()
        with torch.no_grad():
            spk, mem = self.model(self.X)          # [T,B,C]
            logits = mem.mean(dim=0)              # [B,C]
            loss = self.ce(logits, self.y)       # CE loss
            
        return [float(loss)]