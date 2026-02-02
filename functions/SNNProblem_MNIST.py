import torch

from functions.utils.utils import vector_to_weights

class SNNProblem_MNIST:
    def __init__(
        self,
        model,
        train_loader,
        layers,
        lb,
        ub,
        ce,
        device = "cpu",
        K=10
    ):
        self.model = model
        self.train_batch = train_loader
        self.layers = layers
        self.lb = lb
        self.ub = ub
        self.device = device
        self.ce = ce
        self.K = K,
        self.batch_cache = None
        self.dim = len(lb)

        
    def set_current_batches(self, batch_list):
        self.batch_cache = batch_list
        
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
            total_loss = 0.0
            total_n = 0
            
            for data, target in self.batch_cache:
                data = data.to(self.device)
                target = target.to(self.device)
                
                spk_rec, _ = self.model(data)          # [T,B,C]
                logits = spk_rec.sum(dim=0)              # [B,C]
                
                batch_loss = self.ce(logits, target)       # CE loss
                bs = data.size(0)
                
                total_loss += batch_loss.item() * bs
                total_n += bs
            
        return [total_loss / total_n]