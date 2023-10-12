import torch
import torch.optim as opt

class WarmUpAdam(opt.Optimizer):
    def __init__(self, d_model, warmup_steps, optimizer):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self._rate = 0 
        self._step = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    
    def step(self):
        self._step += 1
        if self._rate <= 0.05 / ((self.d_model)**0.5):
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p["lr"] = rate
            self._rate = rate
                
            self.optimizer.step()
        else:
            for p in self.optimizer.param_groups:
                p["lr"] = 0.05 / ((self.d_model)**0.5)
            self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        return None

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def rate(self, step = None):
        if step is None:
            step = self._step
        return (self.d_model**(-0.5)) * min(step**(-0.5), step * (self.warmup_steps ** (-1.5)))
    