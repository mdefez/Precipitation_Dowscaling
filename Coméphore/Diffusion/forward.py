import torch
import torch.nn.functional as F

class DiffusionScheduler:
    def __init__(self, timesteps, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        self.alpha_bars = self.alpha_bars.to(x_start.device)
        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_ab = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_ab * x_start + sqrt_one_minus_ab * noise
