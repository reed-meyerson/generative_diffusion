import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from dataset_operations import MapIDS


class ReverseDiffusion:
    def __init__(
        self,
        model: nn.Module,
        data: IterableDataset,
        beta_min: float = 0.1,
        beta_max: float = 1.0,
        train_eps: float = 10e-5,
        sample_eps: float = 10e-3,
        max_t=3.0,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.model = model
        self.data = data
        self.max_t = max_t
        self.data_example = next(data)

        def f(x: torch.FloatTensor):
            t = self.max_t * torch.rand(1) + self.train_eps
            noise = torch.randn_like(x)
            x_t = self.mu(t) * x + self.sigma(t) * noise
            return ((x_t, t), -noise)

        self.train_dl = MapIDS(f, data)

    def sigma(self, t: torch.FloatTensor) -> torch.FloatTensor:
        delta_beta = self.beta_max - self.beta_min
        beta_min = self.beta_min
        sigma_t_exponent = -0.5 * (t**2.0) * delta_beta - t * beta_min
        sigma_t = 1.0 - torch.exp(sigma_t_exponent)
        return sigma_t

    def mu(self, t: torch.FloatTensor) -> torch.FloatTensor:
        delta_beta = self.beta_max - self.beta_min
        beta_min = self.beta_min
        log_mu_t = -0.25 * (t**2.0) * delta_beta - 0.5 * t * beta_min
        mu_t = torch.exp(log_mu_t)
        return mu_t

    def f(self, x, t):
        return -0.5 * (self.beta_min + t * (self.beta_max - self.beta_min)) * x

    def g_squared(self, t):
        beta = self.beta_min + t * (self.beta_max - self.beta_min)
        exponent = -t * t * (self.beta_max - self.beta_min) - 2.0 * self.beta_min * t
        exponent = torch.Tensor([exponent])
        return beta * (1.0 - torch.exp(exponent))

    def g(self, t):
        return torch.sqrt(self.g_squared(t))

    def score(self, x: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(x, t) / self.sigma(t)

    def train(self):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        prev_100 = []
        for _, ((x, t), y) in zip(range(10000), self.train_dl):
            y_pred = self.model(x, t)
            loss = loss_fn(y_pred, y)
            prev_100.append(loss.detach())
            if len(prev_100) == 100:
                print(sum(prev_100) / 100.0)
                prev_100 = []
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def sample(self):
        with torch.no_grad():
            x = torch.randn_like(self.data_example)
            dt = torch.Tensor([self.sample_eps])
            t = torch.Tensor([self.max_t])
            while t >= 2.0 * dt:
                dx = (
                    self.f(x, t) + self.g_squared(t) * self.score(x, t)
                ) * dt + self.g(t) * torch.sqrt(dt) * torch.randn_like(x)
                x += dx
                t -= dt
        return x
