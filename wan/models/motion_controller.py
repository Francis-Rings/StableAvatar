#!/usr/bin/env python3
import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

class WanMotionControllerModel(torch.nn.Module):
    def __init__(self, freq_dim=256, dim=1536):
        super().__init__()
        self.freq_dim = freq_dim
        self.linear = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        self.init_weight()

    def forward(self, motion_bucket_id):
        emb = sinusoidal_embedding_1d(self.freq_dim, motion_bucket_id * 10)
        emb = self.linear(emb)
        return emb

    def init_weight(self):
        state_dict = self.linear[-1].state_dict()
        state_dict = {i: state_dict[i] * 0 for i in state_dict}
        self.linear[-1].load_state_dict(state_dict)


if __name__ == "__main__":
    dim = 1536
    motion_controller = WanMotionControllerModel()
    motion_bucket_id = 100.0
    motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=torch.float32, device='cpu')
    out = motion_controller(motion_bucket_id).unflatten(1, (6, dim))
    print(out.size())