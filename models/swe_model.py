from typing import Optional

import torch
import torch.nn as nn
from torch.utils import checkpoint

import hydraulics.simulation_utils as sim_utils

__all__ = ['swe_model']


class SweModel(nn.Module):
    def __init__(self, downsample_model, numerical_solver, coarse_dx,
                 coarse_n_x, alpha, simulation_mode=False):
        super(SweModel, self).__init__()
        self.downsample_model = downsample_model
        self.numerical_solver = numerical_solver
        self.coarse_dx = coarse_dx
        self.coarse_n_x = coarse_n_x
        self.alpha = alpha
        self.simulation_mode = simulation_mode
        self.simulation_current_time = 0
        self.simulation_h_n = []
        self.simulation_t = []

    def set_time_delta(self, dt):
        self.numerical_solver.dt = dt

    def set_boundary_conditions(self, boundary_conditions):
        self.numerical_solver.boundary_conditions = boundary_conditions

    def forward(self, fine_grid_z_n, hidden_h_n, hidden_q_x_n, hidden_q_y_n,
                fine_target_time: torch.Tensor,
                coarse_grid_z_n: Optional[torch.Tensor] = None):
        if coarse_grid_z_n is None:
            coarse_grid_z_n = self.downsample_model(fine_grid_z_n)
        self.simulation_current_time = torch.zeros_like(fine_target_time)
        # TODO(niv): magic numbers 36 & 100.
        delta_t = torch.ones_like(fine_target_time) * 100
        for _ in range(36):
            hidden_h_n, hidden_q_x_n, hidden_q_y_n = checkpoint.checkpoint(
                self.solver, coarse_grid_z_n, hidden_h_n, hidden_q_x_n,
                hidden_q_y_n, delta_t)
        return coarse_grid_z_n, hidden_h_n, hidden_q_x_n, hidden_q_y_n

    def to_gpu(self):
        self.downsample_model = self.downsample_model.cuda()
        self.numerical_solver.to_gpu()

    def solver(self, z_n, h_n, q_x_n, q_y_n, target_time):
        current_time = torch.zeros_like(target_time)
        min_h_n = 0.1 * torch.ones(h_n.shape[0]).cuda()
        i = 0
        while not torch.isclose(current_time, target_time, rtol=0,
                                atol=0.01).all():
            with torch.no_grad():
                dt = sim_utils.cfl(self.coarse_dx, torch.max(
                    h_n.view(h_n.shape[0], -1).max(dim=1).values,
                    min_h_n), self.alpha)
                dt = torch.min(torch.abs(target_time - current_time),
                               dt.squeeze()).reshape_as(dt)
                current_time += dt.squeeze()
                self.simulation_current_time += dt.squeeze()
            h_n, q_x_n, q_y_n = self.numerical_solver(z_n, h_n, q_x_n, q_y_n,
                                                      dt)
            i += 1
            if self.simulation_mode and (i % 10) == 0:
                self.simulation_h_n.append(
                    h_n.clone().detach().squeeze().cpu().numpy())
                self.simulation_t.append(self.simulation_current_time.item())
        return h_n, q_x_n, q_y_n


def swe_model(*args, **kwargs):
    return SweModel(*args, **kwargs)
