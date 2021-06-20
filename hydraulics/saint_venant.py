# Lint as: python3

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

G = 9.8
MANNING_COEFF_FLOODPLAIN = 0.05

_X_AXIS = 3
_Y_AXIS = 2
_EPSILON = 1e-8


class SaintVenantFlux(nn.Module):
    """1D saint venant equations with flux and height variables.
    Implemented based on the papers of Bates et al. and Almeida et al. - "A
    simple inertial formulation of the shallow water equations for efficient
    two-dimensional flood inundation modelling.", "Improving the stability of a
    simple formulation of the shallow water equations for 2-D flood modeling".
    """

    def __init__(self, spatial_samples_number: int, spatial_delta: float,
                 theta: Optional[float] = 1):
        super(SaintVenantFlux, self).__init__()
        self.n_x = spatial_samples_number
        self.dx = spatial_delta
        self.dt = None
        self.theta = theta
        self.q_centered_weights = torch.tensor(
            [(1 - theta) / 2, theta, (1 - theta) / 2]).view(1, 1, 1, 3)
        self.replicate_padding = torch.nn.ReplicationPad1d((1, 1, 0, 0))
        self.derivative_weights = torch.tensor(
            [-1 / spatial_delta, 1 / spatial_delta]).view(1, 1, 1, 2)
        self.average_weights = torch.tensor([0.25, 0.25, 0.25, 0.25]).view(
            1, 1, 2, 2)
        self.minimum_flow = torch.tensor(1e-7)
        self.boundary_conditions = None

    def to_gpu(self):
        self.q_centered_weights = self.q_centered_weights.cuda()
        self.derivative_weights = self.derivative_weights.cuda()
        self.average_weights = self.average_weights.cuda()
        self.minimum_flow = self.minimum_flow.cuda()

    def _q_centered(self, q: torch.Tensor,
                    transpose: Optional[bool] = False) -> torch.Tensor:
        if self.theta == 1:
            return q
        if transpose:
            return F.conv2d(
                self.replicate_padding(q.transpose(_Y_AXIS, _X_AXIS)),
                self.q_centered_weights).transpose(_Y_AXIS, _X_AXIS)
        else:
            return F.conv2d(self.replicate_padding(q), self.q_centered_weights)

    def _q_norm(self, q_x: torch.Tensor, q_y: torch.Tensor, dim: int):
        if dim == _Y_AXIS:
            x = F.conv2d(q_x, self.average_weights, padding=(0, 1))
            return (q_y ** 2 + x ** 2 + _EPSILON) ** 0.5
        if dim == _X_AXIS:
            y = F.conv2d(q_y, self.average_weights, padding=(1, 0))
            return (q_x ** 2 + y ** 2 + _EPSILON) ** 0.5

    def _cross_flow(self, water_level: torch.Tensor, stream_bed: torch.Tensor,
                    dim: Optional[int] = _X_AXIS) -> torch.Tensor:
        if dim == _Y_AXIS:
            return torch.max(torch.max(water_level[:, :, 1:],
                                       water_level[:, :, :-1]) - torch.max(
                stream_bed[:, :, 1:], stream_bed[:, :, :-1]), self.minimum_flow)
        if dim == _X_AXIS:
            return torch.max(torch.max(water_level[:, :, :, 1:],
                                       water_level[:, :, :, :-1]) - torch.max(
                stream_bed[:, :, :, 1:], stream_bed[:, :, :, :-1]),
                             self.minimum_flow)

    def _derivative(self, x: torch.Tensor,
                    transpose: Optional[bool] = False) -> torch.Tensor:
        dim = _X_AXIS if transpose else _Y_AXIS
        return F.conv2d(x.transpose(_Y_AXIS, dim),
                        self.derivative_weights).transpose(_Y_AXIS, dim)

    def forward(
        self, z_n: torch.Tensor, h_n: torch.Tensor, q_x_n: torch.Tensor,
        q_y_n: torch.Tensor, dt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs one numerical step in time of saint-venant equations."""
        if torch.isclose(dt, torch.zeros_like(dt)).all():
            return h_n, q_x_n, q_y_n
        self.dt = dt
        # Momentum equation
        previous_x_flux = self._q_centered(q_x_n)
        previous_y_flux = self._q_centered(q_y_n, transpose=True)

        cross_flow_x = self._cross_flow(h_n + z_n, z_n, dim=_X_AXIS)
        cross_flow_y = self._cross_flow(h_n + z_n, z_n, dim=_Y_AXIS)

        slope_x = self._derivative(h_n + z_n)
        slope_y = self._derivative(h_n + z_n, transpose=True)

        numerator_x = previous_x_flux - G * self.dt.expand_as(
            cross_flow_x) * cross_flow_x * slope_x
        numerator_y = previous_y_flux - G * self.dt.expand_as(
            cross_flow_y) * cross_flow_y * slope_y

        denominator_x = (1 + G * self.dt.expand_as(numerator_x) * (
            MANNING_COEFF_FLOODPLAIN ** 2) * self._q_norm(
            q_x_n, q_y_n, _X_AXIS) / (cross_flow_x ** (7 / 3)))
        denominator_y = (1 + G * self.dt.expand_as(numerator_y) * (
            MANNING_COEFF_FLOODPLAIN ** 2) * self._q_norm(
            q_x_n, q_y_n, _Y_AXIS) / (cross_flow_y ** (7 / 3)))

        q_x_n_next = numerator_x / denominator_x
        q_y_n_next = numerator_y / denominator_y

        # q_x is q_x_n_next expanded with boundary conditions
        delta_h_n, q_x, q_y = self.boundary_conditions(h_n, q_x_n_next,
                                                       q_y_n_next)
        # Continuity equation
        h_n = h_n + self.dt.expand_as(h_n) * delta_h_n
        h_n_next = h_n + self.dt.expand_as(h_n) * (
            self._derivative(-q_x) + self._derivative(-q_y, transpose=True))
        return h_n_next, q_x_n_next, q_y_n_next
