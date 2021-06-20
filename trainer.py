import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from absl import logging

from hydraulics import boundary
from hydraulics import simulation_utils as sim_utils
from utils import meters
from utils import model_utils
from utils import tensorboard as tb
from utils import visualization
from utils import evaluation_viewer
from utils import optimization as optim


def forward(model: nn.Module,
            data_loader: torch.utils.data.DataLoader, args, epoch: int,
            training: bool, simulate: bool
            ) -> Tuple[float, Optional[torch.Tensor]]:
    criterion = args.criterion
    optimizer = args.optimizer if training else None
    regularization_loss = args.regularization
    if args.local_rank >= 0:
        loss_meter = meters.DistributedAverageMeter(args.local_rank)
    else:
        loss_meter = meters.AverageMeter()
    gradient_norm = meters.AverageMeter()
    weight_norm = meters.AverageMeter()
    time_meter = meters.AverageMeter()

    for iteration, sample in enumerate(data_loader):
        model.zero_grad()
        if args.boundary_type == boundary.BoundaryType.FLUX:
            fine_grid_z_n, influx, outflux, discharge, target_time, \
            fine_h_n = sample
            coarse_bc = boundary.FluxBoundaryConditions(
                args.coarse_dx, args.coarse_n_x, influx, outflux, discharge)
        else:
            fine_grid_z_n, rain_fall, target_time, fine_h_n = sample
            rain_fall = rain_fall.cuda()
            coarse_bc = boundary.RainBoundaryConditions(rain_fall)
        if not simulate and args.local_rank >= 0:
            model.module.set_boundary_conditions(coarse_bc)
        else:
            model.set_boundary_conditions(coarse_bc)

        fine_grid_z_n = fine_grid_z_n.cuda()
        target_time = target_time.cuda()
        coarse_h_n = torch.zeros(fine_h_n.shape[0], fine_h_n.shape[1],
                                 args.coarse_n_x, args.coarse_n_x).cuda()
        coarse_q_x_n = torch.zeros(fine_h_n.shape[0], fine_h_n.shape[1],
                                   args.coarse_n_x, args.coarse_n_x - 1).cuda()
        coarse_q_y_n = torch.zeros(fine_h_n.shape[0], fine_h_n.shape[1],
                                   args.coarse_n_x - 1, args.coarse_n_x).cuda()
        start = time.time()
        coarse_grid_z_n = None
        coarse_grid_z_n, coarse_h_n, coarse_q_x_n, coarse_q_y_n = model(
            fine_grid_z_n, coarse_h_n, coarse_q_x_n, coarse_q_y_n,
            target_time, coarse_grid_z_n)
        true_coarse_z_n = sim_utils.downsample(fine_grid_z_n, args.scale_factor)
        true_coarse_h_n = sim_utils.downsample(fine_h_n, args.scale_factor)
        true_water_level = true_coarse_h_n.detach()
        loss = criterion(coarse_h_n, true_water_level.to(coarse_h_n.device))
        if args.local_rank != 0:
            loss_meter.update(loss.item(), n=len(fine_grid_z_n))
        if args.regularization_lambda > 0:
            loss += args.regularization_lambda * optim.lpf_regularization(
                coarse_grid_z_n, true_coarse_z_n, regularization_loss)
        if simulate:
            html_report = evaluation_viewer.export_evaluation_html(
                args.model,
                args.sample,
                loss.item(),
                coarse_grid_z_n,
                true_coarse_z_n,
                coarse_h_n,
                true_water_level)
            with open(args.log_dir + '/evaluation.html', 'w') as f:
                f.write(html_report)
            html_movie = visualization.render_water_simulation_movie(
                model.simulation_h_n,
                coarse_grid_z_n.clone().detach().squeeze().cpu().numpy(),
                model.simulation_t)
            with open(args.log_dir + '/solver_solution.html', 'w') as f:
                f.write(html_movie)
            return loss_meter.average
        if training and loss.requires_grad:
            loss.backward()
            if args.local_rank <= 0:
                gradient_norm.update(model_utils.calc_gradient_norm(model))
                weight_norm.update(model_utils.calc_weight_norm(model))
        end = time.time()
        time_meter.update(end - start)
        if training and loss.requires_grad:
            optimizer.step()
        if args.local_rank >= 0:
            dist.reduce(loss, 0)
            if args.local_rank == 0:
                loss /= args.world_size
                loss_meter.update(loss.item(), n=len(fine_grid_z_n))
        if args.local_rank <= 0 and iteration % 10 == 0:
            logging.info(
                '{phase} - Epoch: [{0}][{1}/{2}]\t'
                'Time {time.val:.3f} ({time.average:.3f})\t'
                'Loss {loss.val:.4f} ({loss.average:.4f})\t'
                'Grad {grad.val:.4f} ({grad.average:.4f})\t'.format(
                    epoch, iteration, len(data_loader),
                    phase='TRAINING' if training else 'EVALUATING',
                    time=time_meter, loss=loss_meter, grad=gradient_norm))
        if args.local_rank <= 0:
            if training and loss.requires_grad:
                tb.log_scalars(epoch * len(data_loader) + iteration,
                               gradient_norm=gradient_norm.average,
                               weight_norm=weight_norm.average)
            if training:
                tb.log_scalars(epoch * len(data_loader) + iteration,
                               train_loss_iteration=loss)
    return loss_meter.average


def train(epoch: int, model: nn.Module,
          data_loader: torch.utils.data.DataLoader, args):
    # switch to train mode
    model.train()
    return forward(model, data_loader, args, epoch, True, False)


def validate(epoch: int, model: nn.Module,
             data_loader: torch.utils.data.DataLoader, args):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        return forward(model, data_loader, args, epoch, False, False)


def simulate(model: nn.Module, data_loader: torch.utils.data.DataLoader, args):
    model.eval()
    with torch.no_grad():
        return forward(model, data_loader, args, 0, False, True)
