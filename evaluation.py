import collections
import csv
import datetime
import json
import os
from typing import Dict, NamedTuple

import torch
import torch.nn as nn
from absl import app
from absl import flags
from absl import logging
from torch.utils.data import distributed as dist_data

import trainer
from data import data
from hydraulics import boundary
from hydraulics import saint_venant
from models import averagenet
from models import edge_preserving
from models import detour
from models import swe_model
from utils import optimization
from utils import visualization

_get_dtype = {'float32': torch.float32,
              'float16': torch.float16,
              'float64': torch.float64}

FLAGS = flags.FLAGS

flags.DEFINE_string('comment', '',
                    'Comment for run. Ignored of log_dir is provided')
flags.DEFINE_string('device', 'cuda', 'Device to use.')
flags.DEFINE_string('dtype', 'float32',
                    f'Data type to use. {_get_dtype.keys()}')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_integer('batch_size', 16, 'Batch size. Can be used only when'
                                       ' sample is not provided.')
flags.DEFINE_enum('criterion', 'mse', ['mse', 'smooth_l1', 'inundation'],
                  'Loss function.')
flags.DEFINE_enum('regularization', 'smooth_l1', ['mse', 'smooth_l1'],
                  'Regularization loss function.')
flags.DEFINE_float('regularization_lambda', 0.0, 'Regularization lambda.')
flags.DEFINE_enum('ground_truth_type', 'rain', ['flux', 'rain'],
                  'Type of ground truth.')
flags.DEFINE_float('alpha', 0.7, 'CFL condition coefficient.')
flags.DEFINE_float('theta', 0.7, 'q centered weighting. [0,1].')
flags.DEFINE_integer('ds_factor', 16, 'Downsample factor from fine to coarse.')
flags.DEFINE_enum('model_type', 'averaging', ['edge_preserve', 'averaging'],
                  'Down sample model')
flags.DEFINE_string('model', '', 'Path to model to use. Default is AverageNet')
flags.DEFINE_integer('sample', None, 'Sample number to simulate on.')
flags.DEFINE_boolean('use_train_set', False,
                     'Use train set as index instead of test set.')


def _get_criterion(criterion: str):
    return {'mse': nn.MSELoss, 'smooth_l1': nn.SmoothL1Loss,
            'inundation': optimization.InundationLoss}[criterion]()


def _flags_to_dict() -> Dict:
    names = [x.name for x in FLAGS.get_key_flags_for_module('evaluation.py')]
    values = [x.value for x in FLAGS.get_key_flags_for_module('evaluation.py')]
    return {name: value for name, value in zip(names, values)}


def _namedtuple_to_json_file(args: NamedTuple, filename: str):
    """Converts namedtuple to readable dict format and saves it as json file."""
    args_dict = []
    for k, v in args._asdict().items():
        if type(v) in {bool, str, int, float}:
            args_dict.append({'Name': k, 'Value': v})
        elif k == 'criterion' or k == 'regularization':
            args_dict.append({'Name': k, 'Value': type(v).__name__})
    with open(filename, 'w') as f:
        json.dump(args_dict, f)


def _hyper_parameters(coarse_grid_size, coarse_resolution,
                      train_data, log_dir):
    params = _flags_to_dict()

    params['scale_factor'] = coarse_resolution // train_data.resolution
    params['boundary_type'] = boundary.BoundaryType[
        train_data.boundary_type.upper()]
    params['criterion'] = _get_criterion(FLAGS.criterion).to(
        FLAGS.device, dtype=_get_dtype[FLAGS.dtype])
    params['dtype'] = _get_dtype[params['dtype']]
    params['coarse_dx'] = coarse_resolution
    params['coarse_n_x'] = coarse_grid_size
    params['fine_dx'] = train_data.resolution
    params['local_rank'] = -1
    params['log_dir'] = log_dir
    Args = collections.namedtuple('HyperParameters', sorted(params))
    return Args(**params)


def main(_):
    if FLAGS.debug:
        torch.set_printoptions(precision=5, linewidth=230, sci_mode=False)
    FLAGS.alsologtostderr = True
    current_time = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    log_dir = os.path.join('evaluations', current_time + '_' + FLAGS.comment)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.get_absl_handler().use_absl_log_file('Logger', log_dir)

    ground_truth_type = data.GroundTruthType[FLAGS.ground_truth_type.upper()]
    train_data = data.USGS(ground_truth_type=ground_truth_type,
                           train_set=FLAGS.use_train_set)
    simulation_mode = True if FLAGS.sample is not None else False
    if simulation_mode:
        train_data.simulation_mode(FLAGS.sample)
    batch_size = 1 if simulation_mode else FLAGS.batch_size
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=batch_size)
    coarse_grid_size = int(train_data.grid_size // FLAGS.ds_factor)
    coarse_resolution = train_data.resolution * FLAGS.ds_factor
    solver = saint_venant.SaintVenantFlux(coarse_grid_size, coarse_resolution,
                                          FLAGS.theta)
    solver.to_gpu()
    if bool(FLAGS.model):
        checkpoint = torch.load(FLAGS.model)
        use_group_norm = checkpoint['args']['group_norm']
        downsample_model = detour.resnet(use_group_norm=use_group_norm)
        downsample_model.load_state_dict(checkpoint['state_dict'])
        logging.info(f'model state_dict loaded from {FLAGS.model}')
    else:
        if 'edge_preserve' == FLAGS.model_type:
            downsample_model = edge_preserving.edge_preserve()
        else:
            downsample_model = averagenet.AverageNet()
    model = swe_model.SweModel(downsample_model, solver, coarse_resolution,
                               coarse_grid_size, FLAGS.alpha, simulation_mode)

    model = model.cuda()
    args = _hyper_parameters(coarse_grid_size, coarse_resolution,
                             train_data, log_dir)
    logging.info(args)
    logging.info('Number of model parameters: %s',
                 sum([p.numel() for p in model.parameters()]))
    _namedtuple_to_json_file(args, log_dir + '/args.json')
    torch.save(model.downsample_model.state_dict(), log_dir + '/model.pth')

    if simulation_mode:
        loss = trainer.simulate(model, train_data_loader, args)
        logging.info(f'\nResults: Loss {loss:.4f}')
    else:
        loss = trainer.validate(0, model, train_data_loader, args)
        logging.info(f'\nResults: Loss {loss:.5f}')


if __name__ == '__main__':
    #  CUDA_VISIBLE_DEVICES=0 python evaluation.py --debug --comment debug --ground_truth_type flux
    #  --sample 10
    app.run(main)
