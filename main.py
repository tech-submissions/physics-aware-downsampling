import collections
import json
from typing import Dict, NamedTuple

import torch
import torch.distributed as dist
import torch.nn as nn
from absl import app
from absl import flags
from absl import logging
from torch.utils.data import distributed as dist_data

import trainer
import utils.tensorboard as tb
from data import data
from hydraulics import boundary
from hydraulics import saint_venant
from models import detour
from models import swe_model
from utils import optimization

_get_dtype = {'float32': torch.float32,
              'float16': torch.float16,
              'float64': torch.float64}

FLAGS = flags.FLAGS

flags.DEFINE_string('comment', '',
                    'Comment for run. Ignored of log_dir is provided')
flags.DEFINE_string('device', 'cuda', 'Device to use.')
flags.DEFINE_string('dtype', 'float32',
                    f'Data type to use. {_get_dtype.keys()}')
flags.DEFINE_enum('model', 'detour', ['detour'], 'Down sample model')
flags.DEFINE_string('model_init', '',
                    'Path to model weights to be used at initialization.')
flags.DEFINE_boolean('group_norm', False, 'Use groupnorm instead of batchnorm.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_string('lr_milestones', '[]',
                    'Decays the learning rate by gamma once the number of epoch'
                    ' reaches one of the milestones.')
flags.DEFINE_float('lr_gamma', 0.1,
                   'Multiplicative factor of learning rate decay. Used with'
                   ' milestones.')
flags.DEFINE_float('momentum', 0.9, 'Momentum.')
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay.')
flags.DEFINE_float('regularization_lambda', 0.0, 'Regularization lambda.')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_enum('criterion', 'mse', ['mse', 'smooth_l1'], 'Loss function.')
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'sgd'], 'Optimizer.')
flags.DEFINE_enum('regularization', 'smooth_l1', ['mse', 'smooth_l1'],
                  'Regularization loss function.')
flags.DEFINE_enum('ground_truth_type', 'rain', ['flux', 'rain'],
                  'Type of ground truth.')
flags.DEFINE_float('alpha', 0.7, 'CFL condition coefficient.')
flags.DEFINE_float('theta', 0.7, 'q centered weighting. [0,1].')
flags.DEFINE_integer('scale_factor', 16,
                     'Downsample factor from fine to coarse.')

flags.DEFINE_integer('world_size', torch.cuda.device_count(),
                     'number of distributed processes')
flags.DEFINE_integer('local_rank', -1, 'rank of distributed processes')
flags.DEFINE_string('dist_init', 'env://',
                    'init used to set up distributed training')
flags.DEFINE_string('dist_backend', 'nccl', 'distributed backend')

_TRAIN_SEED = 214
_TEST_SEED = 123
# Batch size can be larger for test set.
TEST_BATCH_SIZE = 16


def _get_criterion(criterion: str):
    return {'mse': nn.MSELoss, 'smooth_l1': nn.SmoothL1Loss,
            'inundation': optimization.InundationLoss}[criterion]()


def _get_optimizer(optimizer: str, model: torch.nn.Module, **kwargs):
    return {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}[optimizer](
        model.parameters(), **kwargs)


def _flags_to_dict() -> Dict:
    names = [x.name for x in FLAGS.get_key_flags_for_module('main.py')]
    values = [x.value for x in FLAGS.get_key_flags_for_module('main.py')]
    return {name: value for name, value in zip(names, values)}


def _namedtuple_to_json_file(args: NamedTuple, filename: str):
    """Converts namedtuple to readable dict format and saves it as json file."""
    args_dict = []
    for k, v in args._asdict().items():
        if type(v) in {bool, str, int, float}:
            args_dict.append({'Name': k, 'Value': v})
        elif k == 'optimizer':
            value = {'class': type(v).__name__}
            for key in sorted(v.param_groups[0].keys()):
                if key != 'params':
                    value[key] = v.param_groups[0][key]
            args_dict.append({'Name': k, 'Value': value})
        elif k == 'criterion' or k == 'regularization':
            args_dict.append({'Name': k, 'Value': type(v).__name__})
    with open(filename, 'w') as f:
        json.dump(args_dict, f)


def _hyper_parameters(model, coarse_grid_size, coarse_resolution,
                      train_data) -> NamedTuple:
    params = _flags_to_dict()

    params['PyTorch'] = torch.__version__
    params['dtype'] = _get_dtype[params['dtype']]
    params['coarse_dx'] = coarse_resolution
    params['coarse_n_x'] = coarse_grid_size
    params['fine_dx'] = train_data.resolution
    params['boundary_type'] = boundary.BoundaryType[
        train_data.boundary_type.upper()]

    params['criterion'] = _get_criterion(FLAGS.criterion).to(
        FLAGS.device, dtype=_get_dtype[FLAGS.dtype])
    params['regularization'] = _get_criterion(FLAGS.regularization).to(
        FLAGS.device, dtype=_get_dtype[FLAGS.dtype])
    optimizer_params = {'lr': FLAGS.lr, 'weight_decay': FLAGS.weight_decay,
                        'momentum': FLAGS.momentum}
    if FLAGS.optimizer == 'adam':
        optimizer_params.pop('momentum')
    params['optimizer'] = _get_optimizer(FLAGS.optimizer, model,
                                         **optimizer_params)
    params['local_rank'] = FLAGS.local_rank
    params['world_size'] = FLAGS.world_size
    Args = collections.namedtuple('HyperParameters', sorted(params))
    return Args(**params)


def main(_):
    if FLAGS.debug:
        torch.set_printoptions(precision=5, linewidth=230, sci_mode=False)
        torch.manual_seed(_TRAIN_SEED)
        torch.cuda.manual_seed(_TRAIN_SEED)
    if FLAGS.local_rank >= 0:
        dist.init_process_group(backend=FLAGS.dist_backend,
                                init_method=FLAGS.dist_init,
                                world_size=FLAGS.world_size,
                                rank=FLAGS.local_rank)
        torch.cuda.set_device(FLAGS.local_rank)
    if FLAGS.local_rank <= 0:
        FLAGS.alsologtostderr = True
        tb.init(FLAGS.log_dir, FLAGS.comment)
        logging.get_absl_handler().use_absl_log_file('Logger', tb.get_log_dir())

    ground_truth_type = data.GroundTruthType[FLAGS.ground_truth_type.upper()]
    train_data = data.USGS(ground_truth_type=ground_truth_type, train_set=True)
    test_data = data.USGS(ground_truth_type=ground_truth_type, train_set=False)
    if FLAGS.local_rank <= 0:
        logging.info(f'USGS-1m loaded with {len(train_data)} train samples, '
                     f'{len(test_data)} test samples')
    if FLAGS.local_rank >= 0:
        train_sampler = dist_data.DistributedSampler(train_data,
                                                     seed=_TRAIN_SEED)
        test_sampler = dist_data.DistributedSampler(test_data, seed=_TEST_SEED,
                                                    shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    train_data_loader = torch.utils.data.DataLoader(
        train_data, shuffle=(train_sampler is None), sampler=train_sampler,
        batch_size=FLAGS.batch_size)
    test_data_loader = torch.utils.data.DataLoader(
        test_data, shuffle=False, sampler=test_sampler,
        batch_size=TEST_BATCH_SIZE)
    coarse_grid_size = int(train_data.grid_size // FLAGS.scale_factor)
    coarse_resolution = train_data.resolution * FLAGS.scale_factor
    solver = saint_venant.SaintVenantFlux(coarse_grid_size, coarse_resolution,
                                          FLAGS.theta)
    solver.to_gpu()
    if 'detour' in FLAGS.model:
        downsample_model = detour.resnet(FLAGS.group_norm)
    else:
        raise ValueError('Unsupported model type.')
    if FLAGS.local_rank <= 0:
        logging.info(downsample_model)
    model = swe_model.SweModel(downsample_model, solver, coarse_resolution,
                               coarse_grid_size, FLAGS.alpha)

    model = model.cuda()
    if bool(FLAGS.model_init):
        device = FLAGS.local_rank if FLAGS.local_rank >= 0 else 0
        checkpoint = torch.load(FLAGS.model_init,
                                map_location=torch.device(device))
        model.downsample_model.load_state_dict(checkpoint['state_dict'])
        if FLAGS.local_rank <= 0:
            logging.info('Model initialized with state dict %s',
                         FLAGS.model_init)
    if FLAGS.local_rank >= 0:
        device_ids = [FLAGS.local_rank]
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=device_ids,
                                                    output_device=device_ids[0])
    FLAGS.lr_milestones = eval(FLAGS.lr_milestones)
    args = _hyper_parameters(model, coarse_grid_size, coarse_resolution,
                             train_data)
    if FLAGS.local_rank <= 0:
        logging.info(args)
        logging.info('Number of model parameters: %s',
                     sum([p.numel() for p in model.parameters()]))
        tb.log_hyper_parameters(args._asdict())
        _namedtuple_to_json_file(args, tb.get_log_dir() + '/args.json')

    if FLAGS.lr_milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            args.optimizer, FLAGS.lr_milestones, gamma=FLAGS.lr_gamma,
            verbose=True if FLAGS.local_rank <= 0 else False)
    best_validation_loss = float('inf')
    for epoch in range(FLAGS.epochs):
        if FLAGS.local_rank >= 0:
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        train_loss = trainer.train(epoch, model, train_data_loader, args)
        validation_loss = trainer.validate(epoch, model, test_data_loader, args)
        if FLAGS.lr_milestones:
            scheduler.step()

        if FLAGS.local_rank <= 0:
            if FLAGS.local_rank < 0:
                state_dict = model.downsample_model.state_dict()
            else:
                state_dict = model.module.downsample_model.state_dict()
            torch.save({'epoch': epoch, 'args': args._asdict(),
                        'state_dict': state_dict},
                       tb.get_log_dir() + f'/checkpoint.pth')
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save({'epoch': epoch, 'args': args._asdict(),
                            'state_dict': state_dict},
                           tb.get_log_dir() + f'/best_checkpoint.pth')
            logging.info(
                f'\nResults - Epoch: {epoch}\tTraining Loss {train_loss:.4f}\t'
                f'Validation Loss {validation_loss:.4f}\n')
            tb.log_scalars(epoch, write_hparams=True, train_loss=train_loss,
                           validation_loss=validation_loss)


if __name__ == '__main__':
    #  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29499 main.py --batch_size 1 --debug --epochs 50 --regularization_lambda 0.5
    app.run(main)
