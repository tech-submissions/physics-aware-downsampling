import enum
import logging
import multiprocessing
import os
import time
import tqdm

import numpy as np
import pandas as pd
import torch

from hydraulics import boundary
from hydraulics import saint_venant
from hydraulics import simulation_utils as sim_utils

INDEX_PATH = '/home/usgs_dem_data/dem/index.csv'
FLUX_OUTPUT_PATH = '/home/usgs_dem_data/flux_ground_truth'
RAIN_OUTPUT_PATH = '/home/usgs_dem_data/rain_ground_truth'
FLUX_GROUND_TRUTH_INDEX = FLUX_OUTPUT_PATH + '/ground_truth_index.csv'
RAIN_GROUND_TRUTH_INDEX = RAIN_OUTPUT_PATH + '/ground_truth_index.csv'
DX = 1
N_X = 2000
DEVICE = 'cuda'
ALPHA = 0.7
THETA = 0.7
MIN_H_N = 0.01
RAINFALL_DISCHARGE = [0.035 / 3600]  # meters/second 35mm/hour - heavy rain
FLUX_DISCHARGE = [200]  # [m^3/s] cubic meters per second
TARGET_TIME = 3600  # [sec]
INDEX_COLUMN_NAMES = ['dem', 'influx', 'outflux', 'discharge',
                      'simulation time', 'ground_truth', 'alpha', 'theta',
                      'min_h_n']


class GroundTruthType(enum.Enum):
    RAIN, FLUX = range(2)


def read_index_row(index: pd.DataFrame, row: int):
    dem = torch.from_numpy(np.load(index.iloc[row]['dem']))
    dem = dem.reshape(1, 1, *dem.shape)
    dem.sub_(dem.mean())
    return dem


def exec_func(queue: multiprocessing.Queue, file_lock: multiprocessing.Lock,
              device_id: int, ground_truth_type: GroundTruthType):
    time.sleep(device_id)
    logging.basicConfig(filename=f'device_{device_id}.log', level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", )
    torch.cuda.set_device(device_id)
    logging.info(f'GPU {device_id} Online')
    if ground_truth_type == GroundTruthType.FLUX:
        ground_truth_output_path = FLUX_OUTPUT_PATH
        ground_truth_index = FLUX_GROUND_TRUTH_INDEX
        discharge_list = FLUX_DISCHARGE
    else:
        ground_truth_output_path = RAIN_OUTPUT_PATH
        ground_truth_index = RAIN_GROUND_TRUTH_INDEX
        discharge_list = RAINFALL_DISCHARGE
    logging.info(f'Ground truth type - {ground_truth_type.name}')
    logging.info(f'Index path - {INDEX_PATH}')
    logging.info(f'Output path - {ground_truth_output_path}')
    logging.info(f'Output index path - {ground_truth_index}')
    index = pd.read_csv(INDEX_PATH, names=['dem', 'influx', 'outflux'])
    while not queue.empty():
        row = queue.get()
        z_n = read_index_row(index, row)
        output_name = (
            '/'.join(index.iloc[row]['dem'].split('/')[-2:]).split('.')[0])
        for idx, discharge in enumerate(discharge_list):
            output_full_path = os.path.join(ground_truth_output_path,
                                            output_name)
            if ground_truth_type == GroundTruthType.FLUX:
                discharge_text = f'{discharge:.0f}cms'
            else:
                discharge_text = f'{discharge * 3600 * 1000:.0f}mm_hour'
            output_full_path = (
                    output_full_path + f'_{discharge_text}_{TARGET_TIME}_{row}.npy')
            logging.info(f'Processing row {row}')
            logging.info(f'dx - {DX}, n_x - {N_X}, device - {DEVICE}, alpha - '
                         f'{ALPHA}, theta - {THETA}, min_h_n - {MIN_H_N}, '
                         f'target_time - {TARGET_TIME}, discharge - {discharge}')
            calculate_ground_truth(ground_truth_type, output_full_path, z_n,
                                   discharge, index.iloc[row])
            rows = [
                (index.iloc[row]['dem'], index.iloc[row]['influx'],
                 index.iloc[row]['outflux'], discharge, TARGET_TIME,
                 output_full_path, ALPHA, THETA, MIN_H_N)
            ]
            df = pd.DataFrame(rows, columns=INDEX_COLUMN_NAMES)
            file_lock.acquire()
            with open(ground_truth_index, 'a') as f:
                df.to_csv(f, index=False, header=False)
            file_lock.release()
            logging.info(f'Row {row} {idx + 1}/{len(discharge_list)} saved to '
                         'disk and indexed')
    logging.info('GPU {} Done'.format(device_id))


def run_simulation(model: torch.nn.Module, dx: float,
                   z_n: torch.Tensor, h_n: torch.Tensor,
                   q_x_n: torch.Tensor, q_y_n: torch.Tensor,
                   current_time, target_time):
    min_h_n = MIN_H_N * torch.ones(h_n.shape[0]).cuda()

    progress_bar = tqdm.tqdm(total=target_time.item())
    while not torch.isclose(current_time, target_time, rtol=0, atol=0.5).all():
        with torch.no_grad():
            dt = sim_utils.cfl(dx, torch.max(
                h_n.view(h_n.shape[0], -1).max(dim=1).values, min_h_n), ALPHA)
            dt = torch.min(
                torch.abs(current_time - target_time), dt.squeeze()
            ).reshape_as(dt)
            current_time += dt.squeeze()
        progress_bar.update(dt.item())
        h_n, q_x_n, q_y_n = model(z_n, h_n, q_x_n, q_y_n, dt)
        if torch.isnan(h_n).any():
            raise RuntimeError('nan values found in coarse solver.')
    progress_bar.close()
    return h_n, q_x_n, q_y_n


def calculate_ground_truth(ground_truth_type: GroundTruthType, output_path: str,
                           z_n: torch.Tensor, discharge: float,
                           index_row: pd.Series):
    model = saint_venant.SaintVenantFlux(N_X, DX, theta=THETA)
    if ground_truth_type == GroundTruthType.FLUX:
        boundary_conditions = boundary.FluxBoundaryConditions(
            DX, N_X, [eval(index_row['influx'])],
            [eval(index_row['outflux'])], [discharge], 1)
    else:
        discharge = torch.as_tensor(discharge, dtype=torch.float).to(DEVICE)
        boundary_conditions = boundary.RainBoundaryConditions(discharge)
    model.boundary_conditions = boundary_conditions
    model.to_gpu()
    z_n = z_n.to(DEVICE)
    h_n = torch.zeros(z_n.shape[0], z_n.shape[1], N_X, N_X).to(DEVICE)
    q_x_n = torch.zeros(z_n.shape[0], z_n.shape[1], N_X, N_X - 1).to(DEVICE)
    q_y_n = torch.zeros(z_n.shape[0], z_n.shape[1], N_X - 1, N_X).to(DEVICE)
    current_time = torch.zeros(1, dtype=torch.float).to(DEVICE)
    print(f'processing {output_path}')
    for time_target in [TARGET_TIME]:
        time_target = torch.tensor(time_target, dtype=torch.float).to(DEVICE)
        h_n, q_x_n, q_y_n = run_simulation(model, DX, z_n, h_n, q_x_n, q_y_n,
                                           current_time, time_target)
        np.save(output_path, h_n.cpu().numpy().squeeze())


if __name__ == '__main__':
    file_lock = multiprocessing.Lock()
    queue = multiprocessing.Queue()
    index = pd.read_csv(INDEX_PATH, names=['dem', 'influx', 'outflux'])
    ground_truth_type = GroundTruthType.FLUX
    for row in range(len(index)):
        queue.put(row)

    processes_num = 8
    processes = [
        multiprocessing.Process(target=exec_func, args=(queue, file_lock, idx,
                                                        ground_truth_type))
        for idx in range(processes_num)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
