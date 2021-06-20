import enum
import logging
from ast import literal_eval

import numpy as np
import pandas as pd
import torch


class GroundTruthType(enum.Enum):
    RAIN, FLUX = range(2)


FLUX_INDEX_PATH = '/home/usgs_dem_data/flux_ground_truth/'
RAIN_INDEX_PATH = '/home/usgs_dem_data/rain_ground_truth/'
TRAIN_INDEX_NAME = 'train_ground_truth_index.csv'
TEST_INDEX_NAME = 'test_ground_truth_index.csv'

INDEX_MAPPING = {GroundTruthType.RAIN: RAIN_INDEX_PATH,
                 GroundTruthType.FLUX: FLUX_INDEX_PATH}


class USGS(torch.utils.data.Dataset):
    def __init__(self, transform=None, target_transform=None,
                 ground_truth_type=GroundTruthType.RAIN, train_set=True):
        file_name = TRAIN_INDEX_NAME if train_set else TEST_INDEX_NAME
        index_path = INDEX_MAPPING[ground_truth_type] + file_name
        self.ground_truth_type = ground_truth_type
        self.index = pd.read_csv(index_path, names=[
            'dem', 'influx', 'outflux', 'discharge', 'target_time',
            'ground_truth', 'alpha', 'theta', 'min_h_n'])
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = torch.float
        self.resolution = 1  # 1X1 meter resolution
        self.grid_size = 2000  # 2000X2000 pixels
        self.boundary_type = (
            'rain' if ground_truth_type == GroundTruthType.RAIN else 'flux')
        self.simulation = False  # Flag for simulation mode
        self.simulation_sample = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        item = item if not self.simulation else self.simulation_sample
        if self.ground_truth_type == GroundTruthType.RAIN:
            return self._read_rain_index_row(item)
        else:
            return self._read_index_row(item)

    def simulation_mode(self, sample):
        logging.info(f'simulation mode enabled with sample number {sample}')
        self.simulation = True
        self.simulation_sample = sample

    def _read_index_row(self, row: int):
        if self.simulation:
            logging.info('DEM Path: ' + self.index.iloc[row]['dem'])
            logging.info(
                'Ground-Truth Path: ' + self.index.iloc[row]['ground_truth'])
        dem = torch.from_numpy(np.load(self.index.iloc[row]['dem']))
        dem = dem.reshape(1, *dem.shape)

        influx = torch.as_tensor(
            literal_eval(self.index.iloc[row]['influx']), dtype=self.dtype)
        outflux = torch.as_tensor(
            literal_eval(self.index.iloc[row]['outflux']), dtype=self.dtype)
        discharge = torch.as_tensor(self.index.iloc[row]['discharge'],
                                    dtype=self.dtype)
        time_stamp = torch.as_tensor(self.index.iloc[row]['target_time'],
                                     dtype=self.dtype)
        h_n = np.load(self.index.iloc[row]['ground_truth'], allow_pickle=True)
        h_n = torch.from_numpy(h_n).reshape(1, *h_n.shape)
        if self.transform is not None:
            dem = self.transform(dem)
        else:
            dem.sub_(dem.mean())
        if self.target_transform is not None:
            h_n = self.target_transform(h_n)
        return dem, influx, outflux, discharge, time_stamp, h_n

    def _read_rain_index_row(self, row: int):
        dem = torch.from_numpy(np.load(self.index.iloc[row]['dem']))
        dem = dem.reshape(1, *dem.shape)

        rain_fall = torch.as_tensor(self.index.iloc[row]['discharge'],
                                    dtype=self.dtype)
        time_stamp = torch.as_tensor(self.index.iloc[row]['target_time'],
                                     dtype=self.dtype)
        h_n = np.load(self.index.iloc[row]['ground_truth'], allow_pickle=True)
        h_n = torch.from_numpy(h_n).reshape(1, *h_n.shape)
        if self.transform is not None:
            dem = self.transform(dem)
        else:
            dem.sub_(dem.mean())
        if self.target_transform is not None:
            h_n = self.target_transform(h_n)
        return dem, rain_fall, time_stamp, h_n
