import numpy as np
import io
import os
from skimage.external import tifffile
from typing import Sequence, Text, Optional
import pandas as pd
import utils
import logging

SOURCE_PATH = '/home/usgs_dem_data_source'
PATH = '/home/usgs_dem_data/dem'
TILE_SIZE = 2000
ALLOWED_MASKED_PERCENTAGE = 0
MAX_TOPOGRAPHY_DIFFERENCE = 100
INFLUX_LENGTH = 400
OUTFLUX_LENGTH = 400


def read_source_files(root_dir: Text):
    if not os.path.exists(root_dir):
        raise ValueError(f'{root_dir} Does not exists')
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tif'):
                yield os.path.join(root, file)


def process_dem(tif_file_path: Text):
    with open(tif_file_path, 'rb') as f:
        tiffbytes = f.read()
    np_map = tifffile.imread(io.BytesIO(tiffbytes))
    np_ma_map = np.ma.masked_array(np_map, mask=(np_map < -2000))
    np_ma_map = utils.fix_missing_values(np_ma_map)

    tiles = utils.divide_to_tiles(np_ma_map, (TILE_SIZE, TILE_SIZE))
    tiles_filtered = []
    for tile in tiles:
        if (utils.masked_values_percentage(
                tile) <= ALLOWED_MASKED_PERCENTAGE) and (
                utils.topography_difference(
                    tile) <= MAX_TOPOGRAPHY_DIFFERENCE):
            tiles_filtered.append(tile)

    dir_name = os.path.dirname(tif_file_path).split('/')[-1]
    file_name = tif_file_path.split('/')[-1].split('.')[0]
    samples = []
    for index, tile in enumerate(tiles_filtered):
        min_location = utils.find_lowest_point(tile)
        x, y = min_location
        rows, cols = tile.shape
        outflux = (x, y, OUTFLUX_LENGTH)
        if x > 0:
            influx_axis = 0 if y < 0 else -1
            influx = (rows // 2, influx_axis, INFLUX_LENGTH)
        else:
            influx_axis = 0 if x < 0 else -1
            influx = (influx_axis, cols // 2, INFLUX_LENGTH)
        full_path = os.path.join(PATH, dir_name, f'{file_name}_{index}.npy')
        samples.append((full_path, influx, outflux))
        np.save(full_path, np.asarray(tile))

    df = pd.DataFrame(samples, columns=['dem', 'influx', 'outflux'])
    index_path = os.path.join(PATH, 'index.csv')
    with open(index_path, 'a') as f:
        df.to_csv(f, index=False, header=False)


if __name__ == '__main__':
    logging.basicConfig()
    for tif_name in read_source_files(SOURCE_PATH):
        logging.info('Processing %s', tif_name)
        process_dem(tif_name)
