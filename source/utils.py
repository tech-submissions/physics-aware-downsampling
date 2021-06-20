import numpy as np
import pandas as pd
from typing import Optional, Sequence
from ast import literal_eval
import torch


def read_index_row(index: pd.DataFrame, row: int):
    dem = torch.tensor(np.load(index.iloc[row][1]))
    influx = literal_eval(index.iloc[row][2])
    outflux = literal_eval(index.iloc[row][3])
    discharge = index.iloc[row][4]
    time_stamps = literal_eval(index.iloc[row][5])
    return dem, influx, outflux, discharge, time_stamps


def topography_difference(tile: np.ma.masked_array):
    return np.max(tile) - np.min(tile)


def masked_values_percentage(tile: np.ma.MaskedArray):
    return np.sum(tile.mask) / tile.size


def fix_missing_values(tile: np.ma.MaskedArray,
                       masked_value_offset: Optional[float] = 30):
    tile.data[tile.mask] = masked_value_offset + np.max(tile)
    return tile


def divide_to_tiles(image, tile_shape):
    im_rows, im_cols = image.shape
    im_rows -= (im_rows % tile_shape[0])
    im_cols -= (im_cols % tile_shape[1])
    image = image[:im_rows, :im_cols]

    tile_rows, tile_cols = tile_shape
    tiles = image.reshape(im_rows // tile_rows, tile_rows, im_cols // tile_cols,
                          tile_cols)
    return tiles.transpose(0, 2, 1, 3).reshape(-1, tile_rows, tile_cols)


def find_lowest_point(tile: np.ma.MaskedArray):
    rows, cols = tile.shape
    min_index = np.argmin(
        np.ma.concatenate([tile[:, 0], tile[:, -1], tile[0, :],
                           tile[-1, :]]))
    # left, right, up, down
    if min_index < rows:
        return min_index, 0
    if min_index < 2 * rows:
        return min_index - rows, -1
    if min_index < 2 * rows + cols:
        return 0, min_index - 2 * rows
    else:
        return -1, min_index - (2 * rows + cols)
