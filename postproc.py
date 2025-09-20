# batch mode post-process
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.signal import medfilt


def median_filter(x_set, kernel_size=11, pad=3):
    """median filter of v, x_set is Kalman state structure, [r, v, a]"""
    # conver non-equal list to ndarray
    length = max(map(len, x_set))

    empty_target = [[np.nan, np.nan]]  # a empty [r, v] target
    x_mat = np.array([xi.tolist() + empty_target * (length - len(xi)) for xi in x_set])
    nf, nt, _ = x_mat.shape
    xv_filt = deepcopy(x_mat)

    for i in range(nt):
        vi = xv_filt[:, i, 1]
        vi_nan_idx = np.where(np.isnan(vi))[0]

        # fill NaN values forward and backward
        df = pd.DataFrame(vi)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)
        vi = df.values.ravel()

        # median filter
        vi_pad = np.pad(vi, pad, mode="edge")
        vi_filter = medfilt(vi_pad, kernel_size=kernel_size)
        vi_filter = vi_filter[pad : pad + nf]

        # fill NaN back (NaNs are not targets)
        vi_filter[vi_nan_idx] = np.nan
        xv_filt[:, i, 1] = vi_filter

    return xv_filt, x_mat


def mad_select_v(xf, x, thd=0.3):
    """detect outliers of v"""
    xd = np.abs(x - xf)
    x_out = deepcopy(x)
    outliers = xd > thd
    x_out[outliers] = xf[outliers]

    return x_out


def mad_filter_v(x_set, kernel_size=11, pad=5):
    """median value is accepted if its div is larger than a thd"""
    xv_filt, xv_orig = median_filter(x_set, kernel_size=kernel_size, pad=pad)

    nt = xv_orig.shape[1]
    for i in range(nt):
        xv_filt[:, i, 1] = mad_select_v(xv_filt[:, i, 1], xv_orig[:, i, 1])

    return xv_filt
