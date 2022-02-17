# CFAR detection in RD plane
# benyuan liu <byliu@fmmu.edu.cn>
import numpy as np
from scipy import signal


def cfar2d_threshold(img, g_cell=2, t_cell=2, thd=1.33):
    """
    CFAR detector in 2D (RD plane)

    img: N range (319) x M doppler (32) NDArray

    filt_mat = [1 1 1
                1 0 1
                1 1 1]
    with g_cell=0, t_cell=1
    """
    # build 2D filter
    nc = 1 + 2 * g_cell + 2 * t_cell
    filt_mat = np.ones((nc, nc))
    filt_mat[t_cell : t_cell + 2 * g_cell + 1, t_cell : t_cell + 2 * g_cell + 1] = 0
    filt_mat = filt_mat / np.sum(filt_mat)

    # convolve to get THD
    med_img = np.median(img)
    thd_img = signal.convolve2d(
        img, filt_mat, boundary="symm", mode="same", fillvalue=med_img
    )

    return thd_img * thd


def cfar2d(img, r_mask=1, v_cell_skip=4, g_cell=2, t_cell=2, thd=1.414):
    """
    cfar detector in RD plane

    img: N range (319) x M doppler (32) NDArray
    """
    # CFAR threshold
    N, M = img.shape
    img = img[:, v_cell_skip : M - v_cell_skip]
    thd_img = cfar2d_threshold(img, g_cell=g_cell, t_cell=t_cell, thd=thd)

    # do not detect at blind frequency, note: doppler is shifted
    thd_img[:, M // 2 - v_cell_skip] = 1e16
    thd_img[:r_mask] = 1e16
    thd_img[-r_mask:] = 1e16

    # extract cluster points
    r_cell, v_cell = np.where(img >= thd_img)
    v_cell += v_cell_skip
    a = img[img >= thd_img]

    return r_cell, v_cell, a
