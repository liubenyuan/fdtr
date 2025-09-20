# process fast, slow time into range, doppler image
# benyuan liu <byliu@fmmu.edu.cn>
# 2019-07-01
import numpy as np
import numpy.fft as fft
from scipy import signal
from scipy.signal.windows import hann
from keystone import kt_chirpz
from utils import butter_highpass_filter


def raw_gate_align(d, gate, delta_r=1.875, n_pulse=32):
    """padding raw data"""
    gate_offset = gate - gate[0]
    r_index = gate_offset // delta_r

    # get walking step
    r_walk_max = np.max(r_index)
    r_walk_min = np.min(r_index)
    r_extend = int(r_walk_max - r_walk_min)
    r_offset = int(-r_walk_min)
    r_index = r_index + r_offset  # range index are positive

    # padding every n_pulses
    n_fasttime, tot_pulse = d.shape
    n_frame = tot_pulse // n_pulse
    dp = np.zeros((n_fasttime + r_extend, tot_pulse), dtype=d.dtype)
    for i in range(n_frame):
        n_prev = int(r_index[i])
        n_after = r_extend - n_prev
        # extract the data of this frame
        idx = i * n_pulse + np.arange(n_pulse)
        d_seg = d[:, idx]
        dp_seg = np.pad(d_seg, [[n_prev, n_after], [0, 0]], mode="edge")
        dp[:, idx] = dp_seg

    return dp, r_offset, r_extend


def rd_gate_align(rds, gate, delta_r=1.875):
    """RD image padding with gate walks"""
    gate_offset = gate - gate[0]
    r_index = gate_offset // delta_r

    # get walking step
    r_walk_max = np.max(r_index)
    r_walk_min = np.min(r_index)
    r_extend = int(r_walk_max - r_walk_min)
    r_offset = int(-r_walk_min)
    r_index = r_index + r_offset  # range index are positive

    # pad
    n_fasttime, n_slowtime, n_frame = rds.shape
    rds_pad = np.zeros((n_fasttime + r_extend, n_slowtime, n_frame), dtype=rds.dtype)
    for i in range(n_frame):
        n_prev = int(r_index[i])
        n_after = r_extend - n_prev
        rd_i = rds[:, :, i]
        rd_p = np.pad(rd_i, [[n_prev, n_after], [0, 0]], mode="edge")
        rds_pad[:, :, i] = rd_p

    return rds_pad, r_offset, r_extend


def range_align_xcorr():
    """
    range alignment using correlation

    maximum RCM: 120m/s * 1ms = 0.12m < 1.875m
    """
    pass


def rd_transform(
    x_mat, n_fft=64, window="hann", highpass=False, order=4, plot_filter=False
):
    """
    FFT on axis=1, per row, of x_raw

    x_mat: N range (319) x M pulses (32) NDArray

    process pipeline:
        1. highpass
        2. window
        3. fft
    """

    # 1. high-pass filter, lowcut can be altered
    if highpass:
        x_mat = butter_highpass_filter(x_mat, 3e3, 32e3, order=order)

    # 2. add window for FFT
    n_fasttime, n_slowtime = x_mat.shape
    window_type = window.lower()
    if window_type == "gaussian":
        w = signal.gaussian(n_slowtime, 8.0)
    elif window_type == "hann":
        #w = signal.hann(n_slowtime, sym=False)
        w = hann(n_slowtime, sym=False)
    elif window_type == "hamming":
        w = signal.hamming(n_slowtime)
    else:
        w = np.ones(n_slowtime)
    x_mat = x_mat * w

    # for even numbered fft, i.e., n_fft=8, frequencies are:
    # [0, 1, 2, 3, 4, -3, -2, -1]
    # get a RD image from R matrix
    rd = fft.fft(x_mat, n_fft, axis=1)

    # remove blind frequency.
    # n_fc = int(n_fft/2)
    # fc_w = int(np.ceil(n_fft/320))
    # rd[:, n_fc-fc_w+1:n_fc+fc_w] = 0

    return rd


def rd_mmv(
    x_mat,
    K=0,
    n_pulse=32,
    n_fft=32,
    keystone=False,
    window=None,
    highpass=False,
    order=4,
):
    """
    build rd image sequencies (3-D: n_range * n_fft * n_frame)

    Parameters
    ----------
    x_mat: n_fasttime * tot_pulse matrix (NDArray)
    K: [new!] doppler ambiguity factor
    n_pulse: number of steped frequencies
    n_fft: FFT bins
    """
    n_range, tot_pulses = x_mat.shape
    n_frame = int(tot_pulses / n_pulse)

    if keystone:
        # x_mat = kt_interp(x_mat)
        x_mat = kt_chirpz(x_mat, K=K)

    # building RD image sequences
    rd = np.zeros((n_range, n_fft, n_frame), dtype=complex)
    for i in range(n_frame):
        # extract the i-th segment
        idx = i * n_pulse + np.arange(n_pulse)
        x_seg = x_mat[:, idx]

        rd[:, :, i] = rd_transform(
            x_seg, n_fft=n_fft, window=window, highpass=highpass, order=order
        )

    return rd
