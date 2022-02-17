# utils for evaluation
# modified by benyuan liu <byliu@fmmu.edu.cn>
# 2019-07-01
import numpy as np
import pandas as pd
import scipy.sparse.linalg as sla
from scipy import signal
from scipy import sparse
import matplotlib.pyplot as plt


def ktb_parameter():
    """
    calculate parameters for KTB2019

    Ka 35 GHz
    delta_r = 1.875m
    delta_r = c/2B, fs = 2*B = c/delta_r, delta_fs = fs/N
    """
    n_fasttime = 320  # 319
    fc = 35e9  # 35G
    delta_r = 1.875  # range resolution
    c = 3e8  # 2.99792e8
    bw = c / (delta_r * 2)
    fs = bw  # 2*bw
    delta_fs = fs / n_fasttime

    # how many time fc folded when sampled at fs
    k_mid = np.floor((fc - fs / 2) / fs)
    k_mid = np.max([0, k_mid])

    # the location of fc (folded by sampling fs) at fasttime samples
    # n_fc = np.round(n_fasttime*fc/fs - (k_mid + 0.5)*n_fasttime)
    n_fc = np.round((fc - fs / 2) % fs / delta_fs)
    n_fc = np.min([n_fasttime, np.max([0, n_fc])])

    par = dict()
    par["fc"] = fc
    par["delta_fs"] = delta_fs
    par["k_mid"] = k_mid
    par["n_fc"] = n_fc
    par["n_fasttime"] = n_fasttime

    return par


def butter_highpass_filter(data, lowcut, fs, order=4, plot=False):
    """butterworth highpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, [low], btype="highpass")
    y = signal.filtfilt(b, a, data)
    # y = signal.lfilter(b, a, data)

    if plot:
        plot_filter(b, a, fs=fs)

    return y


def fir_highpass(x, lowcut, fs, numtaps=11, plot=False):
    """fir highpass filter"""
    hb = signal.firwin(numtaps, cutoff=lowcut, fs=fs, pass_zero=False, window="hann")
    y = signal.filtfilt(hb, 1, x)

    if plot:
        plot_filter(hb, 1, fs=fs)

    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4, plot=False):
    """butterworth filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    y = signal.lfilter(b, a, data)

    if plot:
        plot_filter(b, a, fs=fs)

    return y


def plot_filter(b, a, fs=1):
    """visualize filter response"""
    w, h = signal.freqz(b, a, worN=1000)
    w = w / 1e3  # change to KHz
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot((fs * 0.5 / np.pi) * w, abs(h))
    ax.set_xlabel("Frequency (KHz)")
    ax.set_ylabel("Gain")
    ax.grid(True)
    ax.set_title("Filter response")


def hp_filter(X, lamb=10000, missing_data=True):
    """
    HP filtering, using sparse algos. Missing data are filled with 'ffill'.
    """
    if missing_data:
        dx = pd.DataFrame(X)
        # dx = dx.fillna(method='pad')
        dx = dx.interpolate(method="nearest")
    else:
        dx = X
    w = np.size(X, 0)
    b = [[1] * w, [-2] * w, [1] * w]
    D = sparse.spdiags(b, [0, 1, 2], w - 2, w)
    I_mat = sparse.eye(w)
    B = I_mat + lamb * (D.transpose() * D)
    return sla.dsolve.spsolve(B, dx)


def sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """

    if len(x) != len(s):
        raise (Exception, "x and s must be the same length")

    # Find the period
    T = s[1] - s[0]

    sinc_mat = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    y = np.dot(x, np.sinc(sinc_mat / T))

    return y


def clutter_intensity_function(pos, lc, surveillance_region):
    """
    Clutter intensity function,
    with uniform distribution through the surveillance region, see pg. 8
    :param pos:
    :param lc:
    :param surveillance_region:
    """
    if (
        surveillance_region[0, 0] <= pos[0] <= surveillance_region[0, 1]
        and surveillance_region[1, 0] <= pos[1] <= surveillance_region[1, 1]
    ):

        return lc / (
            (surveillance_region[0, 1] - surveillance_region[0, 0])
            * (surveillance_region[1, 1] - surveillance_region[1, 0])
        )
    else:
        return 0


def z_in_region(z, region):
    """
    test if z is in the volume
    """
    bool_test = [region[i][0] <= zi <= region[i][1] for i, zi in enumerate(z)]

    if sum(bool_test) == len(z):
        return 1.0
    else:
        return 0.0


def volume(region):
    """return the volume of region"""
    v = np.prod(region[:, 1] - region[:, 0])

    return v


def clutter_intensity(z, lc, region):
    return lc * z_in_region(z, region) / volume(region)


def true_tracks_plots(targets_birth_time, targets_death_time, targets_tracks, delta):
    for_plot = {}
    for i, birth in enumerate(targets_birth_time):
        brojac = birth
        x = []
        y = []
        time = []
        for state in targets_tracks[i]:
            x.append(state[0])
            y.append(state[1])
            time.append(brojac)
            brojac += delta
        for_plot[i] = (time, x, y)

    return for_plot


def extract_position_collection(x_set):
    """
    [not used] extract rx, ry state estimates
    """
    pos = []
    for xi in x_set:
        x = []
        for state in xi:
            x.append(state[0:2])
        pos.append(x)
    return pos


def set2plot(x_set, delta):
    """
    x_set is a n_frame x (n x states) list
    extract all the states, named x, y in original implementation and time.
    """
    time = []
    x = []
    y = []
    k = 0
    for xi in x_set:
        for state in xi:
            x.append(state[0])
            y.append(state[1])
            time.append(k)
        k += delta
    return time, x, y


if __name__ == "__main__":
    lowcut = 3e3
    fs = 32e3
    plot = True

    x = np.sin(2 * np.pi * 5.1e3 * np.arange(64) / 32e3)
    xf = butter_highpass_filter(x, lowcut, fs, order=4, plot=plot)
    yf = fir_highpass(x, lowcut, fs, plot=plot)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, "-k")
    ax.plot(xf, "-r")
    ax.plot(yf, "-b")
    ax.grid(True)
