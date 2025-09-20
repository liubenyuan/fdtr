# simple gradient phase focus
# benyuan liu <byliu@fmmu.edu.cn>
# 2019-07-01
from scipy import signal
import numpy as np
from scipy.signal.windows import hann

def grad_r(img, blur=False):
    """
    simple second order gradient on Y-axis

    img: NxM NDArray, line is assumed to be placed along the X-axis
    blur: False, blur reduces the contrast of weak edges
    """
    convole_func = signal.convolve2d
    op_sar = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 8.0
    if blur:
        img = convole_func(img, op_sar, boundary="symm", mode="same")
        # img = signal.medfilt2d(img, kernel_size=3)

    # op_vec = np.array([[-1, -1, -1],
    #                    [1, 1, 1],
    #                    [4, 4, 4],
    #                    [1, 1, 1],
    #                    [-1, -1, -1]])

    # op_vec = np.array([[-1, -1, -1],
    #                    [2, 2, 2],
    #                    [-1, -1, -1]])

    # op_vec = np.array([[-1, -2, -1],
    #                    [2, 4, 2],
    #                    [-1, -2, -1]])

    op_vec = np.array([[-1], [2], [-1]])

    # boundary = 'symm', 'fill', 'wrap'
    # med_img = np.median(img)
    filt_img = convole_func(img, op_vec, boundary="symm", mode="same")
    # filt_img = convole_func(filt_img, sar_filt, boundary='symm', mode='same')

    return filt_img


def mmv_grad_r(images, blur=False):
    """
    apply median_line per slice iteratively on each doppler bin

    images: R * V * nframes
    """
    res = np.zeros(images.shape, dtype=images.dtype)
    for i in range(images.shape[1]):
        # print('RD image enhance, seq={0}'.format(i))
        rt = images[:, i, :]
        res[:, i, :] = grad_r(rt, blur=blur)

    return res


def radon_transform(image, v, method="interp", delta_r=1.875, delta_t=1e-3):
    """
    Radon Transform on slowtime axis, grid search of v
    Radon/Hough transform with Fourier integration part (v phase)

    image: n_fasttime x n_frame NDArray
    v: velocity (m/s)
    method: 'interp', 'discrete'

    r = r -/+ vt, divide both side by delta_r, and let t = T*delta_t
    rcell = rcell -/+ dcell * delta_v / delta_r * delta_t * T
          = rcell -/+ scale * T

    note: that v is positive when flying closer
    """
    n_fasttime, n_frame = image.shape
    dv = v / delta_r * delta_t
    r_mat = np.zeros(image.shape, dtype=image.dtype)

    # 120m/s * 50ms = 6m, 6m / 1.875m = 3.2 cell, pad a minimum +- 4  cells
    # we choose pad=5. Note, np.interp do not need padding.
    # pad = 5
    # r_idx = np.arange(n_fasttime+2*pad) - pad
    # image = np.pad(image, [[pad, pad], [0, 0]], 'edge')
    r_idx = np.arange(n_fasttime)

    # phase correction
    lamb = 3e8 / 35e9  # 2.99792e8
    t_idx = np.arange(n_frame)
    phase_correction = np.exp(-1j * 2 * np.pi * 2 * v / lamb * t_idx * delta_t)

    for t in t_idx:
        r_interp = r_idx - dv * t

        # 1. using sinc interpolate, slow
        # rt = sinc_interp(image[:, t], r_idx, r_interp)
        # r_mat[:, t] = rt[pad:pad+n_fasttime]

        # 2. using numpy interp, make sure r_idx is increasing
        # np.all(np.diff(r_idx) > 0)
        t_ind = t  # int(t + n_frame/2)
        rt = np.interp(r_interp, r_idx, image[:, t_ind])
        r_mat[:, t_ind] = rt

        # 2. using discrete approximation, becareful of the boundaries
        # r_interp = np.floor(r_interp).astype(np.int)
        # rt = image[r_interp, t]
        # r_mat[:, t] = rt[pad:pad+n_fasttime]

    # phase correction/compensation
    r_mat = r_mat * phase_correction

    return r_mat


def radon_phase_compensation(
    r_mat, v_sign=1, delta_v="auto", delta_r=1.875, delta_t=1e-3
):
    """
    Radon Fourier Transform on range x n_frame using
    the index of doppler cell as a prior

    r_mat: N fasttime x M doppler banks x T frames, 3D NDArray
    """
    # calculate the theoretical velocity resolution
    if delta_v == "auto":
        lamb = 3e8 / 35e9  # 2.99792e8
        delta_v = lamb / (2 * delta_t)

    # calculate radon transform, traces are r = r - vt
    n_fasttime, n_doppler, n_frame = r_mat.shape
    rft_mat = np.zeros(r_mat.shape, dtype=r_mat.dtype)
    for k in range(n_doppler):
        if v_sign == 1:
            v = k * delta_v  # positive v
        else:
            v = (k - n_doppler) * delta_v  # negative v

        # compensate r = r - vt
        rft_mat[:, k, :] = radon_transform(
            r_mat[:, k, :], v, delta_r=delta_r, delta_t=delta_t
        )

    return rft_mat


def rd_phase_compensation(r_mat, v_sign=1, delta_v="auto", delta_t=1e-3):
    """
    compensate RD MMV phase

    if accel is known a prior, then
    np.exp(-1j*2*np.pi*2*(v*t + 0.5*a*t**2)/lamb)

    a is different in different [range, doppler] cells
    """
    lamb = 3e8 / 35e9  # 2.99792e8
    if delta_v == "auto":
        delta_v = lamb / (2 * delta_t)

    n_fasttime, n_doppler, n_frame = r_mat.shape
    t = np.arange(n_frame) * delta_t
    # a = 0.51429 / (n_frame * delta_t)
    for k in range(n_doppler):
        if v_sign == 1:
            v = k * delta_v  # positive v
        else:
            v = (k - n_doppler) * delta_v  # negative v

        phase_correction = np.exp(-1j * 2 * np.pi * 2 * (v * t) / lamb)
        # phase_correction = np.exp(-1j*2*np.pi*2*(v*t + 0.5*a*t**2)/lamb)

        r_mat[:, k, :] = r_mat[:, k, :] * phase_correction

    return r_mat


def doppler_fourier_transform(r_mat, n_pulse=32, fft_oversample=0):
    """
    doppler fourier transform after radon doppler transform

    Returns: amplitude and index
    """
    n_fasttime, n_doppler, n_frame = r_mat.shape

    # res = np.zeros((n_fasttime, n_doppler), dtype=r_mat.dtype)
    res = np.zeros((n_fasttime, n_doppler))
    res_idx = np.zeros((n_fasttime, n_doppler))

    # frequency bins
    n_slowfft = int(2 ** (np.ceil(np.log2(n_frame)) + fft_oversample))
    fft_interp = int(n_doppler / n_pulse)
    # [note] using theoretical frequency resolution.
    # Theoretical resolution is proportional to accumulation time only.
    # So, if you using (interp) in RD plane, it should be multiply by
    # fft_interp before calculating the real-frequency
    freqs = fft_interp * np.fft.fftfreq(n_slowfft)
    # window = signal.gaussian(n_frame, 16)
    #window = signal.hann(n_frame, sym=False)
    window = hann(n_frame, sym=False)
    # window = np.ones(n_frame)

    for k in range(n_doppler):
        rv_mat = r_mat[:, k, :] * window
        rv_fft = np.fft.fft(rv_mat, n_slowfft, axis=1)

        # extract the amplitudes and locations of max FFT
        amp_ft = np.abs(rv_fft)
        res[:, k] = np.max(amp_ft, axis=1)
        res_idx[:, k] = freqs[np.argmax(amp_ft, axis=1)]
        # for i in range(n_fasttime):
        #     res_idx[i, k] = doppler_fftpeak(rv_fft[i], freqs,
        #                                     method=findpeak)

    return res, res_idx


def rdft(images, delta_v="auto", n_pulse=32, v_sign=1, delta_r=1.875, delta_t=1e-3):
    """[unused] Radon Doppler Fourier Transform"""
    r_mat = radon_phase_compensation(
        images, v_sign=v_sign, delta_v=delta_v, delta_r=delta_r, delta_t=delta_t
    )
    r_sod = mmv_grad_r(r_mat)
    res, res_index = doppler_fourier_transform(r_sod, n_pulse=n_pulse)

    return res, res_index


def radon_resolve_v(rd, delta_v, delta_r=1.875, delta_t=1e-3):
    """[unused] resolve 1st speed ambiguity (+, -) using RFT"""
    # radon accumulation using positive hypothesis
    r_mat = radon_phase_compensation(
        rd, v_sign=1, delta_v=delta_v, delta_r=delta_r, delta_t=delta_t
    )
    r_pos = np.sum(np.abs(r_mat), axis=2)

    # radon accumulation using negative hypothesis
    r_mat = radon_phase_compensation(
        rd, v_sign=-1, delta_v=delta_v, delta_r=delta_r, delta_t=delta_t
    )
    r_neg = np.sum(np.abs(r_mat), axis=2)

    # calculate sign matrix (which hypothesis accumulates more power?)
    r_sign = np.sign(r_pos / r_neg - 1)

    return r_sign


def doppler_fftpeak(fft, freqs=None, method="max"):
    """find peak of FFT spectrum"""
    fft_abs = np.abs(fft)

    if freqs is None:
        freqs = np.fft.fftfreq(len(fft))

    if method == "max":
        f_peak = freqs[np.argmax(fft_abs)]
    elif method == "barycenter":
        min_fft, max_fft = np.min(fft_abs), np.max(fft_abs)
        w = (fft_abs - min_fft) / (max_fft - min_fft)
        w_max = 1.0
        w[w < w_max * 0.6] = 0
        # w[w < 1.5*np.median(w)] = 0
        w = w / np.sum(w)
        f_peak = np.sum(freqs * w)

    return f_peak


def doppler_cpi_estimate(
    rp, fft_interp=1, fft_oversample=0, window="hann", findpeak="max"
):
    """
    Doppler fine-resolution cell estimation on MMV frames

    [note] using theoretical frequency resolution.
    Theoretical resolution is proportional to accumulation time only.
    So, if you using (interp) in RD plane, it should be multiply by
    fft_interp = int(n_doppler / n_pulse) before calculating the real-frequency
    """
    n_frame = len(rp)
    n_slowfft = int(2 ** (np.ceil(np.log2(n_frame)) + fft_oversample))
    freqs = fft_interp * np.fft.fftfreq(n_slowfft)

    # windowing signal before FFT
    if window == "hann":
        window_fcn = hann(n_frame, sym=False)
    elif window == "gauss":
        window_fcn = hann(n_frame, sym=False)
    elif window == "kaiser":
        window_fcn = signal.kaiser(n_frame, 3.5)
    else:
        window_fcn = np.ones(n_frame)
    rp_win = rp * window_fcn

    # find the peak of FFT fine-resoluted spectrum
    rp_fft = np.fft.fft(rp_win, n_slowfft)
    vv = doppler_fftpeak(rp_fft, freqs, method=findpeak)

    return vv


def accel_estimation(rp, fft_interp, delta_v, delta_t=1e-3, findpeak="barycenter"):
    """single segment speed estimation by finite difference"""
    lamb = 3e8 / 35e9
    n_frame = len(rp)

    # divide rp into equally two parts
    if n_frame % 2 == 1:  # odd
        n_part = n_frame // 2 + 1
        n_dist = n_part - 2
        rp1 = rp[:n_part]
        rp2 = rp[n_part - 1 :]
    else:
        n_part = n_frame // 2
        n_dist = n_part
        rp1 = rp[:n_part]
        rp2 = rp[n_part:]
    # print('n_part = {0}, n_dist = {1}'.format(n_part, n_dist))

    # estimate the accelerations by dividing the rp into two parts
    t = np.arange(n_part) * delta_t
    acc = 0.0
    n_boost = 1
    for i in range(n_boost):
        # quadratic phase compensation
        phase_correction = np.exp(-1j * 2 * np.pi * 2 * (0.5 * acc * t**2) / lamb)
        rp1_comp = rp1 * phase_correction
        rp2_comp = rp2 * phase_correction

        # with a fine-resoluted fft grid
        vv_p1 = doppler_cpi_estimate(
            rp1_comp,
            fft_interp=fft_interp,
            fft_oversample=4,
            window="hann",
            findpeak=findpeak,
        )
        vv_p2 = doppler_cpi_estimate(
            rp2_comp,
            fft_interp=fft_interp,
            fft_oversample=4,
            window="hann",
            findpeak=findpeak,
        )

        # estimate the acceleration, we tend to lower the acc
        acc = (vv_p2 - vv_p1) * delta_v / (n_dist * delta_t)
        # print('vv1 = %f, vv2 = %f' % (vv_p1*delta_v, vv_p2*delta_v))
        # print('{0}: acc = {1}'.format(i, acc))

    # limit acceleration
    if np.abs(acc) > 80:
        acc = 0

    return acc


def accel_compensation(rp, acc, fft_interp, delta_v, delta_t=1e-3):
    """acceleration compensation with a known a prior"""
    lamb = 3e8 / 35e9
    n_frame = len(rp)
    t = np.arange(n_frame) * delta_t
    phase_correction = np.exp(-1j * 2 * np.pi * 2 * (0.5 * acc * t**2) / lamb)
    rp_comp = rp * phase_correction
    vv = doppler_cpi_estimate(
        rp_comp, fft_interp=fft_interp, fft_oversample=2, window="hann", findpeak="max"
    )
    v_delta = vv * delta_v

    return v_delta


def speed_accel_estimate(rp, fft_interp, delta_v, delta_t=1e-3):
    """speed and acceleration estimation"""
    acc = accel_estimation(rp, fft_interp, delta_v=delta_v, delta_t=delta_t)
    # final compensation
    v_delta = accel_compensation(rp, acc, fft_interp, delta_v)

    return v_delta, acc
