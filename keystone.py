# keystone transform
# benyuan liu <byliu@fmmu.edu.cn>
import numpy as np
import numpy.fft as fft


def chirpz_fast(sig, ratio, A=1):
    """
    chirp z transform on 1D signal

    Chirp z-Transform.
    As described in

    Rabiner, L.R., R.W. Schafer and C.M. Rader.
    The Chirp z-Transform Algorithm.
    IEEE Transactions on Audio and Electroacoustics, AU-17(2):86--92, 1969

    The discrete z-transform,
        X(z) = sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
        z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
        X(z_k) = sum_{n=0}^{N-1} x_n z_k^{-n}

    In our implementation, N=M
    """
    M = np.size(sig)
    L = np.power(2, int(np.ceil(np.log2(2 * M - 1))))

    # [WARNING] phase center: mid-point of sig
    m_idx = np.arange(M)
    # W = np.exp(-2j*ratio*np.pi/M)
    # w_coeff = np.power(A, -m_idx) * np.power(W, m_idx**2/2.0)
    w_coeff = np.exp(-1j * ratio * np.pi / M * (m_idx**2))
    sig_fft = fft.fft(w_coeff * sig, L)

    # coefficient
    #h = np.zeros(L, dtype=np.complex)
    h = np.zeros(L, dtype=complex)
    # h[:M] = np.power(W, -m_idx**2/2.0)
    # h[L-M+1:] = np.power(W, -m_idx[M-1:0:-1]**2/2.0)
    h[:M] = np.conj(w_coeff)
    # m_idx2 = L - np.arange(L-M+1, L)
    m_idx2 = np.arange(M - 1, 0, -1)
    h[L - M + 1 :] = np.exp(1j * ratio * np.pi / M * (m_idx2**2))
    window_fft = fft.fft(h)

    # convolution
    res_fft = fft.ifft(sig_fft * window_fft)
    res_fft = res_fft[:M]
    # res_fft = np.convolve(sig * w_coeff, np.conj(w_coeff), mode='same')

    # [range, slowtime] (ifft can be ignored if you need range doppler)
    res_t = res_fft * w_coeff

    return res_t


def chirpz(sig, ratio, A=1):
    """chirp z transform on 1D signal"""
    M = np.size(sig)
    L = 2 * M - 1
    # [WARNING] phase center: mid-point of sig
    m_idx = np.arange(M)
    w_coeff = A * np.exp(-1j * ratio * np.pi / M * (m_idx**2))
    h = np.zeros(L, dtype=complex)
    h[:M] = np.conj(w_coeff)
    h2 = (L - np.arange(M, L)) ** 2
    h[M:] = A * np.exp(1j * ratio * np.pi / M * h2)

    # convolution
    window_fft = fft.fft(h)
    sig_fft = fft.fft(sig * w_coeff, L)
    res_fft = fft.ifft(sig_fft * window_fft)
    res_fft = res_fft[:M]
    # res_fft = np.convolve(sig * w_coeff, np.conj(w_coeff), mode='same')

    # [range, slowtime] (ifft can be ignored if you need range doppler)
    # res_t = fft.ifft(res_fft * w_coeff)
    res_t = res_fft * w_coeff

    return res_t


def kt_chirpz(x_mat, K=0, fc=35e9, delta_fs=250e3, n_fc=0):
    """
    keystone transformation without interpolation
    [note] align phase center on the mid-point of sig

    References:
        [1] A Keystone Transform Without Interpolation, Daiyin Zhu, 2007
        [2] Research on Implementation of Keystone Transform, (Zh_CN)
            Wang Juan, Zhao Yongbo, 2011
    """
    n_fasttime, n_slowtime = x_mat.shape
    # [warning] not power of 2
    s_mat = fft.fft(x_mat, axis=0)
    k_mat = np.zeros_like(s_mat)
    freq_bins = fft.fftfreq(n_fasttime) * n_fasttime

    for i in range(n_fasttime):
        ratio = fc / (freq_bins[i] * delta_fs + fc)
        ratio_chirpz = 1.0 / ratio
        k_vec = chirpz_fast(s_mat[i], ratio_chirpz)
        k_doppler = fft.ifft(k_vec)

        # compensate for doppler ambiguity
        m_idx = np.arange(len(k_doppler))
        comp = np.exp(1j * 2 * np.pi * K * ratio * m_idx)
        k_comp = k_doppler * comp

        # save
        k_mat[i] = k_comp

    kt_range = fft.ifft(k_mat, axis=0)
    return kt_range


def kt_interp(x_mat, fc=35e9, delta_fs=250e3, n_fc=0):
    """
    Range Cell Migration (RCM) correction of S(f, tau) using
    keystone transform (sinc interpolation).
    for bspline interpolation, see (keystone implementation using MATLAB):
        https://github.com/ghost200802/MatlabProgram/
    also see:
        https://github.com/angeldsLee/research_matlab/

    [note] phase center is the start of frame
    x_mat : N fasttime x M slowtime NDArray

    parameters for KTB:
        delta_r=1.875m, B = 80e6, n_bins = n_fasttime + 1 = 320
        if fs = 2*B = 160e6, delta_fs = fs/320 = 500e3
        if fs = B (IQ),  delta_fs = 250e3, performs better

    References:
        [1] Dim Target Detection Based on Keystone Transform, eq (9)
    """
    n_fasttime, n_slowtime = x_mat.shape

    # convert (range, slowtime) to (range frequency f_tau, slowtime)
    s_mat = fft.fft(x_mat, axis=0)  # per column, n_fasttime axis
    k_mat = np.zeros_like(s_mat)
    freq_bins = fft.fftfreq(n_fasttime) * n_fasttime

    # sinc interpolation
    n_index = np.arange(n_slowtime)
    for m in range(n_fasttime):

        # time scale ratio
        ratio_m = fc / (fc + freq_bins[m] * delta_fs)
        # ALT: using sqrt keystone to reduce QDM (acceleration)
        # ratio = np.sqrt(ratio)

        n_interp = ratio_m * n_index
        # kt_mat[m] = sinc_interp(s_mat[m], n_index, n_interp)
        k_mat[m] = np.interp(n_interp, n_index, s_mat[m])

    # IFFT, transform to range
    k_range = fft.ifft(k_mat, axis=0)

    return k_range
