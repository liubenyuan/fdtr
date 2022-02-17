# top-level module for KTB detector
#  - meta-detector (sequential)
#  - implement tracditional detect-before-track (DBT) class
# benyuan liu <byliu@fmmu.edu.cn>
from copy import deepcopy
import numpy as np

# preprocessing code for Pulsed Doppler Radar
from rd import rd_gate_align, rd_mmv
from focus import (
    rd_phase_compensation,
    radon_phase_compensation,
    mmv_grad_r,
    doppler_fourier_transform,
    accel_estimation,
    accel_compensation,
)
from cfar import cfar2d
from cluster import rdscan
from tracker import Tracker


def meta_detector(
    rd,
    delta_v,
    delta_r=1.875,
    v_sign=1,
    n_pulse=32,
    r_mask=0,
    thd=3.03,
    first_stage="dft",
):
    """multi-stage, cascaded detectors with labels"""
    n_range, n_doppler, n_frame = rd.shape
    fft_interp = n_doppler // n_pulse

    # 1st stage processing and the largest target detector
    if first_stage == "dft":
        # KT is need to remove range cell-velocity migration (RCM)
        r_mat = rd_phase_compensation(rd, v_sign=v_sign, delta_v=delta_v)
        rd_c1, rd_c1_idx = doppler_fourier_transform(
            r_mat, n_pulse=n_pulse, fft_oversample=0
        )
    elif first_stage == "rft":
        r_mat = radon_phase_compensation(rd, v_sign=v_sign, delta_v=delta_v)
        rd_c1, rd_c1_idx = doppler_fourier_transform(
            r_mat, n_pulse=n_pulse, fft_oversample=0
        )
    else:
        raise NotImplementedError("{} not supported".format(first_stage))

    # 1st stage detector
    r_c1, v_c1, a_c1 = cfar2d(
        rd_c1,
        r_mask=r_mask,
        v_cell_skip=3 * fft_interp,
        g_cell=1 * fft_interp,
        t_cell=2,
        thd=1.15 * thd,
    )
    vv_c1 = rd_c1_idx[r_c1, v_c1]
    label_c1 = np.ones_like(a_c1) * 0

    # 2nd stage processing using SOD
    r_sod = mmv_grad_r(r_mat)
    rd_c2, rd_c2_idx = doppler_fourier_transform(
        r_sod, n_pulse=n_pulse, fft_oversample=0
    )

    # 2nd stage detector
    r_c2, v_c2, a_c2 = cfar2d(
        rd_c2,
        r_mask=r_mask,
        v_cell_skip=3 * fft_interp,
        g_cell=2 * fft_interp,
        t_cell=2,
        thd=0.85 * thd,
    )
    vv_c2 = rd_c2_idx[r_c2, v_c2]
    label_c2 = np.ones_like(a_c2) * 1

    # stack results of cascaded detctors
    r_cell = np.hstack([r_c1, r_c2])
    v_cell = np.hstack([v_c1, v_c2])
    a = np.hstack([a_c1, a_c2])
    vv = np.hstack([vv_c1, vv_c2])
    label = np.hstack([label_c1, label_c2])

    # output the location/cell of target
    rd_cell = np.vstack([r_cell, v_cell]).T
    if v_sign == -1:
        v = v_cell - n_doppler
    else:
        v = v_cell
    v = (v + vv) * delta_v
    # r = (r_cell - r_offset)*delta_r - n_frame*1e-3*v + gate[0]
    r = r_cell * delta_r - n_frame * 1e-3 * v
    p = np.vstack([r, v]).T
    c = rdscan(p, a, label=label, cell=rd_cell, d_lim=8.6, method="leader")

    """
    the slot of c:
        - [0, 1, 2] r, v, amplitude
        - [3] group
        - [4, 5] r_cell, v_cell
    """

    return c, [r_mat, r_sod]


def ktb_lemon_detector(
    rd, gate, delta_v, delta_r=1.875, v_sign=1, n_pulse=32, thd=3.03, first_stage="dft"
):
    """[freeze] submission: the lemon detector"""
    # align range cells
    rd_align, r_offset, r_extend = rd_gate_align(rd, gate)
    # mask gate ranges (artifacts of SOD)
    r_mask = r_extend + 3

    # detector
    c, _ = meta_detector(
        rd_align,
        delta_v,
        delta_r=delta_r,
        v_sign=v_sign,
        n_pulse=n_pulse,
        r_mask=r_mask,
        thd=thd,
        first_stage=first_stage,
    )

    # align range with a moving gate
    c[:, 0] = c[:, 0] - r_offset * delta_r + gate[0]
    return c


class DBT(Tracker):
    """traditional Detect Before Track (DBT) class"""

    def setup(
        self,
        delta_v,
        delta_r=1.875,
        delta_t=1e-3,
        v_sign=1,
        n_pulse=32,
        n_fft=64,
        keystone=True,
        window="hann",
        highpass=True,
        thd=3.03,
        first_stage="dft",
    ):
        """
        wrapper for detector

        Parameters:
            delta_v : doppler resolution in RD plane
            delta_r : raw range resolution
            delta_t : frame interval (a frame consists 32 pulses)
            v_sign : sign of velocity (1st degree ambiguity)
            n_pulse : number of pulses of a frame
            n_fft : number of doppler cells
            keystone : keystone transformation of raw data
            window : windowing function before doppler processing
            highpass : moving target identification
            thd : the thresholds of cascaded detectors
            first_stage : first stage detector type, 'dft' or 'rft'
        """

        # radar parameters
        self.delta_v = delta_v
        self.delta_r = delta_r
        self.delta_t = delta_t
        self.v_sign = v_sign
        # calculate ambiguity factor
        if v_sign == 1:
            K = 0
        elif v_sign == -1:
            K = -1
        else:
            raise NotImplementedError("Bad value: sign = {}".format(v_sign))
        self.K = K
        self.n_pulse = n_pulse
        self.n_fft = n_fft
        self.fft_interp = n_fft // n_pulse
        self.keystone = keystone
        self.window = window
        self.highpass = highpass

        # detector parameters
        self.thd = thd
        self.first_stage = first_stage

    def rd_prep(self, d_mat):
        """convert [range, slowtime] to [range, doppler, frame]"""
        rd = rd_mmv(
            d_mat,
            K=self.K,
            n_pulse=self.n_pulse,
            n_fft=self.n_fft,
            keystone=self.keystone,
            window=self.window,
            highpass=self.highpass,
        )

        return rd

    def detect(self, rd, gate):
        """detect data from RD MMV plane"""
        # align range cells
        rd_align, r_offset, r_extend = rd_gate_align(rd, gate)
        # mask gate ranges (artifacts)
        r_mask = r_extend + 3

        # KTB detector
        c, rmat = meta_detector(
            rd_align,
            delta_v=self.delta_v,
            delta_r=self.delta_r,
            v_sign=self.v_sign,
            n_pulse=self.n_pulse,
            r_mask=r_mask,
            thd=self.thd,
            first_stage=self.first_stage,
        )

        # align range with a moving gate
        c[:, 0] = c[:, 0] - r_offset * self.delta_r + gate[0]
        d = c[:, :2]

        return d, c, rmat

    def accel_estimation(self, c, rmat):
        """single segment speed estimation by finite difference"""
        nt = len(c)
        d = np.zeros((nt, 3))

        for i in range(nt):
            ri = int(c[i, 4])
            vi = int(c[i, 5])
            gi = int(c[i, 3])

            # if group is 0, this is a large signal target,
            # we may use 'max' instead 'barycenter' for robust
            if gi == 0:
                findpeak = "max"
            else:
                findpeak = "barycenter"

            # extract target profile
            rmat_i = rmat[gi]
            rp = rmat_i[ri, vi, :]

            acc = accel_estimation(
                rp,
                fft_interp=self.fft_interp,
                delta_v=self.delta_v,
                delta_t=self.delta_t,
                findpeak=findpeak,
            )
            # print('acc = %f' % acc)
            d[i] = np.hstack([c[i, :2], acc])

        return d

    def track(self, d):
        """track targets, R=c[:, 0], V=c[:, 1]"""
        assignments = self.update(d)

        return assignments

    def accel_compensation(self, c, rmat, assignments):
        """refine results by accel compensation"""
        # extract filtered state
        state = self.current_state()
        state_accel = state[:, 2]

        # get dimensions
        n_range, n_doppler, n_frame = rmat[0].shape
        fft_interp = n_doppler // self.n_pulse
        # num_target = len(c)

        c_comp = deepcopy(c)
        for s_idx, d_idx in enumerate(assignments):
            if d_idx != -1:
                ri = int(c[d_idx, 4])
                vi = int(c[d_idx, 5])
                gi = int(c[d_idx, 3])
                acc = state_accel[s_idx]

                # extract target profile
                rmat_i = rmat[gi]
                rp = rmat_i[ri, vi, :]
                dv = accel_compensation(rp, acc, fft_interp, delta_v=self.delta_v)

                # speed ambiguity (1)
                if self.v_sign == -1:
                    v = vi - n_doppler
                else:
                    v = vi

                # cluster
                v = v * self.delta_v + dv

                # [KTB only] heuristic parameters
                # magic value = step of np.fft.fftfreq(25) * delta_v / 2.0
                if self.v_sign == -1:
                    v += 0.04285714285714286
                else:
                    v -= 0.04285714285714286
                # v += acc * (n_frame/1.667) * 1e-3
                # if self.v_sign == -1:
                #     v += acc * (n_frame/1.667) * 1e-3
                # else:
                #     v += acc * (n_frame/1.667) * 1e-3
                # r = (ri - r_offset)*self.delta_r - n_frame*1e-3*v + gate[0]

                # update refined v
                c_comp[d_idx, 1] = v
                # overwrite 'group label' slot with 'acceleration'
                c_comp[d_idx, 3] = acc

        return c_comp

    def transform(self, d_mat, gate):
        """integrated DBT"""
        # 1. convert [range, n_slowtime] to [range, doppler, frame]
        rd = self.rd_prep(d_mat)
        # 2. detect using cascaded detectors
        d, c, rmat = self.detect(rd, gate)
        # [optional] accel estimation
        d_with_acc = self.accel_estimation(c, rmat)
        # 3. track, estimate vel and accel,
        #    find correct assignments to past estimations
        assignments = self.track(d_with_acc)
        # 4. speed and acceleration compensation
        c_comp = self.accel_compensation(c, rmat, assignments)

        # [todo] override amplitude with acceleration
        for i in range(len(c_comp)):
            c_comp[i, 2] = d_with_acc[i, 2]

        return c_comp
