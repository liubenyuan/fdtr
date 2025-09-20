# main for ktb
import sys
import os
import time
from copy import deepcopy

import numpy as np
from scipy.io import loadmat
import pickle

from detector import DBT
from tracker import Tracker, Link, Track, model_ca, model_dbt
from postproc import mad_filter_v
from score import ktb_save


version = "offline"

if version == "offline":
    verbose = True
    dump_pickle = True
    gate_name = "data_gate"
else:
    verbose = False
    dump_pickle = False
    gate_name = "radar_data_gate"  # on-line key value

argc = len(sys.argv)
if argc != 4:
    #raise Exception("usage: main.py folder data_id output.txt")
    arg_folder = 'data/'
    arg_id = 'data1'         #改数据只需要修改这里
    arg_output = f"data/{arg_id}/output_{arg_id}.txt"
else:
    arg_folder = sys.argv[1]
    arg_id = sys.argv[2]
    arg_output = sys.argv[3]
if version == "offline":
    gate_name = f"{arg_id}_gate"
if verbose:
    print(arg_folder, arg_id, arg_output)

# dataset and gate files
data_file = os.path.join(arg_folder, arg_id, "{0}.mat".format(arg_id))
gate_file = os.path.join(arg_folder, arg_id, "{0}_gate.mat".format(arg_id))

"""
load data and gate
"""
df = loadmat(data_file)
d = df["radar_pulse_squence"]
df_g = loadmat(gate_file)
gate = df_g[gate_name][0]

# radar parameters
n_pulse = 32
n_fft = 2 * n_pulse
fft_interp = int(n_fft / n_pulse)
lamb = 3e8 / 35e9  # 2.99792e8 / 35e9
delta_f = 32e3 / n_fft  # interpolated speed resolution
delta_v = delta_f * lamb / 2.0  # lambd / 2 / PRI*N
# theoretical speed resolution
# delta_v = lamb / (2 * pri_int) = lamb / (2*1ms)
delta_r = 1.875
delta_t = 1e-3  # 1ms
thd = 3.03
# RD parameters
keystone = True
highpass = True
window = "hann"
first_stage = "dft"  # you should enable Keystone if using 'dft'

# RD and detect pipeline,
n_fasttime, tot_pulse = d.shape
tot_frame = int(tot_pulse / n_pulse)
accum_t = 25e-3
accum_frame = int(accum_t / delta_t)
slide_t = 25e-3
slide_frame = int(slide_t / delta_t)
tot_seg = int((tot_frame - accum_frame) / slide_frame) + 1
if verbose:
    print("pulses=%d, frames=%d, segs=%d\n" % (tot_pulse, tot_frame, tot_seg))

# build DBT (+, -)
model = model_dbt(dt=accum_t)
# positive velocity
dbt_p = DBT(model=model, init=Track)
dbt_p.setup(
    delta_v,
    v_sign=1,
    n_pulse=n_pulse,
    n_fft=n_fft,
    keystone=keystone,
    window=window,
    highpass=highpass,
)
# negative velocity
dbt_n = DBT(model=model, init=Track)
dbt_n.setup(
    delta_v,
    v_sign=-1,
    n_pulse=n_pulse,
    n_fft=n_fft,
    keystone=keystone,
    window=window,
    highpass=highpass,
)

# transform and detect using both positive and negative hypothesis
x_set_p = []
x_set_n = []
for i in range(tot_seg):
    if verbose:
        print("now process %d/%d" % (i, tot_seg))
    tic = time.time()

    # extract segment
    seg_len = n_pulse * accum_frame
    slide_len = n_pulse * slide_frame
    idx_seg = i * slide_len + np.arange(seg_len)  # 32*N pulses
    idx_frame = i * slide_frame + np.arange(accum_frame)

    # data and gate
    d_seg = d[:, idx_seg]
    gate_seg = gate[idx_frame]

    # compute detector
    xi_p = dbt_p.transform(d_seg, gate_seg)
    xi_n = dbt_n.transform(d_seg, gate_seg)

    if verbose:
        print("Detection time: " + str(time.time() - tic) + " sec")
        print([xi_p, xi_n])

    x_set_p.append(xi_p)
    x_set_n.append(xi_n)

# append the last seg (25ms), length of x_set is 81
# phase starts at the beginning of a 25ms segment
x_set_p.append(x_set_p[-1])
x_set_n.append(x_set_n[-1])

# dump x_set
if dump_pickle:
    with open("./data/{0}_pm.pickle".format(arg_id), "wb") as pf:
        pickle.dump([x_set_p, x_set_n], pf)

# 0. combin x_set
x_set = []
for xp, xn in zip(x_set_p, x_set_n):
    xi = np.asarray(xp).tolist()
    for xi_n in xn:
        xi.append(xi_n)
    x_set.append(np.asarray(xi))

# 1. link measurement to data
n_seg = len(x_set)
model = model_ca(dt=accum_t)
data_link = Tracker(model=model, init=Link, verbose=False)
for i, xi in enumerate(x_set):
    xi = np.asarray(xi)
    if len(xi) == 0:  # no target present
        measurement = np.array([[]])
    else:
        measurement = xi[:, :2]
    data_link.update(measurement)

# range rate check and removal, rebuild using linked sets
data_link.range_rate_hypo()
x_set_linked = data_link.to_set(n_seg)
x_set_track = deepcopy(x_set_linked)

# 2. kalman filter formal run
data_track = Tracker(model, init=Track, verbose=False)
x_set_filter = []
for i, xi in enumerate(x_set_track):
    xi = np.asarray(xi)
    if len(xi) == 0:  # no target present
        measurement = np.array([[]])
    else:
        measurement = xi[:, :2]
    data_track.update(measurement)

    # if (i % 2) == 1 and i > 1:
    #    x_set_filter.append(data_track.retrodict(2))
    if (i % 2) == 0 and i > 0:
        x_set_filter.append(data_track.current_prediction())

# 3. average filter (hint: strong targets has high velocity variance!)
x_set_eval = mad_filter_v(x_set_filter)

# 4. save. x_set is Kalman structure: [r, v, a] format, ktb_save autoconvert
ktb_save(arg_output, x_set_eval)
