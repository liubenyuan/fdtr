# score function
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


def ktb_save(file, x_set):
    """
    save ktb detectors

    x_set : NDArray
        follows Kalman state structure: [r, v, a]
    """
    file_name = os.path.basename(file)

    with open(file, "w") as fid:
        fid.write("219\t{0}\r\n".format(file_name))
        for t, x in enumerate(x_set):
            # remove NaN rows
            mask = np.any(np.isnan(x), axis=1)
            x = x[~mask]
            # save score
            num_t = len(x)
            fid.write("{0}ms\t{1}\t".format(50 * (t + 1), num_t))
            for i, xi in enumerate(x):
                fid.write(
                    "object:{0}\t{1:.4f}\t{2:.4f}\t".format((i + 1), xi[1], xi[0])
                )
            fid.write("\r\n")


def ktb_load(file):
    """
    load ktb detect results

    returns follows KTB structure: [v, r]
    """
    df = pd.read_csv(file, skiprows=1, header=None, delimiter=r"\s+")

    x_set_vr = []
    for index, row in df.iterrows():
        v = []
        num_t = row[1]
        for t in range(num_t):
            vr = [row[1 + 3 * t + 2], row[1 + 3 * t + 3]]
            v.append(vr)
        x_set_vr.append(v)

    return x_set_vr


def ktb_loadmat(file_mat):
    """
    load ktb detect results from .mat file

    returns follows KTB structure: [v, r]
    """
    df = loadmat(file_mat)

    x_set = []
    for i, num_t in enumerate(df["NumT"][0]):
        v = []
        for t in range(num_t):
            tr = "r{0}".format(t + 1)
            tv = "v{0}".format(t + 1)
            vt = [df[tv][0][i], df[tr][0][i]]
            v.append(vt)
        x_set.append(v)

    return x_set


def ktb_online_loadmat(file_mat):
    """
    load ktb detect results from .mat file [online version]

    returns follows KTB structure: [v, r]
    """
    df = loadmat(file_mat)

    x_set = []
    v_all = df["v"]
    r_all = df["r"]
    for i, num_t in enumerate(df["numT"][0]):
        v = []
        for t in range(num_t):
            vt = [v_all[t, i], r_all[t, i]]
            v.append(vt)
        x_set.append(v)

    return x_set


def score_ospa(x_true, x):
    """
    calculate scores and types of errors

    x_true, x: structure [v, r]
    """
    score = 0
    n_true = len(x_true)
    n_estim = len(x)

    n_card = n_true - n_estim  # cardinal
    n_miss = max(0, n_card)  # -1 for mis
    score = score - n_miss

    def eval_score(xi_true, xi):
        err_square = np.abs(xi_true - xi)

        # range test
        if err_square[1] <= 5:
            score_r = 1
        elif err_square[1] <= 10:
            score_r = 2
        else:
            score_r = 3

        # velocity test
        if err_square[0] <= 0.12:
            score_v = 1
        elif err_square[0] <= 0.5:
            score_v = 2
        else:
            score_v = 3

        return score_r, score_v

    # calculate scores
    x_true = np.array(x_true)
    score_r = np.zeros(n_estim, dtype=np.int)
    score_v = np.zeros(n_estim, dtype=np.int)
    for i in range(n_estim):
        # find xi against all x_true (nearest)
        x_err = x_true - x[i]
        x_dist = np.sum(x_err**2, axis=1)
        x_idx = np.argmin(x_dist)

        # optim assign
        x_eval = x_true[x_idx]
        ri, vi = eval_score(x_eval, x[i])
        score_r[i], score_v[i] = ri, vi

    n_l0 = 0
    n_l1 = 0
    n_l2 = 0
    for i in range(n_estim):
        """
        +1 within [+-5m, +-0.12m/s]
        +0 within [+-10m, +-0.5m/s]
        -3 outside [+-10m, +-0.5m/s]

        Truth table
            | 0.12 | 0.5 |  X
        --------------------------
         5  |  1   |  0  |  -3
        --------------------------
         10 |  0   |  0  |  -3
        --------------------------
         X  |  -3  |  -3 |  -3
        --------------------------
        """
        if score_r[i] == 1 and score_v[i] == 1:
            score_i = 1
            n_l0 += 1
        elif score_r[i] <= 2 and score_v[i] <= 2:
            score_i = 0
            n_l1 += 1
        else:
            score_i = -3
            n_l2 += 1

        score += score_i

    # calculate the accurance of error types
    n_r0 = np.sum(score_r == 1)
    n_r1 = np.sum(score_r == 2)
    n_r2 = np.sum(score_r == 3)
    n_v0 = np.sum(score_v == 1)
    n_v1 = np.sum(score_v == 2)
    n_v2 = np.sum(score_v == 3)

    target_type = dict()
    target_type["card"] = n_card
    target_type["miss"] = n_miss
    target_type["r0"] = n_r0
    target_type["r1"] = n_r1
    target_type["r2"] = n_r2
    target_type["v0"] = n_v0
    target_type["v1"] = n_v1
    target_type["v2"] = n_v2
    target_type["l0"] = n_l0
    target_type["l1"] = n_l1
    target_type["l2"] = n_l2

    return score, target_type


def ktb_score(file_mat, x_set, version="offline"):
    """
    evaluate KTB scores

    x_set follows KTB structure: [v, r]
    """
    # load ground truth
    if version == "offline":
        x_set_true = ktb_loadmat(file_mat)
    else:
        x_set_true = ktb_online_loadmat(file_mat)

    # check estimates
    n_true = len(x_set_true)
    n_estim = len(x_set)
    if n_true != n_estim:
        print("Error, length mismatch, req=%d, est=%d" % (n_true, n_estim))

    # score and record error types
    score = 10000
    target_type = []
    for xi_true, xi in zip(x_set_true, x_set):
        score_i, target_type_i = score_ospa(xi_true, xi)
        score += score_i
        target_type.append(target_type_i)

    return score, target_type


def ktb_score_plot(target_type):
    """plot and analysis KTB scores"""
    # error types
    n_card = []
    n_miss = []
    n_r0 = []
    n_r1 = []
    n_r2 = []
    n_v0 = []
    n_v1 = []
    n_v2 = []
    n_l0 = []
    n_l1 = []
    n_l2 = []

    for t in target_type:
        n_card.append(t["card"])
        n_miss.append(t["miss"])
        n_r0.append(t["r0"])
        n_v0.append(t["v0"])
        n_r1.append(t["r1"])
        n_v1.append(t["v1"])
        n_r2.append(t["r2"])
        n_v2.append(t["v2"])
        n_l0.append(t["l0"])
        n_l1.append(t["l1"])
        n_l2.append(t["l2"])

    tot_card = np.sum(np.abs(n_card))
    tot_miss = np.sum(n_miss)
    tot_l0 = np.sum(n_l0)
    tot_l1 = np.sum(n_l1)
    tot_l2 = np.sum(n_l2)
    t_str = "card=%d, miss=%d, L0=%d, L1=%d, L2=%d" % (
        tot_card,
        tot_miss,
        tot_l0,
        tot_l1,
        tot_l2,
    )

    fig, ax = plt.subplots(8, figsize=(6, 8))
    fig.subplots_adjust(
        left=0.1, bottom=0.10, right=0.95, top=0.90, wspace=0.1, hspace=0.5
    )
    ax[0].plot(n_card, "-ko", markerfacecolor="none", markersize=7)
    ax[0].grid(True)
    ax[0].legend(["cardinal error"], loc=1)
    ax[1].plot(n_miss, "-bo", markerfacecolor="none", markersize=7)
    ax[1].grid(True)
    ax[1].legend(["miss detection"], loc=1)
    ax[2].plot(n_l0, "-g", markerfacecolor="none")
    ax[2].grid(True)
    ax[2].legend(["<5m, <0.12m/s"], loc=1)
    ax[3].plot(n_l1, "-m", markerfacecolor="none")
    ax[3].grid(True)
    ax[3].legend(["<10m, <0.5m/s"], loc=1)
    ax[4].plot(n_l2, "-r", markerfacecolor="none")
    ax[4].grid(True)
    ax[4].legend([">10m or >0.5m/s"], loc=1)
    ax[5].plot(n_r0, "-gx", markersize=7)
    ax[5].plot(n_v0, "-go", markersize=7)
    ax[5].grid(True)
    ax[5].legend(["<5m", "<0.12m/s"], loc=1)
    ax[6].plot(n_r1, "-mx", markersize=7)
    ax[6].plot(n_v1, "-mo", markersize=7)
    ax[6].grid(True)
    ax[6].legend(["<10m", "<0.5m/s"], loc=1)
    ax[7].plot(n_r2, "-rx", markersize=7)
    ax[7].plot(n_v2, "-ro", markersize=7)
    ax[7].grid(True)
    ax[7].legend(["false alarms"], loc=1)
    ax[7].set_xlabel("segments (50ms)")
    ax[0].set_title(t_str)

    return fig


if __name__ == "__main__":
    subj = "testdata3"

    # load ground truth
    file_mat = "/data/ktb2019/data/{0}_value.mat".format(subj)
    x_set_true = ktb_loadmat(file_mat)

    # load template score file
    score_file = "/data/ktb2019/sc_{0}.txt".format(subj)
    x_set = ktb_load(score_file)

    # evaluate scores
    score, error_type = ktb_score(file_mat, x_set)
    print(score)
    ktb_score_plot(error_type)
