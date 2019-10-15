# CLUSTER algorithm
# RDSCAN (a DBSCAN like cluster algorithm)
# benyuan liu <byliu@fmmu.edu.cn>
import numpy as np


def rdscan_cluster(p, r_lim=40, d_lim=1.0):
    """
    DBSCAN on RD

    p : Nx2 NDArray, [range, doppler]

    Range error = +- 5m, range cell = 5/1.875 = 2.6
    Doppler error = +- 0.12, doppler cell = 0.12/0.0857142 = 1.4 (2)

    Error range: (5m, 0.12m/s), (10m, 0.5m/s)
    """
    N = len(p)

    # calculate cluster linkage with points within a [r_lim, d_lim] square
    n_cluster = 0  # number of clusters
    c_label = np.zeros(N)  # cluster labels (global index)
    p_idx = np.arange(N)  # point global index

    while (len(p) > 0):
        n_cluster = n_cluster + 1
        new = [0]

        while(len(new) > 0):
            c_label[p_idx[new]] = n_cluster  # update labels recursively
            pc = p[new]  # the new commers of this cluster
            p = np.delete(p, new, axis=0)  # delete rows of point
            p_idx = np.delete(p_idx, new)  # delete index

            new = []
            for i, pi in enumerate(p):
                diff = np.abs(pi - pc)
                vr = (diff[:, 0] < r_lim)
                vd = (diff[:, 1] < d_lim)
                in_test = np.array([br and bd for br, bd in zip(vr, vd)])

                # this point is within a square of either point in pc
                if in_test.any():
                    new.append(i)

    # get linkage for clusters (labels)
    return c_label


def rdscan(p, a=None, label=None, cell=None,
           r_lim=40, d_lim=8.6, method='leader'):
    """
    rdscan with weighted cluster centers

    a is amplitude (absolute values)

    method:
        'leader': choose the center as the one with maximum amplitude
        'weight': choose the center as weighted average of cluster points
    """
    # get clusters
    c = rdscan_cluster(p, r_lim=r_lim, d_lim=d_lim)
    num_target = len(c)

    # get weights
    if a is None:
        w = np.ones(num_target)
    else:
        w = a

    if label is None:
        label = np.zeros(num_target)

    if cell is None:
        cell = np.zeros(num_target)

    # infer the number of clusters
    if len(a) == 0:
        n_cluster = 0
    else:
        n_cluster = int(np.max(c))

    # calculate centers of clusters
    pc = []
    for i in range(n_cluster):
        idx = np.where(c == (i+1))
        pi = p[idx]
        # we favor accumulation results with large amplitude
        ai = w[idx]
        gi = label[idx]
        cell_i = cell[idx]

        amax_idx = np.argmax(ai)
        gc = gi[amax_idx]
        rd_cell = cell_i[amax_idx]
        if method == 'leader':
            ac = np.max(ai)
            rd = pi[amax_idx]
            gc = gi[amax_idx]

        elif method == 'weight':
            wi = ai**2
            wi = wi/np.sum(wi)  # normalize

            rd = np.dot(pi.T, wi)
            ac = np.sum(ai*wi)

        # cluster all informations
        pc.append(np.hstack([rd, ac, gc, rd_cell]))

    return np.array(pc)
