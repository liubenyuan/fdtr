# prototype class for a single target tracker
# benyuan liu <byliu@fmmu.edu.cn>
import numpy as np
from .kalman import KF


class Link(object):
    """ Linker """

    def __init__(self, model, d0, track_id, confirm=0):
        """ Initialize """
        # management
        self.track_id = track_id  # identification of each track object
        self.skipped_frames = 0  # number of frames skipped undetected
        self.tracked_frames = 0  # number of frames tracked
        self.confirm = confirm
        self.trace = []  # trace path
        self.trace_idx = []  # trace global index (optional)

        # model
        self.dt = model['dt']
        self.F = model['F']
        self.A = model['A']

        # init
        d0 = np.asarray(d0)
        self.last_prediction = d0
        self.prediction = d0  # predicted centroids (x,y)
        self.state = np.dot(self.A.T, d0)

    def update(self, d, flag_miss=0, idx=0):
        if flag_miss:
            # missed, update using model
            self.tracked_frames -= 1
            self.skipped_frames += 1

            # infer accel using the latest velocities, a = mean(vdiff)
            rv_vec = np.asarray(self.trace)
            if len(rv_vec) <= 1:  # diff requires at least two velocities
                a = 0
            else:
                v = rv_vec[:, 1]
                a = np.mean(np.diff(v[-5:]))  # latest 5, if any

            # propagate [R, V] with [A]
            self.state[2] = a / self.dt  # replace accel with running average
            state = np.dot(self.F, self.state)
            # pred = state[:2]
            pred = np.dot(self.A, state)
        else:
            self.tracked_frames += 1
            self.skipped_frames = 0

            # detected, always trust detection (alpha=1.0)
            state = np.dot(self.A.T, d)
            # state = np.hstack([d, 0])
            pred = d

        # append new results
        self.state = state
        self.last_prediction = self.prediction
        self.prediction = pred
        self.trace.append(self.prediction)
        self.trace_idx.append(idx)


class Track(object):
    """ Kalman Filter """

    def __init__(self, model, d0, track_id, confirm=0):
        """ Initialize """
        # management
        self.track_id = track_id  # identification of each track object
        self.skipped_frames = 0  # number of frames skipped undetected
        self.tracked_frames = 0  # number of frames tracked
        self.confirm = confirm
        self.trace = []  # trace path
        self.trace_idx = []  # trace global index (optional)

        # model
        d = np.asarray(d0)
        self.KF = KF(model, d)  # KF instance to track this object
        self.A = model['A']
        self.prediction = d  # predicted centroids
        self.state = self.KF.u

    def update(self, d, flag_miss=0, idx=0):
        if flag_miss:  # missed, update using prediction
            self.tracked_frames -= 1
            self.skipped_frames += 1
            self.KF.predict()  # u = np.dot(self.F, self.last_prediction)
            u = self.KF.correct(d, 1)
        else:  # detected, update using detection
            self.tracked_frames += 1
            self.skipped_frames = 0
            self.KF.predict()
            u = self.KF.correct(d, 0)

        # maps state to measurement [r, v]
        pred = np.dot(self.A, u)
        # pred = u[:2]

        # append new results
        self.KF.last_prediction = self.prediction
        self.prediction = pred
        self.state = self.KF.u
        self.trace.append(self.prediction)
        self.trace_idx.append(idx)


class GHK(object):
    """ alpha-beta-gamma filter """

    def __init__(self, model, d0, track_id, confirm=0,
                 alpha=0.9, beta=0.75, gamma=0.15):
        """ Initialize """
        # management
        self.track_id = track_id  # identification of each track object
        self.skipped_frames = 0  # number of frames skipped undetected
        self.tracked_frames = 0  # number of frames tracked
        self.confirm = confirm
        self.trace = []  # trace path
        self.trace_idx = []  # trace global index (optional)

        # model
        d = np.asarray(d0)
        self.last_prediction = d
        self.prediction = d  # predicted centroids (x,y)
        self.state = np.hstack([d, 0])  # assume acceleration=0
        self.dt = model['dt']
        self.F = model['F']
        self.A = model['A']

        # g-h-k filter parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.vec = np.array([alpha, beta, gamma])

    def update(self, d, flag_miss=0, idx=0):
        # propagate state
        next_state = np.dot(self.F, self.state)

        # correct prediction with new data
        if flag_miss:  # missed, update using model
            self.tracked_frames -= 1
            self.skipped_frames += 1
            self.state = next_state
        else:  # detected, update using detection
            self.tracked_frames += 1
            self.skipped_frames = 0

            # innovation
            innov = np.zeros(3)
            innov[:2] = d - next_state[:2]

            # estimate a usig v'
            vdiff = innov[1] / self.dt
            # if np.abs(vdiff) > 50:
            #     vdiff = np.sign(vdiff) * 50
            #     vdiff = 0
            innov[2] = vdiff

            # g-h-k filter
            self.state = next_state + self.vec*innov

        """
        if flag_miss==0:
            print('flag_miss = {}'.format(flag_miss))
            print('prev state')
            print(next_state)
            print('innov')
            print(innov)
            print('state')
            print(self.state)
        """

        # append new results
        pred = self.state[:2]
        self.last_prediction = self.prediction
        self.prediction = pred
        self.trace.append(self.prediction)
        self.trace_idx.append(idx)
