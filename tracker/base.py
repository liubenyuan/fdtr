# simple tracker using multiple kalman filter
# benyuan liu <byliu@fmmu.edu.cn>
# 2019-07-19
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import medfilt
import matplotlib.pyplot as plt


class Tracker(object):
    """Tracker class that updates track vectors of object tracked"""

    def __init__(
        self,
        model,
        init,
        verbose=False,
        dist_thresh=10,
        max_frames_to_skip=10,
        max_skip_prediction=2,
        min_confirm_length=5,
        max_unconfirm_length=5,
        max_trace_length=10000,
        min_violate_num=30,
        track_id=0,
    ):
        """
        Initialize variable used by Tracker class

        dist_thresh: distance threshold. When exceeds the threshold,
                     track will be deleted and new track is created
        max_frames_to_skip: maximum allowed frames to be skipped for
                            the track object undetected
        max_skip_prediction: maximum memory prediction for current_prediction
        min_confirm_length: minimum frames before new track is confirmed
        max_unconfirm_length: maximum missed frames before a unconfirmed
                              track being deleted [not implemented]
        max_trace_lenght: trace path history length
        min_violate_num: minimum number of range rate check errors
        track_id: identification of each track object
        """
        self.model = model
        self.init = init
        self.verbose = verbose
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_skip_prediction = max_skip_prediction
        self.min_confirm_length = min_confirm_length
        self.max_unconfirm_length = max_unconfirm_length
        self.max_trace_length = max_trace_length
        self.min_violate_num = min_violate_num
        self.tracks = []
        self.active_sets = []
        self.track_id = track_id  # global track ID [must start from 0]
        self.frame_idx = -1  # global snapshot index

    def data_associate(self, detections):
        """
        data association
        - Calculate cost using sum of square distance
          between predicted vs detected centroids
        - Using Hungarian Algorithm assign the correct
          detected measurements to predicted tracks
          https://en.wikipedia.org/wiki/Hungarian_algorithm
        """
        # calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.active_sets)
        M = len(detections)
        cost = np.zeros(shape=(N, M))

        # cost matrix (pair wise distances)
        for i, ti in enumerate(self.active_sets):
            for j in range(len(detections)):
                diff = self.tracks[ti].prediction - detections[j]
                if "scale_bins" in self.model:
                    diff = diff * self.model["scale_bins"]
                distance = np.sqrt(np.sum(diff**2))
                # hamming distance
                # distance = r_cell_err + v_cell_err
                cost[i][j] = distance

        # cost = 0.5 * cost
        cost[cost > 1.5 * self.dist_thresh] = 100 * self.dist_thresh

        # using Hungarian Algorithm assign
        # the correct detected measurements to predicted tracks
        row_ind, col_ind = linear_sum_assignment(cost)

        # debugging, association is NOT optimum!
        if self.verbose:
            print("\n")
            print("frame = ", self.frame_idx)
            print("current track status and centers:")
            for i in self.active_sets:
                print(self.tracks[i].confirm, self.tracks[i].prediction)
            print("measurements:")
            print(detections)
            print("cost:")
            print(cost)
            print("links:")
            print(row_ind, col_ind)
            print("\n")

        assignment = -1 * np.ones(N, dtype=int)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # check for cost distance assignment v.s. threshold.
        # if cost is very high then un_assign (delete) the unassign it
        for i in range(N):
            if cost[i][assignment[i]] > self.dist_thresh:
                assignment[i] = -1

        return assignment

    def update(self, detections):
        """
        Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update state, last_prediction and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """
        self.frame_idx += 1

        # initialize, create tracks at time -1 where no tracks vector found
        if len(self.active_sets) == 0:
            for i in range(len(detections)):
                # [todo] always confirm init targets
                track = self.init(self.model, detections[i], self.track_id, confirm=1)
                self.tracks.append(track)
                self.active_sets.append(self.track_id)
                self.track_id += 1

        # data association
        assignment = self.data_associate(detections)

        # now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)
        # start new tracks, these tracks are not assigned,
        # therefore they are initialized but not updated,
        # they will be association next time.
        if len(un_assigned_detects) != 0:
            for new_idx in un_assigned_detects:
                d0 = detections[new_idx]
                track = self.init(self.model, d0, self.track_id, confirm=0)
                # track.update(d0, 1, self.frame_idx)  # assign
                self.tracks.append(track)
                self.active_sets.append(self.track_id)
                self.track_id += 1

        # update KF state, last_prediction and tracks trace
        for i in range(len(assignment)):
            track = self.tracks[self.active_sets[i]]

            # correct using measurements or memory
            if assignment[i] != -1:  # detect
                flag_miss = 0
                d = detections[assignment[i]]
            else:  # miss
                flag_miss = 1
                d = 0
            track.update(d, flag_miss, self.frame_idx)

            # confirm new tracks
            if track.tracked_frames > self.min_confirm_length:
                track.confirm = 1

            # hit maximum traces, remove extra trace
            if len(track.trace) > self.max_trace_length:
                n_trace = len(track.trace)
                n_head = n_trace - self.max_trace_length
                track.trace = track.trace[n_head:]
                track.trace_idx = track.trace_idx[n_head:]

        # if tracks are not detected for long time, deactive them, not delete
        dies = []
        for i in self.active_sets:
            if self.tracks[i].skipped_frames > self.max_frames_to_skip:
                dies.append(i)
        dies_idx = [self.active_sets.index(i) for i in dies]
        # the latest max_frames_to_skip traces are invalid
        for i in dies:
            track = self.tracks[i]
            n_tail = len(track.trace) - self.max_frames_to_skip
            track.trace = track.trace[:n_tail]
            track.trace_idx = track.trace_idx[:n_tail]
        # delete the dies entry in active sets (rebuild the living sets)
        self.active_sets = [i for i in self.active_sets if i not in dies]
        # the state of deactivate track in previous assignment is removed
        assignment = np.delete(assignment, dies_idx)

        return assignment

    def current_state(self):
        """extract the state of activate tracks"""
        x_set = []
        for i in self.active_sets:
            track = self.tracks[i]
            x_set.append(track.state)
        return np.asarray(x_set)

    def current_prediction(self):
        """extract the prediction of confirmed tracks"""
        x_set = []
        for i in self.active_sets:
            track = self.tracks[i]
            if track.confirm == 1 and track.skipped_frames < self.max_skip_prediction:
                # [KTB] minimal false alarms
                x_set.append(track.prediction)
        return np.asarray(x_set)

    def retrodict(self, T=1):
        """retrodiction, smoother, propagate backward"""
        x_set = []
        for tracks in self.tracks:
            if tracks.confirm == 1:
                x = tracks.KF.retrodict(T)
                x_set.append(x)

        return np.asarray(x_set)

    def remove_inactive_tracks(self):
        """remove tracks that are not activated (not in the active sets)"""
        del_tracks = []
        for track in self.tracks:
            if track.track_id not in self.active_sets:
                del_tracks.append(track.track_id)
        for del_track_id in del_tracks:
            for k, track in enumerate(self.tracks):
                if del_track_id == track.track_id:
                    del self.tracks[k]

    def to_set(self, tot_frame):
        """convert traces to set"""
        traces = []
        for track in self.tracks:
            if track.confirm == 1:
                trace = [[]] * tot_frame
                for i, k in enumerate(track.trace_idx):
                    trace[k] = track.trace[i]
                traces.append(trace)

        # 1. allow undetected targets
        # 2. allow empty targets
        # 3. allow varying number of targets
        x_set = []
        for i in range(tot_frame):
            xi = []
            for t in traces:
                if len(t[i]) != 0:  # trace is not empty
                    xi.append(np.asarray(t[i]))
            x_set.append(xi)

        return x_set

    def range_rate_hypo(self, delta_r=1.875):
        """
        check if range rate and the sign of velocity violates
        We DO NOT check if the track is confirmed or not.

        should range rate check goes on-line?
        """
        id_del = []
        for track in self.tracks:
            # loop over all tracks
            trace = np.asarray(track.trace)

            # range rate can not be computed if trace_len = 0 or 1
            len_trace = len(trace)
            if len_trace <= 1:
                id_del.append(track.track_id)
                continue

            range_rate = np.diff(trace[:, 0])
            v = trace[:, 1][1:]

            # range_rate_sign = np.sign(np.sum(range_rate))
            # v_sign = np.sign(np.sum(v))
            range_rate_sign = np.sign(range_rate)
            v_sign = np.sign(v)

            # the detection of range may glitch (noise), mask-off
            range_rate[np.abs(range_rate) < delta_r / 2.5] = 0

            # infer sign on index where range moves
            idx = np.where(range_rate != 0)[0]
            num_violate = np.sum(range_rate_sign[idx] == v_sign[idx])
            if self.verbose:
                print("Violate hypothesis = {}".format(num_violate))

            # decide where the tracks violates
            # violate_threshold = self.min_violate_num
            violate_threshold = len_trace // 4  # 1/4 tracks are violated
            if track.confirm == 0:
                id_del.append(track.track_id)
            elif num_violate > violate_threshold:
                id_del.append(track.track_id)

        for i in id_del:
            for k, track in enumerate(self.tracks):
                if i == track.track_id:
                    del self.tracks[k]

    def speed_rate_hypo(self):
        """
        [test] calculate accelerations and correct speed estimates
        """
        plt.figure()
        for track in self.tracks:
            # loop over all tracks
            trace = np.asarray(track.trace)
            trace_len = len(trace)
            accel = np.diff(trace[:, 1])
            accel = np.append(accel, accel[-1])
            # dt=25ms, 1 ~ 4g
            # accel[accel > 1.0] = 1.0
            # accel[accel < -1.0] = -1.0
            accel = medfilt(accel, kernel_size=9)
            plt.plot(accel)
            # correct velocity
            for i in range(trace_len):
                track.trace[i][1] -= accel[i] / 2.0
