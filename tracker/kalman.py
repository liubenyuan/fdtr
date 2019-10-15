# simple CV single target Kalman filter
import numpy as np
from numpy import dot
from numpy.linalg import inv


class KF(object):
    """ Kalman Filter class """

    def __init__(self, model, d0=None, dt=0.025):
        """ a simple model """
        # update model parameters
        self.dt = model['dt']  # delta time
        self.A = model['A']  # observation matrix
        self.F = model['F']  # state transition matrix
        self.Q = model['Q']  # process noise matrix
        self.R = model['R']  # observation noise matrix

        # previous states
        self.P = model['P']  # previous covariance matrix
        if d0 is None:
            d0 = np.zeros(2)  # initial measurements
        u0 = np.dot(self.A.T, d0)  # map d back to u
        self.u = u0  # previous state vector
        self.last_prediction = u0
        self.b = np.zeros(2)  # vector of observations

    def predict(self):
        """
        Predict state vector u and variance of uncertainty P (covariance)
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        """
        # Predicted state estimate
        self.u_pred = np.dot(self.F, self.u)
        # Predicted estimate covariance
        self.P_pred = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.last_prediction = self.u_pred  # same last predicted result

        return self.u_pred

    def correct(self, b, flag_miss=False):
        """
        Correct or update state vector u and
        variance of uncertainty P (covariance).

        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """
        if flag_miss:  # missed, infer measurement using state memory
            self.b = np.dot(self.A, self.last_prediction)
        else:  # detected, update using detection
            self.b = b

        C = np.dot(self.A, np.dot(self.P_pred, self.A.T)) + self.R
        K = np.dot(self.P_pred, np.dot(self.A.T, np.linalg.inv(C)))

        innova = self.b - np.dot(self.A, self.u_pred)
        self.u = self.u_pred + np.dot(K, innova)
        self.P = self.P_pred - np.dot(K, np.dot(C, K.T))
        self.last_prediction = self.u

        return self.u

    def retrodict(self, T=1):
        """
        retrodiction

        Kr =
        """
        P = self.P.copy()
        u = self.u.copy()
        for k in range(T):
            P_pred = dot(self.F, dot(P, self.F.T)) + self.Q

            K = dot(dot(P, self.F.T), inv(P_pred))
            u += dot(K, u - dot(self.F, u))
            P += dot(dot(K, P - P_pred), K.T)

        return u
