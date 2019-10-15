# kalman filter models
import numpy as np


def model_cv(dt=50e-3, delta_r=1.875, delta_v=2.14285):
    """
    generate a constant velocity model
    """
    model = dict()

    model['dt'] = dt
    model['A'] = np.array([[1, 0],
                           [0, 1]])
    # convert errors to the number of range or doppler cells
    # model['scale_bins'] = np.array([1, 1])
    model['scale_bins'] = np.array([1.0/delta_r, 1.0/delta_v])
    model['F'] = np.array([[1.0, -dt],
                           [0.0, 1.0]])

    sv = 4
    G = np.matrix([[0.5*dt**2],
                   [dt]])
    Q = G*G.T*(sv**2)
    # self.Q = np.eye(2) * (sv ** 2)
    model['Q'] = np.array(Q)

    # observation noise matrix
    se = 1  # m
    model['R'] = np.array([[0.5, 0],
                          [0, 1e-4]]) * (se ** 2)

    # kalman parameters
    model['P'] = np.diag([20, 1])

    return model


def model_ca(dt=50e-3, delta_r=1.875, delta_v=2.14285):
    """
    generate a constant acceleration model

    x = Fx + q, q~N(0, Q)
    y = Ax + r, r~N(0, R)
    """
    model = dict()
    model['dt'] = dt
    model['A'] = np.array([[1, 0, 0],
                           [0, 1, 0]])
    # convert errors to the number of range or doppler cells
    # model['scale_bins'] = np.array([1, 1])
    model['scale_bins'] = np.array([1.0/delta_r, 1.0/delta_v])
    model['F'] = np.array([[1.0, -dt, -0.5*dt**2],
                           [0.0, 1.0, dt],
                           [0.0, 0.0, 1.0]])

    sv = 4
    G = np.matrix([[0.5*dt**2],
                   [dt],
                   [1]])
    Q = G*G.T*(sv**2)
    # self.Q = np.eye(2) * (sv ** 2)
    model['Q'] = np.array(Q)

    # observation noise matrix
    se = 1  # m
    model['R'] = np.array([[0.5, 0],
                          [0, 1e-4]]) * (se ** 2)

    # kalman parameters
    model['P'] = np.diag([20, 1, 1])

    return model


def model_dbt(dt=50e-3, delta_r=1.875, delta_v=2.14285):
    """
    generate a constant jerk model

    x = Fx + q, q~N(0, Q)
    y = Ax + r, r~N(0, R)
    """
    model = dict()
    model['dt'] = dt

    # convert errors to the number of range or doppler cells
    # model['scale_bins'] = np.array([1, 1])
    # a scale of 0.001 means this dimension contributes 0 to dist
    model['scale_bins'] = np.array([1.0/delta_r,
                                    1.0/delta_v,
                                    0.0001])

    # state transition matrix
    model['F'] = np.array([[1.0, -dt, -0.5*dt**2, -1/6*dt**3],
                           [0.0, 1.0, dt, 0.5*dt**2],
                           [0.0, 0.0, 1.0, dt],
                           [0.0, 0.0, 0.0, 1.0]])

    # state noise matrix
    sv = 0.2
    G = np.matrix([[1/6*dt**3],
                   [0.5*dt**2],
                   [dt],
                   [1]])
    Q = G*G.T*(sv**2)
    # self.Q = np.eye(2) * (sv ** 2)
    model['Q'] = np.array(Q)

    # measurement matrix (state -> measurement)
    model['A'] = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])

    # measurement noise matrix
    se = 10
    model['R'] = np.array([[1.875, 0, 0],
                           [0, 0.5, 0],
                           [0, 0, 10]]) * (se ** 2)
    # model['R'] = np.eye(3) * (se ** 2)

    # kalman parameters
    model['P'] = np.diag([400, 400, 400, 100])

    return model


def model_ca_visual(dt=50e-3, delta_r=1.875, delta_v=2.14285):
    """
    [test only] generate a constant acceleration model
    same as model_ca but with [R, V, A] measurements

    x = Fx + q, q~N(0, Q)
    y = Ax + r, r~N(0, R)
    """
    model = dict()
    model['dt'] = dt
    model['A'] = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    # convert errors to the number of range or doppler cells
    # model['scale_bins'] = np.array([1, 1])
    model['scale_bins'] = np.array([1.0/delta_r,
                                    1.0/delta_v,
                                    0.0001])
    model['F'] = np.array([[1.0, -dt, -0.5*dt**2],
                           [0.0, 1.0, dt],
                           [0.0, 0.0, 1.0]])

    sv = 4
    G = np.matrix([[0.5*dt**2],
                   [dt],
                   [1]])
    Q = G*G.T*(sv**2)
    # self.Q = np.eye(2) * (sv ** 2)
    model['Q'] = np.array(Q)

    # observation noise matrix
    se = 1  # m
    model['R'] = np.array([[0.5, 0, 0],
                          [0, 1e-4, 0],
                          [0, 0, 1]]) * (se ** 2)

    # kalman parameters
    model['P'] = np.diag([20, 1, 1])

    return model
