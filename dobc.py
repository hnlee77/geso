import numpy as np
from numpy import sin, cos, tanh
from fym.utils.rot import dcm2quat, quat2dcm, angle2quat, quat2angle


class Obsv_agent:
    def __init__(self):
        self.K1 = 0.1*np.eye(3)
        self.K2 = 0.5*np.eye(3)

    def get_f(self, plant):
        omega = plant.omega.state
        Jx, Jy, Jz = plant.J
        breakpoint()
        return np.vstack((-9.81,
                          omega[1]*omega[2]*(Jy-Jz)/Jx,
                          omega[0]*omega[2]*(Jz-Jx)/Jy,
                          omega[0]*omega[1]*(Jx-Jy)/Jz
                          ))

    def get_g(self, plant):
        quat = plant.quat.state
        ang = quat2angle(quat)
        breakpoint()
        Jx, Jy, Jz = plant.J
        m = plant.m
        return np.array([-cos(ang[2])*cos(ang[1])/m, 0, 0, 0],
                        [0, 1/Jx, 0, 0],
                        [0, 0, 1/Jy, 0]
                        [0, 0, 0, 1/Jz])

    def get_control(self, plant, obsv, ref):
        q_d = ref
        quat = plant.quat.state
        omega = plant.omega.state
        q_0e = quat_error(quat, q_d)[0]
        q_ve = quat_error(quat, q_d)[1:]
        w_e = omega
        dq_ve = (
            0.5 * np.vstack((- q_ve.T, hat(q_ve) + q_0e*np.eye(3))).dot(w_e)[1:]
        )
        N = (
            - hat(omega).dot(plant.J).dot(omega)
            + plant.J.dot(self.K1).dot(dq_ve) + q_ve)
        u_N = - N - self.K2.dot(w_e + self.K1.dot(q_ve))
        u_E = - obsv.state[3:6]
        torque = u_N + u_E
        return np.vstack((0, torque))


def quat_error(q, qd):
    q0, qv = q[0], q[1:]
    qd0, qdv = qd[0], qd[1:]
    qe0 = q0 * qd0 + qv.T.dot(qdv)
    qev = qd0 * qv - q0 * qdv + hat(qv).dot(qdv)
    return np.vstack((qe0, qev))


def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])


if __name__ == "__main__":
    pass
