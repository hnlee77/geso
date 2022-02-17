import numpy as np
from numpy import sin, cos, tanh
from fym.utils.rot import angle2quat, quat2angle


class Obsv_agent:
    def __init__(self):
        self.K1 = np.diag([5, 1, 1, 10])
        self.K2 = np.diag([5, 1, 1, 10])

    def get_f(self, plant):
        omega = plant.omega.state
        Jx, Jy, Jz = plant.J[0, 0], plant.J[1, 1], plant.J[2, 2]
        return np.vstack((plant.g,
                          omega[1]*omega[2]*(Jy-Jz)/Jx,
                          omega[0]*omega[2]*(Jz-Jx)/Jy,
                          omega[0]*omega[1]*(Jx-Jy)/Jz
                          ))

    def get_g(self, plant):
        quat = plant.quat.state
        ang = quat2angle(quat)
        Jx, Jy, Jz = plant.J[0, 0], plant.J[1, 1], plant.J[2, 2]
        return np.diag([-cos(ang[2])*cos(ang[1])/plant.m, 1/Jx, 1/Jy, 1/Jz])

    def get_control(self, plant, obsv, ref):
        pos = plant.pos.state
        vel = plant.vel.state
        quat = plant.quat.state
        ang = quat2angle(quat)
        omega = plant.omega.state
        e = np.vstack((pos[2], np.vstack(ang))) - ref
        edot = np.vstack((vel[2], omega))
        v = - self.K1.dot(e) - self.K2.dot(edot)
        f = self.get_f(plant)
        g = self.get_g(plant)
        u_N = np.linalg.inv(g).dot(- f + v)
        # u_E = -obsv.state[4:8]
        u_star = np.linalg.inv(g).dot(v)
        return u_N, u_star


def hat(v):
    v1, v2, v3 = v.squeeze()
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])


if __name__ == "__main__":
    pass
