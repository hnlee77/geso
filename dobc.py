import numpy as np
from numpy import sin, cos, tanh
from fym.utils.rot import angle2quat, quat2angle


class DOBC:
    def __init__(self, K1, K2):
        self.K1 = K1
        self.K2 = K2

    def get_f(self, g, J, omega):
        Jx, Jy, Jz = np.diag(J)
        return np.vstack((g,
                          omega[1]*omega[2]*(Jy-Jz)/Jx,
                          omega[0]*omega[2]*(Jz-Jx)/Jy,
                          omega[0]*omega[1]*(Jx-Jy)/Jz
                          ))

    def get_g(self, m, J, quat):
        ang = quat2angle(quat)[::-1]
        Jx, Jy, Jz = np.diag(J)
        return np.diag([-cos(ang[0])*cos(ang[1])/m, 1/Jx, 1/Jy, 1/Jz])
        # return np.diag([-1/m, 1/Jx, 1/Jy, 1/Jz])

    def get_control(self, plant, obsv, ref, isdob=False):
        pos = plant.pos.state
        vel = plant.vel.state
        quat = plant.quat.state
        ang = quat2angle(quat)[::-1]
        omega = plant.omega.state
        e = np.vstack((pos[2], np.vstack(ang))) - ref
        edot = np.vstack((vel[2], omega))

        v = - self.K1.dot(e) - self.K2.dot(edot)
        f = self.get_f(plant.g, plant.J, omega)
        g = self.get_g(plant.m, plant.J, quat)
        # NDI controller
        u_N = np.linalg.inv(g).dot(- f + v)
        if isdob is True:
            u_E = -obsv.state[4:8]
            u = u_N + u_E
        else:
            u = u_N
        # observer input
        u_star = u + np.linalg.inv(g).dot(f)
        return u, u_star

    def pid_ctrl(self, x, xdot, ref, p, d):
        e = np.vstack(x - ref)
        edot = np.vstack(xdot - 0)
        cmd = - p * e - d * edot
        return cmd


if __name__ == "__main__":
    pass
