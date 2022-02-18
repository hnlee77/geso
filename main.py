import numpy as np
import matplotlib.pyplot as plt

import fym
from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import dcm2quat, quat2dcm, angle2quat, quat2angle
import ftc.config
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager
from ftc.agents.CA import ConstrainedCA

from multicopter import Multicopter
import plotting
from dobc import DOBC

# plt.rc("font", **{
#     "family": "sans-serif",
#     "sans-serif": ["Helvetica"],
# })
plt.rc("text", usetex=False)
plt.rc("lines", linewidth=1)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=0.8)

cfg = ftc.config.load()


class MyEnv(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=20)
        init_pos = np.vstack((0, 0, 0))
        init_ang = np.deg2rad([20, 30, 10])*(np.random.rand(3) - 0.5)
        init_quat = (angle2quat(init_ang[2], init_ang[1], init_ang[0]))
        self.plant = Multicopter(
            pos=init_pos,
            vel=np.zeros((3, 1)),
            quat=init_quat,
            omega=np.zeros((3, 1)),
        )
        self.rotor_n = self.plant.mixer.B.shape[1]

        n = 3  # (n-1)-order derivative of disturbance
        l = 4  # output dimension
        self.obsv = BaseSystem(np.zeros((l*(n + 1), 1)))
        self.B = np.zeros((l*(n+1), l))
        self.C = np.zeros((l*(n+1), l)).T
        self.B[0:l, 0:l] = np.eye(l)
        self.C[0:l, 0:l] = np.eye(l)
        self.A = np.eye(l*(n+1), l*(n+1), l)
        wb = 20
        clist = np.array([4, 6, 4, 1])
        llist = clist * wb ** np.array([1, 2, 3, 4])
        L = []
        for lval in llist:
            L.append(lval*np.eye(l))
        self.L = np.vstack(L)

        # Define agents
        self.K1 = np.diag([5, 10, 10, 10])
        self.K2 = 3*np.diag([5, 10, 10, 10])
        self.dobc = DOBC(self.K1, self.K2)
        # self.CCA = ConstrainedCA(self.plant.mixer.B)

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            LoE(time=3, index=0, level=0.5),
            # LoE(time=6, index=2, level=0.1),
        ], no_act=self.rotor_n)

        # Define FDI
        self.fdi = self.fault_manager.fdi
        self.detection_time = self.fault_manager.fault_times + self.fdi.delay

    def step(self):
        *_, done = self.update()
        return done

    def control_allocation(self, t, forces, What):
        rotors = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        return rotors

    def set_dot(self, t):
        pos_des, ang_des = self.get_ref(t)
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
        pos = self.plant.pos.state
        vel = self.plant.vel.state
        omega = self.plant.omega.state

        # PID position control
        ispospid = False
        if ispospid is True:
            phi_des = self.dobc.pid_ctrl(pos[1], vel[1], pos_des[1], 0, 0)
            theta_des = - self.dobc.pid_ctrl(pos[0], vel[0], pos_des[0], 0, 0)
            ang_des = np.vstack((phi_des, theta_des, 0))

        inner_des = np.vstack((pos_des[2], ang_des))  # inner controller output
        forces, u_star = self.dobc.get_control(self.plant, self.obsv,
                                               inner_des, isdob=True)
        rotors_cmd = self.control_allocation(t, forces, What)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # disturbances by faults
        d = self.plant.get_d(W, rotors)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)

        self.plant.set_dot(t, rotors)

        # Extended state observer
        x_hat = self.obsv.state
        dhat = x_hat[4:8]
        y_hat = self.C.dot(x_hat)
        y = np.vstack((-self.plant.m*vel[2], self.plant.J.dot(omega)))
        self.obsv.dot = (
            self.A.dot(x_hat)
            + self.B.dot(u_star) + self.L.dot(y - y_hat)
        )

        return dict(t=t, x=self.plant.observe_dict(),
                    control=forces, rotors=rotors, rotors_cmd=rotors_cmd,
                    d=d, dhat=dhat, W=W, What=What, ref=pos_des, ang_des=ang_des)

    def get_ref(self, t):
        pos_des = np.vstack((0, 0, 0))
        ang_des = np.vstack((0, 0, 0))
        return pos_des, ang_des


def exp_run(loggerpath):
    env = MyEnv()
    env.logger = fym.Logger(loggerpath)
    # env.logger.set_info(cfg=ftc.config.load())

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            env_info = {
                "detection_time": env.detection_time,
                "rotor_min": env.plant.rotor_min,
                "rotor_max": env.plant.rotor_max,
            }
            env.logger.set_info(**env_info)
            break

    env.close()


if __name__ == "__main__":
    np.random.seed(1)
    loggerpath = "data/data.h5"
    exp_run(loggerpath)

    plotting.plot_info()
