import numpy as np
from numpy import sin, cos, tanh
from math import pi, isclose
from tqdm import trange
import torch

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
from dobc import Obsv_agent, hat

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
        super().__init__(dt=0.01, max_t=20, solver="rk4")
        init_pos = np.vstack((0, 0, 0))
        # init_ang = np.deg2rad([0, 0, 0])*(np.random.rand(3) - 0.5)
        init_ang = np.deg2rad([20, 30, 0])*(np.random.rand(3) - 0.5)
        init_quat = (angle2quat(init_ang[2], init_ang[1], init_ang[0]))
        self.plant = Multicopter(
            pos=init_pos,
            vel=np.zeros((3, 1)),
            quat=init_quat,
            omega=np.zeros((3, 1)),
        )
        self.trim_forces = np.vstack([self.plant.m * self.plant.g, 0, 0, 0])
        self.rotor_n = self.plant.mixer.B.shape[1]

        n = 3
        self.obsv = BaseSystem(np.zeros((4*(n + 1), 1)))
        self.B = np.zeros((4*(n+1), 4))
        self.C = np.zeros((4*(n+1), 4)).T
        self.B[0:4, 0:4] = np.eye(4)
        self.C[0:4, 0:4] = np.eye(4)
        self.A = np.eye(4*(n+1), 4*(n+1), 4)
        wb = 20
        clist = np.array([4, 6, 4, 1])
        llist = clist * wb ** np.array([1, 2, 3, 4])
        L = []
        for lval in llist:
            L.append(lval*np.eye(4))
        self.L = np.vstack(L)
        self.obsv_agent = Obsv_agent()

        # # Define agents
        # self.CCA = ConstrainedCA(self.plant.mixer.B)

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            # LoE(time=3, index=0, level=0.5),
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
        ref = self.get_ref(t)
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
        fault = W - np.eye(self.rotor_n)
        pos = self.plant.pos.state
        # vel = self.plant.vel.state
        quat = self.plant.quat.state
        ang = quat2angle(quat)

        x_hat = self.obsv.state
        y_hat = self.C.dot(x_hat)
        dhat = x_hat[4:8]

        # # PID position control
        # phi_des = - 0.02*(pos[1] - ref[1]) - 0.1*vel[1]
        # theta_des = - (- 0.02*(pos[0] - ref[0]) - 0.1*vel[0])
        # ang_des = np.vstack((phi_des, theta_des, 0))

        ang_des = np.vstack((0, 0, 0))
        des = np.vstack((ref[2], ang_des))
        u, u_star = self.obsv_agent.get_control(self.plant, self.obsv, des)
        forces = u
        rotors_cmd = self.control_allocation(t, forces, What)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        d = self.plant.get_tau_d(fault, rotors)
        # fd = self.plant.mixer.B.dot(fault.dot(rotors))[0]
        # print(fd)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)

        self.plant.set_dot(t, rotors)
        y = np.vstack((pos[2], np.vstack(ang)))
        self.obsv.dot = (
            self.A.dot(x_hat)
            + self.B.dot(u_star) + self.L.dot(y - y_hat)
        )

        return dict(t=t, x=self.plant.observe_dict(),
                    control=u, rotors=rotors, rotors_cmd=rotors_cmd,
                    dhat=dhat, d=d, W=W, What=What, ref=ref, ang_des=ang_des)

    def get_ref(self, t):
        ref = np.vstack((0, 0, -2))
        return ref


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
