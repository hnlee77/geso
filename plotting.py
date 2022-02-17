import numpy as np
import fym
from fym.utils.rot import quat2angle
import matplotlib.pyplot as plt


def plot_info():
    loggerpath = "data/data.h5"
    data, info = fym.load(loggerpath, with_info=True)
    rotor_min = info["rotor_min"]
    rotor_max = info["rotor_max"]

    # plt.subplots_adjust(wspace=0.5, hspace=0.2)

    # Position
    plt.figure()

    ax = plt.subplot(311)
    for i, _label in enumerate(["x", "y", "z"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["x"]["pos"][:, i, 0], "k", label=_label)
        plt.plot(data["t"], data["ref"][:, i, 0], "r--", label=_label+" (cmd)")
        plt.legend(loc="upper right")
        # plt.ylim([-5, 5])
    # plt.axvspan(3, 3.042, alpha=0.2, color="b")
    # plt.axvline(3.042, alpha=0.8, color="b", linewidth=0.5)

    # plt.axvspan(6, 6.011, alpha=0.2, color="b")
    # plt.axvline(6.011, alpha=0.8, color="b", linewidth=0.5)

    # plt.annotate("Rotor 0 fails", xy=(3, 0), xytext=(3.5, 0.5),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))
    # plt.annotate("Rotor 2 fails", xy=(6, 0), xytext=(7.5, 0.2),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Position, m")
    plt.tight_layout()

    # velocity
    plt.figure()
    # plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, _label in enumerate([r"$V_x$", r"$V_y$", r"$V_z$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["x"]["vel"][:, i, 0], "k", label=_label)
        plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Velocity, m/s")
    plt.tight_layout()

    # euler angles
    plt.figure()
    # plt.ylim([-50, 50])

    angles = np.vstack([quat2angle(data["x"]["quat"][j, :, 0]) for j in range(len(data["x"]["quat"][:, 0, 0]))])
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(angles[:, 2-i]), "k", label=_label)
        plt.plot(data["t"], np.rad2deg(data["ang_des"][:, i, 0]), "r--", label=_label+" (cmd)")
        plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Euler angles, deg")
    plt.tight_layout()

    # angular rates
    plt.figure()
    # plt.ylim([-100, 100])

    ax = plt.subplot(311)
    for i, _label in enumerate(["p", "q", "r"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, i, 0]), "k", label=_label)
        plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rates, deg/s")
    plt.tight_layout()

    # Rotor
    plt.figure()

    ax = plt.subplot(321)
    for i in range(data["rotors"].shape[1]):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([rotor_min-5, rotor_max+5])
        plt.plot(data["t"], data["rotors"][:, i], "k-", label="Response")
        plt.plot(data["t"], data["rotors_cmd"][:, i], "r--", label="Command")
        plt.plot(data["t"], rotor_min*np.ones(len(data["t"]), ), "b-.", label=r"$rotor_{min}$")
        plt.plot(data["t"], rotor_max*np.ones(len(data["t"]), ), "b-.", label=r"$rotor_{max}$")
        if i == 1:
            plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor thrust")
    plt.tight_layout()

    # Generalized forces
    plt.figure()

    ax = plt.subplot(221)
    for i, _label in enumerate([r"$F$", r"$M_{\phi}$", r"$M_{\theta}$", r"$M_{\psi}$"]):
        if i != 0:
            plt.subplot(221+i, sharex=ax)
        plt.plot(data["t"], data["control"][:, i], "k-", label=_label)
        plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Generalized forces")
    plt.tight_layout()

    # Disturbances
    plt.figure()
    # plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i in range(data["d"].shape[1]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["dhat"][:, i, 0], "k", label="Estimated")
        plt.plot(data["t"], data["d"][:, i, 0], "r--", label="Actual")
        if i == 0:
            plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Disturbances by faults")
    plt.tight_layout()

    # FDI
    plt.figure()

    ax = plt.subplot(321)
    for i in range(data["W"].shape[1]):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([0-0.1, 1+0.1])
        plt.plot(data["t"], data["What"][:, i, i], "k-", label="Estimated")
        plt.plot(data["t"], data["W"][:, i, i], "r--", label="Actual")
        if i == 1:
            plt.legend(loc="lower right")
    plt.gcf().supylabel("FDI")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_info()
