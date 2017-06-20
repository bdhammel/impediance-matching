import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from eos import EOS
from util import get_intersection, MonteCarloVariable

def impedence_matching_exp(ref_mat, tar_mat, ref_Us, tar_Us, showplot=False):
    """Simulate an impedance matching experiment

    Args
    ---
    ref_mat (EOS)
    tar_mat (EOS)
    ref_Us (flaot) : measured Shock velocity in the reference material 
    tar_Us (flaot) : measured Shock velocity in the target material 
    showplot (Bool) : output a plot of the impedance matching experiment

    Returns
    -------
    Up_interface (float) : the particle velocity at the reference-sample interface
    P_interface (float) : the pressure at the reference-sample interface
    """
    # Particle speed points, used in retrieving info from interpolated functions
    up = np.linspace(0,100,1000)


    # Find the pressure state in the reference material, given the measured shock
    # speed
    refh_PvUp = interp1d(*ref_mat.hugoniot.get_vars("Up", "P"), bounds_error=False)
    ref_rayleigh = ref_mat.hugoniot.get_rayleigh(ref_Us)
    Up_ref, P_ref = get_intersection(ref_rayleigh, refh_PvUp, near_x=ref_Us)

    # Calculate the release from the pressure state in the reference material 
    ref_rels = ref_mat.hugoniot.release_isentrope(P_ref)

    # Find the Rayleigh line in the sample target material given the measure 
    # shock speed in that material
    tar_rayleigh = tar_mat.hugoniot.get_rayleigh(tar_Us)
    Up_interface, P_interface = get_intersection(tar_rayleigh, ref_rels, near_x=.7*tar_Us)

    # Show a plot of the impedance matching experiment
    if showplot:
        # Check to make sure the intersection of the reference Rayleigh line
        # matches with the point on the known hugoniot for that material 
        refh_PvUs = interp1d(*ref_mat.hugoniot.get_vars("Us", "P"))
        P_test = refh_PvUs(ref_Us)

        plt.figure("IME")
        plt.plot(up, refh_PvUp(up), label="Reference")
        plt.axhline(P_test, c='g', ls='--')
        plt.plot(up, ref_rels(up), label="Release")
        plt.plot(up, ref_rayleigh(up), label="Ref Rayleigh line")
        plt.plot(up, tar_rayleigh(up), label="Target Rayleigh line")
        plt.scatter(Up_ref, P_ref, color='g', marker='o')
        plt.scatter(Up_interface, P_interface, color='r', marker='o')
        plt.xlabel("Up [km s-1]")
        plt.ylabel("Pressure [GPa]")
        plt.xlim(0, 1.5*Up_interface)
        plt.ylim(0, 1.2*P_ref)
        plt.legend()
        plt.grid()

    return Up_interface[0], P_interface[0] 

def monte_carlo_error_analysis(ref_mat, tar_mat, ref_Us, tar_Us):
    pass


if __name__ == "__main__":

    def quartz_gamma(Us):
        """Taken from Marcus' email to Jim (Jun 15 2017)
        """
        a1 = 0.579
        a2 = 0.129
        a3 = 12.81

        if Us > 14.69:
            return a1*(1 - np.exp(-a2*(Us - a3)**1.5))
        else:
            return 0.11016009*Us - 1.4544581;

    def quartz_UsvUp(up, method="knudson"):

        A0 = 1.754
        A1 = 1.862
        A2 = -0.03364
        A3 = 0.0005666

        if method == "knudson":
            """Us-Up relations for the calculation of Quartz hugoniot
            Both Us-Up relations from Knudson's paper are near identical for
            values of up < 25
                return A0 + A1*up + A2*up**2 + A3*up**3

            M. Knudson and M. Desjarlais Phys Rev B (88) 2013
                Using equation A1 with constants from table XII
            """
            return A0 + A1*up + A2*up**2 + A3*up**3
            return 6.278 + 1.193*up - 2.505*up*np.exp(-.3701*up)
        elif method == "linear":
            us1 = 6.914 + 1.667*(up[up<6.358]-3.0244)
            us2 = 19.501 + 1.276*(up[up>=6.358]-11.865)
            return np.hstack((us1, us2))

    def onpick(event):
        plt.close("IME")
        UsQ, UsCH = barrios.ix[event.ind][["UsQ", "UsCH"]].values[0]
        impedence_matching_exp(q, ch, UsQ, UsCH, showplot=True)

    """
    plt.ion()
    plt.close('all')

    q = EOS("../data/eos_24.dat", 
            hugoniot_method="analytic",
            gamma=quartz_gamma,
            UsvUp=quartz_UsvUp
            )

    bnd_eng = 32
    ch = EOS("../data/eos_32.dat")

    barrios = pd.read_csv("../data/barrios.dat", sep="\s+", index_col=False)

    f_PCHvUp = interp1d(*ch.hugoniot.get_vars("Up", "P"))

    calc_Up_PCH = np.asarray(
            [impedence_matching_exp(q, ch, UsQ, UsCH) 
                for UsQ, UsCH in barrios[["UsQ", "UsCH"]].values])

    calc_Up, calc_PCH = np.asarray(list(zip(*calc_Up_PCH)))
    UsCH = barrios.UsCH.values
    calc_rho = ch.rho0 * UsCH/ (UsCH - calc_Up)

    ax1 = barrios.plot.scatter("UpCH", "PCH", label="Barrios et al")
    ax1.scatter(barrios.UpCH.values, f_PCHvUp(barrios.UpCH.values)/100, color='g', marker="+", 
            label="CH EOS") 
    ax1.scatter(calc_Up, calc_PCH/100, color='r', marker="+", 
           label="My analysis", picker=1) 
    ax1.grid()
    ax1.legend()
    ax1.figure.canvas.mpl_connect('pick_event', onpick)

    ax2 = barrios.plot.scatter("rhoCH", "PCH", label="Barrios et al")
    ax2.scatter(calc_rho, calc_PCH/100, c="r", marker="+", label="My analysis")

    Up, Us, rho, P = ch.hugoniot.get_vars("Up", "Us", "rho", "P")

    ax2.plot(rho, P/100, 'g', label="EOS")

    ch._e[ch.pres < 450] += bnd_eng
    ch._calculate_hugoniot()

    rho, P = ch.hugoniot.get_vars("rho", "P")
    ax2.plot(rho, P/100, 'g--', label="EOS w/ dissociation")

    ax2.set_xlim(2.5, 3.7)
    ax2.set_ylim(0,9)
    ax2.grid()
    ax2.legend()

    #fig = plt.figure(111)
    #plt.plot(calc_Up, barrios.UsCH.values)
    #plt.plot(Up, Us)
    #plt.grid()
    #plt.xlim(calc_Up.min(), calc_Up.max())
    #plt.ylim(7.5,40)
    #plt.xlabel("Up")
    #plt.ylabel("Us")

    """
