import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    refh_PvUp = ref_mat.hugoniot.fYvX("Up", "P", bounds_error=False)
    ref_rayleigh = ref_mat.hugoniot.get_rayleigh(ref_Us)
    Up_ref, P_ref = get_intersection(ref_rayleigh, refh_PvUp, near_x=ref_Us)

    # Calculate the release from the pressure state in the reference material 
    ref_rels = ref_mat.hugoniot.release(P_ref, model="mg", gamma=.66)

    # Find the Rayleigh line in the sample target material given the measure 
    # shock speed in that material
    tar_rayleigh = tar_mat.hugoniot.get_rayleigh(tar_Us)
    Up_interface, P_interface = get_intersection(tar_rayleigh, ref_rels, near_x=.7*tar_Us)

    # Show a plot of the impedance matching experiment
    if showplot:
        # Check to make sure the intersection of the reference Rayleigh line
        # matches with the point on the known hugoniot for that material 
        refh_PvUs = ref_mat.hugoniot.fYvX("Us", "P", bounds_error=False)
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

    def quartz_UsvUp(up, method="cubic"):

        A0 = 1.754
        A1 = 1.862
        A2 = -0.03364
        A3 = 0.0005666

        if method == "cubic":
            """Us-Up relations for the calculation of Quartz hugoniot
            Both Us-Up relations from Knudson's paper are near identical for
            values of up < 25

            M. Knudson and M. Desjarlais Phys Rev B (88) 2013
                Using equation A1 with constants from table XII

            return A0 + A1*up + A2*up**2 + A3*up**3
            """
            return 6.278 + 1.193*up - 2.505*up*np.exp(-.3701*up)
        elif method == "linear":
            us1 = 6.914 + 1.667*(up[up<6.358]-3.0244)
            us2 = 19.501 + 1.276*(up[up>=6.358]-11.865)
            return np.hstack((us1, us2))

    def quartz_mglr():
        """The MGLR release model of quarts
        """
        pass

    def onpick(event):
        plt.close("IME")
        UsQ, UsCH = barrios.ix[event.ind][["UsQ", "UsCH"]].values[0]
        impedence_matching_exp(q, ch, UsQ, UsCH, showplot=True)

    plt.ion()
    plt.close('all')

    ch = EOS("../data/eos_32.dat")
    q = EOS("../data/eos_24.dat", 
            hugoniot_method="analytic",
            UsvUp=quartz_UsvUp
            )
    #q.hugoniot.release_model = quartz_mglr()


    #import Barrios data
    barrios = pd.read_csv("../data/barrios.dat", sep="\s+", index_col=False)

    # Do impedance matching given Us in Quarts and Us in CH from Barrios
    # calculate rho given the particle velocity
    calc_Up_PCH = np.asarray(
            [impedence_matching_exp(q, ch, UsQ, UsCH) 
                for UsQ, UsCH in barrios[["UsQ", "UsCH"]].values])

    calc_Up, calc_PCH = np.asarray(list(zip(*calc_Up_PCH)))
    UsCH = barrios.UsCH.values
    calc_rho = ch.rho0 * UsCH / (UsCH - calc_Up)

    # Plot the data points for Pressure and Up that Barrios found and the 
    # data point that I found
    ax1 = barrios.plot.scatter("UpCH", "PCH", label="Barrios et al")
    ax1.scatter(calc_Up, calc_PCH/100, color='r', marker="+", 
           label="My analysis", picker=1) 
    # f_PCHvUp = ch.hugoniot.fYvX("Up", "P", bounds_error=False )
    # ax1.scatter(barrios.UpCH.values, f_PCHvUp(barrios.UpCH.values)/100, color='g', marker="+", 
    #         label="CH EOS") 
    ax1.grid()
    ax1.legend()
    ax1.figure.canvas.mpl_connect('pick_event', onpick)

    
    # Plot the data points for Pressure and rho that Barrios found and the 
    # data point that I found
    ax2 = barrios.plot.scatter("rhoCH", "PCH", label="Barrios et al")
    ax2.scatter(calc_rho, calc_PCH/100, c="r", marker="+", label="My analysis")

    # Overlay a plot of the calculated CH hugoniot and CH hugoniot w/ disociation  
    fPCHvRho = ch.hugoniot.fYvX("rho", "P", bounds_error=False)
    rho = np.linspace(2, 4, 1000)
    ax2.plot(rho, fPCHvRho(rho)/100, 'g', label="EOS")

    bnd_eng = 32
    ch._e[ch.pres < 450] += bnd_eng
    ch._calculate_hugoniot()
    fPCHvRho = ch.hugoniot.fYvX("rho", "P")
    ax2.plot(rho, fPCHvRho(rho)/100, 'g--', label="EOS w/ dissociation")

    ax2.set_xlim(2.5, 3.7)
    ax2.set_ylim(0,9)
    ax2.grid()
    ax2.legend()

