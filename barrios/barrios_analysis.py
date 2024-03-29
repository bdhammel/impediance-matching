import types
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
import os, sys

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir_name not in sys.path:
    sys.path.append(parent_dir_name)

from util import MonteCarloVariable
from eos import EOS, quartz_UsvUp, quartz_mglr, quartz_ode_mglr
from impediance_match import impedence_matching_exp, monte_carlo_error_prop

rc('font', family='sans-serif', size=13)


def onpick(event):
    """Enable interactive picking on the plotted data

    Displays the impedance matching plot for a given data point

    Args
    ----
    event (matplotlib event) : 

    Example
    -------
    fig = plt.figure()
    plt.plot([1,4,9,16])
    fig.canvas.mpl_connect('pick_event', onpick)
    """
    plt.close("IME")
    UsQ, UsCH = barrios.ix[event.ind][["UsQ", "UsCH"]].values[0]
    impedence_matching_exp(q, ch, UsQ, UsCH, showplot=True)


if __name__ == "__main__":

    plt.ion()
    plt.close('all')

    # Import Polystyrene EOS data, from a HYADES EOS file, and change initial 
    # conditions
    ch = EOS("../data/eos_32.dat")
    ch._rho0 = 1.05
    ch._P0 = 0

    # Import Quartz EOS data and use an analytic method bases on Us vs Up
    # to calculate the Hugoniot
    q = EOS("../data/eos_24.dat", 
            hugoniot_method="analytic",
            upmin=0,
            upmax=29,
            UsvUp=quartz_UsvUp(model="knudson")
            )

    # Play around with different quarts release models
    q.hugoniot.custom_release_model = types.MethodType(
            quartz_ode_mglr, 
            q.hugoniot) 
    q.hugoniot.set_release_model("custom")
    #q.hugoniot.set_release_model("mg", gamma=.66)

    #import Barrios' data
    barrios = pd.read_csv("../data/barrios.dat", sep="\s+", index_col=False)
    barrios.PCH = barrios.PCH.apply(lambda x: x*100) # convert to GPa

    # Do impedance matching given Us in Quarts and Us in CH from Barrios
    # calculate rho given the particle velocity
    calc_Up_PCH = np.asarray(
            [impedence_matching_exp(q, ch, UsQ, UsCH) 
                for UsQ, UsCH in barrios[["UsQ", "UsCH"]].values])

    # extract the calculated Up, P, and rho data
    calc_Up, calc_PCH = calc_Up_PCH.T
    UsCH = barrios.UsCH.values
    calc_rho = ch.rho0 * UsCH / (UsCH - calc_Up)

    # Plot the data points for Pressure and Up that Barrios found and the 
    # data point that I found
    ax1 = barrios.plot.scatter("UpCH", "PCH", label="Barrios et al")
    ax1.scatter(calc_Up, calc_PCH, color='r', marker="+", 
           label="My analysis", picker=1) 

    # f_PCHvUp = ch.hugoniot.fYvX("Up", "P", bounds_error=False )
    # ax1.scatter(
    #    barrios.UpCH.values, 
    #    f_PCHvUp(barrios.UpCH.values)/100, 
    #    color='g', 
    #    marker="+", 
    #   label="CH EOS") 

    ax1.grid()
    ax1.legend()
    ax1.figure.canvas.mpl_connect('pick_event', onpick)

    """Do monte carlo error analysis 
    """
    # Plot the data points for Pressure and rho that Barrios found and the 
    # data point that I found
    plt.figure()
    ax2 = plt.subplot(111)
    #ax2 = barrios.plot.scatter("rhoCH", "PCH", 
    #        xerr="ran.2", yerr="ran.1", label="Barrios et al")
    #ax2.scatter(calc_rho, calc_PCH/100, c="r", marker="o", label="My analysis")

    # Run the monte carlo analysis by varying the input data within the error bars
    # reported by Barrios
    analysis = []
    for i, exp in barrios.iterrows():
        Us_ref = MonteCarloVariable(exp["UsQ"], exp["+/-"])
        Us_tar = MonteCarloVariable(exp["UsCH"], exp["+/-.1"])

        analysis.append(
                monte_carlo_error_prop(q, ch, Us_ref, Us_tar)
                )

    # Extract the data from the analysis and plot
    up, uperr, p, perr, rho, rhoerr = np.transpose(analysis)
    ax2.errorbar(rho, p, xerr=rhoerr, yerr=perr, ms=4, c='r', fmt='s', label="My analysis")

    # Overlay a plot of the calculated CH hugoniot and CH hugoniot w/ disociation  
    # Hugoniot is calculated based on the contour method of the CH EOS
    rho, P = ch.hugoniot.get_vars("rho", "P")
    ax2.plot(rho, P, 'g', label="EOS")

    # add in a bond dissociation energy 
    bnd_eng = 32 # kJ/g
    ch._e[ch.pres > 450] -= bnd_eng
    ch._calculate_hugoniot()
    rho, P = ch.hugoniot.get_vars("rho", "P")
    ax2.plot(rho, P, 'g--', label="EOS w/ dissociation")

    fig = plt.gcf()
    fig.set_size_inches((5,5))
    plt.xlabel("Density [g cm${}^{-3}$]")
    plt.ylabel("Pressure [GPa]")

    ax2.set_xlim(2.3, 3.65)
    ax2.set_ylim(0,900)
    ax2.grid()
    ax2.legend(loc=2)
    plt.tight_layout()



