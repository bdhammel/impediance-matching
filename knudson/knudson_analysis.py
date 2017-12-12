"""This is a test script to check my method of impedance matching with quartz 
again the published work 

"""

import types
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eos import EOS, quartz_UsvUp, quartz_mglr, quartz_ode_mglr
from impediance_match import impedence_matching_exp, monte_carlo_error_prop
from util import MonteCarloVariable


def onpick(event):
    plt.close("IME")
    UsQ, UsGDP = knudson.ix[event.ind][["UsQ", "UsGDP"]].values[0]
    impedence_matching_exp(q, GDP, UsQ, UsGDP, showplot=True)

if __name__ == "__main__":

    plt.ion()
    plt.close('all')

    GDP = EOS("../data/eos_32.dat")
    GDP._rho0 = 1.035
    GDP._P0 = 0
    q = EOS("../data/eos_24.dat", 
            hugoniot_method="analytic",
            upmin=0,
            upmax=50,
            UsvUp=quartz_UsvUp(model="knudson")
            )

    # overload the quarts hugoniot with the MGLR model 
    q.hugoniot.custom_release_model = types.MethodType(quartz_ode_mglr, q.hugoniot) 
    q.hugoniot.set_release_model("custom")

    knudson = pd.read_csv("../data/knudson_test.csv", index_col=False)
    UsGDP = knudson.UsGDP.values

    ax1 = plt.subplot(111)
    ax1.plot(knudson.rho_rho0.values, knudson.PGDP.values, 'o')

    """
    calc_Up_PGDP = np.asarray(
            [impedence_matching_exp(q, GDP, UsQ, UsGDP) 
                for UsQ, UsGDP in knudson[["UsQ", "UsGDP"]].values])
    calc_Up, calc_PGDP = calc_Up_PGDP.T
    calc_rho = GDP.rho0 * UsGDP / (UsGDP - calc_Up)

    ax1.plot(calc_rho/GDP.rho0, calc_PGDP, 'o', picker=1)
    """

    analysis = []
    for i, exp in knudson.iterrows():
        Us_ref = MonteCarloVariable(exp["UsQ"], exp["UsQer"])
        Us_tar = MonteCarloVariable(exp["UsGDP"], exp["UsGDPer"])

        analysis.append(
                monte_carlo_error_prop(q, GDP, Us_ref, Us_tar)
                )

    up, uperr, p, perr, rho, rhoerr = np.transpose(analysis)
    ax1.errorbar(rho/GDP.rho0, p, xerr=rhoerr/GDP.rho0, yerr=perr, 
           picker=2, fmt='o')
    #ax1.figure.canvas.mpl_connect('pick_event', onpick)

