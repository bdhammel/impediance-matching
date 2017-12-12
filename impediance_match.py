import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, cumtrapz
from scipy.interpolate import interp1d

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
    up = np.linspace(0,50,100)

    # Find the pressure state in the reference material, given the measured shock
    # speed
    refh_PvUp = ref_mat.hugoniot.fYvX("Up", "P", bounds_error=False)
    ref_rayleigh = ref_mat.hugoniot.get_rayleigh(ref_Us)
    Up_ref, P_ref = get_intersection(ref_rayleigh, refh_PvUp, near_x=.75*ref_Us)

    # Calculate the release from the pressure state in the reference material 
    ref_rels = ref_mat.hugoniot.release(P_ref)
      
    # ref_rels = ref_mat.hugoniot.release(P_ref, model="mg", 
    #         method="hammel-integral", gamma=.66) #var_gamma(ref_Us))

    # Find the Rayleigh line in the sample target material given the measure 
    # shock speed in that material
    tar_rayleigh = tar_mat.hugoniot.get_rayleigh(tar_Us)
    Up_interface, P_interface = get_intersection(
            tar_rayleigh, ref_rels, near_x=.7*tar_Us)

    # Show a plot of the impedance matching experiment
    if showplot:
        # Check to make sure the intersection of the reference Rayleigh line
        # matches with the point on the known hugoniot for that material 
        refh_PvUs = ref_mat.hugoniot.fYvX("Us", "P")
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

    return Up_interface, P_interface


def monte_carlo_error_prop(ref_mat, tar_mat, _ref_Us, _tar_Us):
    """Preform a monte carlo run for a given experiments

    Args
    ____
    ref_mat (EOS)
    tar_mat (EOS)
    ref_Us (MoneteCarloVariable)
    tar_Us (MoneteCarloVariable)

    Returns
    -------
    mean of particle speed
    standard deviation of particle speed
    mean of pressure
    standard deviation of pressure
    mean of density
    standard deviation of density
    """

    # Particle speed points, used in retrieving info from interpolated functions
    up = np.linspace(0, 50, 100)

    # Find the pressure state in the reference material, given the measured shock
    # speed
    refh_PvUp = ref_mat.hugoniot.fYvX("Up", "P", bounds_error=False)

    epochs = range(100)

    Up_collection = []
    P_collection = []
    rho_collection = []

    for _ in epochs:
        # re-randomize inputs
        ref_Us = _ref_Us.generate()
        tar_Us = _tar_Us.generate()

        ref_rayleigh = ref_mat.hugoniot.get_rayleigh(ref_Us)
        Up_ref, P_ref = get_intersection(ref_rayleigh, refh_PvUp, near_x=.75*ref_Us)

        # Calculate the release from the pressure state in the reference material 
        ref_rels = ref_mat.hugoniot.release(P_ref)
          
        # Find the Rayleigh line in the sample target material given the measure 
        # shock speed in that material
        tar_rayleigh = tar_mat.hugoniot.get_rayleigh(tar_Us)
        Up_interface, P_interface = get_intersection(
                tar_rayleigh, ref_rels, near_x=.7*tar_Us)

        rho = tar_mat.rho0 * tar_Us / (tar_Us - Up_interface)

        Up_collection.append(Up_interface)
        P_collection.append(P_interface)
        rho_collection.append(rho)

    return (np.mean(Up_collection), np.std(Up_collection), 
            np.mean(P_collection), np.std(P_collection),
            np.mean(rho_collection), np.std(rho_collection))


