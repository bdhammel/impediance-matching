import re
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from  scipy.interpolate import interp1d, interp2d
from  scipy.optimize import fmin, minimize, fminbound

"""

den : g/cc
temp : keV
pres : dyne/cm2
eng : erg

df = pd.read_fwf("../data/eos_44.dat", widths=[15]*5, skiprows=2, header=None)
"""

class EOS:
    def __init__(self, filename):
        self._read_eos_table(filename)
        self._calculate_hugoniot()

    def _read_eos_table(self, filename):
        """Read in an EOS table from the HYADES database
        """
        with open(filename, "r") as f:
            self.name, *_ = f.readline().split()
            self.eosnum, self.Z, self._rho0 = np.asarray(
                    f.readline().split(), dtype=float)[[0,1,3]]
            self.eosnum = int(self.eosnum)
            eos_iter = re.finditer('.{15}', f.read())

        NR = int(float(next(eos_iter).group()))
        NT = int(float(next(eos_iter).group()))

        import_data = lambda l:  np.asarray(
                [next(eos_iter).group() for _ in range(l)], 
                dtype=np.float32)

        self._d = import_data(NR)[1:]
        self._t = import_data(NT)[1:]
        self._p = np.reshape(import_data(NT*NR), (NT, NR))[1:,1:] 
        self._e = np.reshape(import_data(NT*NR), (NT, NR))[1:,1:] 

        self._dden, self._ttemp = np.meshgrid(self._d, self._t)
        self._interp_p = interp2d(self._d, self._t, self._p)
        self._interp_e = interp2d(self._d, self._t, self._e)


    @property
    def rho0(self) -> "g/cc":
        return self._rho0
    @property
    def T0(self) -> "KeV":
        return self._T0

    @property
    def P0(self) -> "GPa":
        return self._P0 * 1e-10

    @property
    def E0(self) -> "Erg":
        return self_rho0

    @property
    def den(self) -> "g/cc":
        return self._dden

    @property
    def temp(self) -> "KeV":
        return self._ttemp

    @property
    def pres(self) -> "GPa":
        return self._p * 1e-10

    @property
    def eng(self) -> "erg":
        return self._e

    def _plot2D(self, Z, plotname):
        """Base plotting script for 2D plot
        """
        plt.figure(plotname)
        plot = plt.pcolormesh(self.den, self.temp, Z, norm=LogNorm())
        plt.colorbar(plot)
        plt.ylim(1e-5, 100)
        plt.yscale('log')
        plt.xlabel("Density")
        plt.ylabel("Temperature")
        plt.draw()

    def plot_energy(self):
        self._plot2D(self.eng, "Energy")

    def plot_pressure(self):
        self._plot2D(self.pres, "Pressure")

    def isentrope(self, x, y):
        rho, P, E = self.hugoniot("rho", "P", "E")

        return F(V) - gama/V * (E - self._E0)

    def _calculate_hugoniot(self):
        """Calculate the materials hugoniot using the contour method
        """

        try: 
            hvars
        except:
            self.hvars = {}
        else:
            print("Hugoniot already calculated")
            return

        self._T0 = 2.5e-5
        self._P0 = self._P_lookup(rho=self._rho0, T=self._T0)
        self._E0 = self._E_lookup(rho=self._rho0, T=self._T0)

        zero = self._e - self._E0 - .5*(self._p + self._P0) * (1/self._rho0 - 1/self._dden)

        fig = plt.figure('tmp')
        cs = plt.contour(self._d, self._t, zero, [0])
        plt.close(fig)
        p = cs.collections[0].get_paths()[0]
        v = p.vertices

        self.hvars["rho"] = v[:,0]
        self.hvars["temp"] = v[:,1]
        self.hvars["P"] = np.asarray([
            self._P_lookup(rho=rho, T=T)[0] for rho, T in v])
        self.hvars["eng"] = np.asarray([
            self._E_lookup(rho=rho, T=T)[0] for rho, T in v])
        self.hvars["Up"] = np.sqrt(
                (self.hvars["rho"]-self._rho0)/(self.hvars["rho"] * self._rho0)  
                * (self.hvars["P"] - self._P0)
                )
        self.hvars["Us"] = (self.hvars["P"] - self._P0)/(self._rho0*self.hvars["Up"])

        # convert to better units
        self.hvars["Us"] *= 1e-5 # km/s
        self.hvars["Up"] *= 1e-5 # km/s
        self.hvars["P"] *= 1e-10 # GPa

    def hugoniot(self, *args):
        """Return calculated points on the hugoniot
        """
        return (self.hvars[key] for key in args)

    def get_rayleigh(self, Us):
        """Get the Rayleigh line through P-Up space given a shock speed

        Args
        ---
        Us : shock speed km/s
        """
        return lambda u : self.rho0 * Us * u + self.P0
        
    def _E_lookup(self, rho:"g/cc", T:"keV")->"erg":
        """Find an energy value given a density and temperature

        Needed in the calculation of the hugoniot
        """
        return self._interp_e(rho, T)

    def _P_lookup(self, rho:"g/cc", T:"keV")->"dynes/cm2":
        """Find a Pressure value given a density and temperature

        Needed in the calculation of the hugoniot
        """
        return self._interp_p(rho, T)

    def check_hugoniot(self):
        """Check the calculated hugoniot against the experimental points from 
        http://www.ihed.ras.ru/rusbank/
        """
        hugoniot_pts = pd.read_table(
                "../data/hugoniot_{}.dat".format(self.eosnum), 
                sep="\s+", skiprows=3, index_col=False)
        hugoniot_pts = hugoniot_pts[hugoniot_pts.m == 1]

        Uph, Ph = self.hugoniot("Up", "P")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(Uph, Ph, 'g-o')
        hugoniot_pts.plot.scatter("U", "P", ax=ax)
        plt.ylim(-10, 800)
        plt.xlim(-.5, 15)
        plt.grid()

def mirrored_hugoniot(mirror_about_up, f_P):
    """Return a mirrored hugoniot in the P-Up plane about a particle velocity

    Args
    ----
    mirror_about_up : the particle velocity to mirror the hugoniot about
    """
    return lambda up: f_P(2*mirror_about_up-up)

def get_intersection(f, g, near):
    """Get the intersection of two interpolated functions

    Args
    ----
    f : function 1
    g : function 2
    near : a starting point for the solver
    """
    h = lambda x: (f(x) - g(x))**2
    x = fmin(h, near, maxfun=1000, disp=False)
    return x, f(x)


def impedence_matching_exp(ref_mat, tar_mat, ref_Us, tar_Us, showplot=False):
    """Simulate an impedance matching experiment
    """
    up = np.linspace(0,100,1000)

    refh_PvUp = interp1d(*ref_mat.hugoniot("Up", "P"), bounds_error=False)
    ref_rayleigh = ref_mat.get_rayleigh(ref_Us)
    Up_ref, P_ref = get_intersection(ref_rayleigh, refh_PvUp, near=ref_Us)

    ref_rels = mirrored_hugoniot(Up_ref, refh_PvUp)
    tar_rayleigh = tar_mat.get_rayleigh(tar_Us)
    Up_interface, P_interface = get_intersection(tar_rayleigh, ref_rels, near=tar_Us)

    if showplot:
        refh_PvUs = interp1d(*ref_mat.hugoniot("Us", "P"))
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

if __name__ == "__main__":
    plt.ion()
    plt.close('all')

    al = EOS("../data/eos_44.dat")
    ch = EOS("../data/eos_32.dat")
    q = EOS("../data/eos_24.dat")

    barrios = pd.read_csv("../data/barrios.dat", sep="\s+", index_col=False)
    ax = barrios.plot.scatter("UpCH", "PCH", label="barrios et al")

    f_PCH = interp1d(*ch.hugoniot("Up", "P"))
    ax.scatter(barrios.UpCH.values, f_PCH(barrios.UpCH.values)/100, color='g', marker="+", 
            label="CH EOS") 

    calc_PCH = np.asarray(
            [impedence_matching_exp(q, ch, UsQ, UsCH)[1] 
                for UsQ, UsCH in barrios[["UsQ", "UsCH"]].values])

    ax.scatter(barrios.UpCH.values, calc_PCH/100, color='r', marker="+", 
            label="Calculated Q Pressure", picker=5) 

    ax.grid()
    plt.legend()

    def onpick(event):
        plt.close("IME")
        UsQ, UsCH = barrios.ix[event.ind][["UsQ", "UsCH"]].values[0]
        impedence_matching_exp(q, ch, UsQ, UsCH, showplot=True)

    ax.figure.canvas.mpl_connect('pick_event', onpick)
