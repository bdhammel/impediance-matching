import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from  scipy.interpolate import interp1d, interp2d
from  scipy.optimize import fmin

class EOS:
    def __init__(self, filename, hugoniot_method="EOS-contour", **kwargs):
        """Import equation of state data

        Args
        ----
        filename
        hugoniot_method

        Kwargs
        ------
        us (lambda or function) : Needed for 'analytic' calculation of hugoniot
            Needs to be of the form func(up) -> us

        """
        self._read_eos_table(filename)
        self._calculate_hugoniot(hugoniot_method, **kwargs)

    def _read_eos_table(self, filename):
        """Read in an EOS table from the HYADES database

        Data has the units:
        den : g/cc
        temp : keV
        pres : dyne/cm2
        eng : erg

        Converts:
        dunes/cm2 to GPa  
        erg to kJ
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
        self._p = np.reshape(import_data(NT*NR), (NT, NR))[1:,1:] * 1e-10
        self._e = np.reshape(import_data(NT*NR), (NT, NR))[1:,1:] * 1e-10

        self._dden, self._ttemp = np.meshgrid(self._d, self._t)

    @property
    def _interp_p(self):
        return interp2d(self._d, self._t, self._p)

    @property
    def _interp_e(self):
        return interp2d(self._d, self._t, self._e)

    @property
    def rho0(self) -> "g/cc":
        return self._rho0

    @property
    def V0(self) -> "cc/g":
        """Specific volume
        """
        return 1/self._rho0

    @property
    def T0(self) -> "KeV":
        return self._T0

    @property
    def P0(self) -> "GPa":
        return self._P0 

    @property
    def E0(self) -> "kJ":
        return self._E0

    @property
    def den(self) -> "g/cc":
        return self._dden

    @property
    def V(self) -> "cc/g":
        return 1/self._dden

    @property
    def temp(self) -> "KeV":
        return self._ttemp

    @property
    def pres(self) -> "GPa":
        return self._p

    @property
    def eng(self) -> "kJ":
        return self._e

    def _plot2D(self, Z, plotname):
        """Base plotting script for 2D plot
        """
        plt.figue(plotname)
        plot = plt.pcolormesh(self.den, self.temp, Z, norm=LogNorm())
        plt.colorbar(plot)
        plt.ylim(1e-5, 100)
        plt.yscale('log')
        plt.xlabel("Density")
        plt.ylabel("Temperature")
        plt.draw()

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

    def plot_energy(self):
        self._plot2D(self.eng, "Energy")

    def plot_pressure(self):
        self._plot2D(self.pres, "Pressure")

    def _calculate_hugoniot(self, method="EOS-contour", **kwargs):
        """Calculate the materials hugoniot using the contour method
        """
        self.hugoniot = Hugoniot(self, method, **kwargs)

class Hugoniot:

    def __init__(self, mat, method="analytic", **kwargs):
        """Find the hugoniot through a material, mat

        Each hugoniot finder needs to save the variables: 
        rho, T, P, E, Up, Us, and V

        Args
        ----
        mat (EOS) : material to find the hugoniot for
        method (srt) : method of calculating the hugoniot

        Kwargs
        ------
        us
        gamma

        """
        self.mat = mat
        self._hvars = {}
        if method == "EOS-contour":
            self._contour_method()
        elif method == "analytic":
            self._gamma = kwargs["gamma"]
            self._UsvUp = kwargs["UsvUp"]
            self._analytic_method()
        elif method == "experimental":
            self._exp_method()

    @property
    def rho0(self):
        return self.mat.rho0

    @property
    def V0(self):
        return 1/self.rho0

    @property
    def T0(self) -> "KeV":
        return self._T0

    @property
    def P0(self) -> "GPa":
        return self._P0 

    @property
    def E0(self) -> "kJ":
        return self._E0

    def _contour_method(self):
        """Solve for the hugoniot by finding the Zero contour through the EOS
        phase space of func = (E-E0) - 1/2 * (P+P0) * (V0 - V)
        then follow the contour of func = 0

        """

        self._T0 = 2.5e-5
        self._P0 = self.mat._P_lookup(rho=self.rho0, T=self.T0)
        self._E0 = self.mat._E_lookup(rho=self.rho0, T=self.T0)

        zero = self.mat.eng - self.E0 - .5*(self.mat.pres + self.P0) * (self.V0 - self.mat.V)

        # Use Matplotlib to find the contour
        # This method should be checked and probably replaced with a better one 
        # Unsure how accurate the interpolation is to find func = 0
        fig = plt.figure('tmp')
        cs = plt.contour(self.mat._d, self.mat._t, zero, [0])
        plt.close(fig)
        p = cs.collections[0].get_paths()[0]
        v = p.vertices

        self._hvars["rho"] = v[:,0]
        self._hvars["T"] = v[:,1]
        self._hvars["P"] = np.asarray([
            self.mat._P_lookup(rho=rho, T=T)[0] for rho, T in v])
        self._hvars["E"] = np.asarray([
            self.mat._E_lookup(rho=rho, T=T)[0] for rho, T in v])
        self._hvars["Up"] = np.sqrt(
                (self._hvars["rho"]-self.rho0)/(self._hvars["rho"] * self.rho0)  
                * (self._hvars["P"] - self.P0)
                )
        self._hvars["Us"] = (self._hvars["P"] - self.P0)/(self.rho0*self._hvars["Up"])
        self._hvars["V"] = 1/self._hvars["rho"]

    def _analytic_method(self):
        """
        """
        up = np.linspace(0,100,1000)
        us = self._UsvUp(up)

        self._P0 = 0
        self._E0 = 0

        P = self.rho0*us*up
        rho = self.rho0*us/(us-up)
        V = 1/rho
        E = .5*P*(self.V0 - V)

        self._hvars["Us"] = us
        self._hvars["Up"] = up
        self._hvars["P"] = P
        self._hvars["E"] = E
        self._hvars["V"] = V
        self._hvars["rho"] = rho

    def get_vars(self, *args):
        """Return calculated points on the hugoniot
        """
        return (self._hvars[key] for key in args)

    def release_isentrope(self, *args, method="hawreliak", **kwargs) -> "GPa":
        """Calculate the release isentrope using the Gruneisen correction to the
        hugoniot 

        This method should be over written

        Args
        ----
        P0 : the pressure state to release from 
        gamma : The gurneisen parameter

        Return 
        ------
        interpolated function of pressure along the isentrope in P, Up plane
        """
        if method == "hawreliak":
            return self._hawreliak_release(*args, **kwargs)
        elif method == "knudson":
            return self._knudson_release(*args)
        elif method == "brygoo":
            return self._brygoo_release(*args)

    def _knudson_release(self, Up):
        """
        Following the formalism outlined in M. Knudson and M. Desjarlais 2013
        """
        rho, P, E, Up, Us = self.get_vars("rho", "P", "E", "Up", "Us")


        # Calculate Co of the equation Us = C0 + SUp
        # Equation (4) 
        Co = Us - S*up
        print(Co)

        # Get the value for a non-constant gamma
        # Equation (6)

    def _hawreliak_release(self, P0):
        """
        Args
        ----
        P0 : Release pressure state
        """

        rho, P, E, Up, Us = self.get_vars("rho", "P", "E", "Up", "Us")
        fUpvV = interp1d(1/rho, Up, bounds_error=False)
        fPvV = interp1d(1/rho, P, bounds_error=False)
        fEvV = interp1d(1/rho, E)
        fUsvP = interp1d(P, Us)

        gamma = self._gamma(fUsvP(P0))

        # On release V0 is the volume that the shocked pressure state
        # and the final Volume to consider is when P=0
        aproxV0 = 1/rho[np.argmin(np.abs(P-P0))]
        aproxVf = 1/rho[np.argmin(np.abs(P))]
        V0 = fmin(lambda v: np.abs(fPvV(v) - P0), aproxV0, disp=False)

        V = np.linspace(aproxVf, V0, 10000)
        dV = np.average(np.diff(V))

        F = fPvV(V) - gamma/V*(fEvV(V) - fEvV(V0))

        # the last point in the array is the Pressure to release from, 
        # therefore, the integral needs to be performed in reverse order - to 
        # traverse away from the release pressure 
        intF = np.cumsum(
                np.append(
                    .5*( np.power(V[:-1], gamma)*F[:-1] 
                    - np.power(V[1:], gamma)*F[1:]) * dV,
                    0
                    )[::-1]
                )[::-1]

        # Find the pressures along the isentrope 
        Ps = F - gamma / np.power(V, gamma+1) * intF

        # Find the particle velocities corresponding the release isentrope
        # pressures
        Ups = np.cumsum(
                np.append(
                    np.sqrt(np.diff(Ps))*np.sqrt(-dV),
                    fUpvV(V0)
                    )[::-1]
                )[::-1]

        return interp1d(Ups, Ps, bounds_error=False)


    def get_rayleigh(self, Us):
        """Get the Rayleigh line through P-Up space given a shock speed

        Args
        ---
        Us : shock speed km/s
        """
        return lambda u : self.rho0 * Us * u + self.P0
        

    def check_hugoniot(self):
        """Check the calculated hugoniot against the experimental points from 
        http://www.ihed.ras.ru/rusbank/
        """
        hugoniot_pts = pd.read_table(
                "../data/hugoniot_{}.dat".format(self.eosnum), 
                sep="\s+", skiprows=3, index_col=False)
        hugoniot_pts = hugoniot_pts[hugoniot_pts.m == 1]

        Uph, Ph = self.get_vars("Up", "P")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(Uph, Ph, 'g-o')
        hugoniot_pts.plot.scatter("U", "P", ax=ax)
        plt.ylim(-10, 800)
        plt.xlim(-.5, 15)
        plt.grid()


    def mirrored_hugoniot(f_P, mirror_about_up=None, mirror_at_p=None, aproxUp=15):
        """Return a mirrored hugoniot in the P-Up plane about a particle velocity

        Args
        ----
        mirror_about_up : the particle velocity to mirror the hugoniot about
        """
        if mirror_about_up:
            return lambda up: f_P(2*mirror_about_up-up)
        if mirror_at_p:
            mirror_about_up = fmin(
                    lambda up: (f_P(up)-mirror_at_p)**2, 
                    aproxUp, 
                    disp=None)
            return lambda up: f_P(2*mirror_about_up-up)


