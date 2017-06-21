import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fmin
from scipy.integrate import odeint

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

        # make den and temp into a 2D array
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

    def _E_lookup(self, rho:"g/cc", T:"keV")->"kJ":
        """Find an energy value given a density and temperature

        Needed in the calculation of the hugoniot
        """
        return self._interp_e(rho, T)

    def _P_lookup(self, rho:"g/cc", T:"keV")->"GPa":
        """Find a Pressure value given a density and temperature

        Needed in the calculation of the hugoniot
        """
        return self._interp_p(rho, T)

    def plot_energy(self):
        self._plot2D(self.eng, "Energy")

    def plot_pressure(self):
        self._plot2D(self.pres, "Pressure")

    def _calculate_hugoniot(self, method="EOS-contour", **kwargs):
        """Calculate the materials hugoniot

        Args
        ----
        method : method of calculating the hugoniot. See Hugoniot class for 
            valid mehtods
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
            - EOS-contour : calclulate the Hugoniot base on the contour method
                through EOS phase space
            - analytic : Used a fitted function for the Us-Up curve
            - experimental : Fit experimental data points (NOT IMPLEMENTED YET)
        """
        self.mat = mat
        self._hvars = {}
        if method == "EOS-contour":
            self._contour_method()
        elif method == "analytic":
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
        """Calculate the hugoniot based on the function Us = Us(Up)
        Other variables are calculated from the RK jump conditions and assuming
        P0 = 0 and E0 = 0 

        Requires the UsvUp kwarg to be passed during EOS initialization
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

        Args
        ----
        (str) any valid hugoniot variables 
        """
        return (self._hvars[key] for key in args)

    def fYvX(self, X, Y, bounds_error=False):
        """Get an interpolated function of Y = Y(X) of two hugoniot variables

        Args
        ----
        X : X value hugoniot variable
        Y : Y value hugoniot variable
        bounds_error : throws an error if function recieves value outside of 
            interpolation range

        Returns
        -------
        numpy interp1d function

        Example
        -------
        fPvUp = hugoniot.fYvX("Up", "P")
        """
        f = interp1d(*self.get_vars(X, Y), bounds_error=bounds_error)
        return f
    
    def aproxY_given_X(self, Y, X, X0):
        """Return an approximate value for a hugoniot variable, given another
        hugoniot variable

        Args
        ----
        Y, X (str) : Hugoniot variables to use i.e. "P", "E", "Up", "Us", "V"
        X0 (float) : the reference value
        """
        Y, X = self.get_vars(Y, X)
        return Y[np.argmin(np.abs(X-X0))]

    def YgivenX(self, Y, X, X0):
        """Get the Value of a hugoniot variable, Y, given a hugoniot variable, X
        """
        fY = self.fYvX(X, Y)
        return float(fY(X0))

    def release(self, *args, model="mirrored_hugoniot", **kwargs) -> "GPa":
        """Calculate the release isentrope using the Gruneisen correction to the
        hugoniot 

        This method can be over written if a custom release is desired, the
        return function just needs to be matched.

        Args
        ----
        P1 : the pressure state to release from 
        model:
            mirrored_hugoniot : use the mirror image of the hugoniot in the 
                P-Up plane to estimate the release isentroope 
                args
                ---
                P1 : Pressure state to release from =
            mg : Use the Mie-Gruniesen EOS to calculate the isentrope
                args
                ----
                P1 : Pressure state to release from =
                gamma : a constant value for gamma
            custom : Use a custom release model
                requires overloading the Hugoniot.release_model method
                To do this, the method type needs to be passed.
                See release_model docstring for more info

        Return 
        ------
        interpolated function of pressure along the isentrope in P-Up plane
        """
        if model  == "mirrored_hugoniot":
            return self._mirrored_hugoniot(*args, **kwargs)
        elif model  == "mg":
            return self._mg_release(*args, **kwargs)
        elif model == "custom":
            return self.release_model(*args, **kwargs)

    def release_model(self, *args, **kwargs):
        """Should be over written with a custom release model

        To do this, the method type needs to be passed.
        I.E.: 
        import types
        q.hugoniot.release_model = types.MethodType(quartz_mglr, 
                                                    q.hugoniot) 

        Where q is an EOS instance 

        Must Return
        -----------
        interpolated function of P = P(Up) pressure along the isentrope 
        """
        pass

    def _mirrored_hugoniot(self, P1):
        """Return a mirrored hugoniot in the P-Up plane about a particle velocity

        Args
        ----
        P1 : Hugoniot pressure state to release from 

        Returns
        ------
        Interpolated function of P = P(Up) of the release hugoniot
        """
        aproxUp = self.aproxY_given_X("Up", "P", X0=P1)
        fPvUp = self.fYvX("Up", "P", bounds_error=False)
        Up1 = fmin(
            lambda up: (fPvUp(up)-P1)**2, aproxUp, disp=None)
        return lambda up: fPvUp(2*Up1-up)

    def _mg_release(self, P1, gamma, method="integral"):
        """Mie-Gurniesen Release

        Solves the function Ps = Ph + gamma/V * (Es-Eh)
        Using the integral method from J. Hawreliak

        Args
        ----
        P1 : Release pressure state
        gamma : constant value of gamma to use in the Mie-Gurniesen correction

        Returns
        -------
        interpolated function of P = P(Up) pressure along the isentrope 
        """
        Us1 = self.YgivenX("Us", "P", P1)
        Up1 = self.YgivenX("Up", "P", P1)
        E1 = self.YgivenX("E", "P", P1)
        V1 = self.YgivenX("V", "P", P1)

        fPhvV = self.fYvX("V", "P")
        fEhvV = self.fYvX("V", "E")

        if method == "integral":
            _steps = 1000
            dV = (self.V0-V1)/_steps
            F = [P1]
            V = [V1]
            Ps = [P1]
            E = [E1]
            intF = [0]
            Ups = [Up1]

            for n in range(1, _steps):

                V.append(V[n-1] + dV)

                F.append(fPhvV(V[n]) - gamma/V[n]*(fEhvV(V[n]) - E1))

                intF.append(
                        (V[n]**gamma * F[n] + V[n-1]**gamma * F[n-1]) * dV/2 \
                        + intF[n-1]
                        )

                # Find the pressures along the isentrope 
                Ps.append(F[n] - gamma / np.power(V[n], gamma+1) * intF[n])

                # Find the particle velocities corresponding the release isentrope
                # pressures
                Ups.append(Ups[n-1] + np.sqrt((Ps[n-1] - Ps[n])/dV)*dV)
        elif method == "ode":
            def dhdv(h, V, Up1):
                a, b = c(Up1)
                fv1 = lambda v : gamma*v**gamma/self.V0
                fv2 = lambda v: b*(1-v/self.V0)

                return fv1(V) * a**2 * fv2(V)**2/(1-fv2(V))**3

            def c(up):
                if up < 6.358:
                    return [1.87299, 1.667]
                else:
                    return [4.36049, 1.276]

            V = np.linspace(V1, self.V0, 1000)
            dV = np.diff(V).mean()

            h_v = odeint(dhdv, P1, V, args=(Up1,)).T[0]
            Ps = V**(gamma+1)*(h_v - h_v[0]) + fPhvV(V)

        #import ipdb; ipdb.set_trace()


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




