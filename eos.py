import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fmin
from scipy.integrate import odeint

class EOS:
    def __init__(self, filename=None, hugoniot_method="EOS-contour", **kwargs):
        """Import equation of state data

        Args
        ----
        filename (str) : path to the eos data
        hugoniot_method (str) : the method of calculating the hugoniot. Options
            are:
            - EOS-contour : calculate the Hugoniot base on the contour method
                through EOS phase space
            - analytic : Used a fitted function for the Us-Up curve
            - experimental : Fit experimental data points (NOT IMPLEMENTED YET)
                

        Kwargs
        ------
        us (lambda or function) : Needed for 'analytic' calculation of hugoniot
            Needs to be of the form func(up) -> us
        rho0 (float) : if an EOS is not provided, then an initial density is
            needed 

        """
        if filename:
            self._read_eos_table(filename)
        else:
            self._rho0 = kwargs["rho0"]

        if hugoniot_method:
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
        method (str) : method of calculating the hugoniot. See Hugoniot class for valid 
            methods

        Kwargs
        ------
        us (lambda or function) : Needed for 'analytic' calculation of hugoniot
            Needs to be of the form func(up) -> us
        """
        self.hugoniot = Hugoniot(
                mat=self, method=method, **kwargs)

class Hugoniot:

    """

    Parameters
    ----------

    _hvars (dic) : the material properties variables of the hugoniot
        rho, T, P, E, Up, Us, and V

    """

    def __init__(self, mat, method="analytic", **kwargs):
        """Find the hugoniot through a material, mat. 

        Each hugoniot finder method needs to save the variables:  
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
            self._analytic_method(kwargs["upmin"], kwargs["upmax"])
        elif method == "experimental":
            self._exp_method()

    @property
    def rho0(self):
        """Initial density of the material"""
        return self.mat.rho0

    @property
    def V0(self):
        """Initial volume, reciprocal of density, of the material"""
        return 1/self.rho0

    @property
    def T0(self) -> "KeV":
        """Initial tempature"""
        return self._T0

    @property
    def P0(self) -> "GPa":
        """Initial pressure"""
        return self._P0 

    @property
    def E0(self) -> "kJ":
        """Initial internal energy"""
        return self._E0

    def _contour_method(self):
        """Solve for the hugoniot by finding the Zero contour through the EOS
        phase space of func = (E-E0) - 1/2 * (P+P0) * (V0 - V)
        then follow the contour of func = 0

        """

        # set initial conditions of the material 
        self._T0 = 2.5e-5
        self._P0 = self.mat._P_lookup(rho=self.rho0, T=self.T0)
        self._E0 = self.mat._E_lookup(rho=self.rho0, T=self.T0)

        # build the contour function 
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

    def _analytic_method(self, upmin, upmax):
        """Calculate the hugoniot based on the function Us = Us(Up)
        Other variables are calculated from the RK jump conditions and assuming
        P0 = 0 and E0 = 0 

        Requires the UsvUp kwarg to be passed during EOS initialization
        """
        up = np.linspace(upmin,upmax,1000)
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

    def release(self, Pref) -> "GPa":
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
        if self._release_model_name  == "mirrored_hugoniot":
            return self._mirrored_hugoniot(Pref,
                    *self._release_model_args, **self._release_model_kwargs)
        elif self._release_model_name  == "mg":
            return self._mg_release(Pref,
                    *self._release_model_args, **self._release_model_kwargs)
        elif self._release_model_name == "custom":
            return self.custom_release_model(Pref,
                    *self._release_model_args, **self._release_model_kwargs)

    def set_release_model(self, model, *args, **kwargs):
        self._release_model_name = model
        self._release_model_args = args
        self._release_model_kwargs = kwargs

    def custom_release_model(self, *args, **kwargs):
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

    def _mg_release(self, P1, gamma, method="ode"):
        """Mie-Gurniesen Release

        Solves the function Ps = Ph + gamma/V * (Es-Eh)

        Args
        ----
        P1 : Release pressure state
        gamma : constant value of gamma to use in the Mie-Gurniesen correction

        Returns
        -------
        interpolated function of P = P(Up) pressure along the isentrope 
        """

        # Get initial values 
        Us1 = self.YgivenX("Us", "P", P1)
        Up1 = self.YgivenX("Up", "P", P1)
        E1 = self.YgivenX("E", "P", P1)
        V1 = self.YgivenX("V", "P", P1)

        # construct interpolation functions 
        fPhvV = self.fYvX("V", "P")
        fEhvV = self.fYvX("V", "E")

        if method == "hawreliak-integral":
            """Acceptable to be used on non-linear Us-Up hugoniots
            method outlined in Jim's powerpoint "SummaryOfQuartzRelease," emailed
            on 5/13/2017
            """
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

        elif method == "brygoo-ode":
            """Solve the ODE equation outlines in Stephanie's paper:

            S. Brygoo et al. "Development of melted quartz as impedance-matching 
            standard for strong laser shock measurements" June 2, 2008
            """

            raise Exception("You didn't get this method working, dude")

            def c(up):
                if up < 6.358:
                    return [1.87299, 1.667]
                else:
                    return [4.36049, 1.276]

            def dhdv(h, V, Up1):
                a, b = c(Up1)
                fv1 = lambda v : gamma*v**gamma/self.V0
                fv2 = lambda v: b*(1-v/self.V0)

                return fv1(V) * a**2 * fv2(V)**2/(1-fv2(V))**3

            V = np.linspace(V1, self.V0, 1000)
            dV = np.diff(V).mean()
            h1 = 0

            h_v = odeint(dhdv, y0=P1, t=V, args=(Up1,)).T[0]
            Ps = V**(gamma+1)*(h_v - h_v[0]) + fPhvV(V)

        elif method == "hammel-integral":
            """Optimized version for the hawreliak integral method 
            to speed up runtime when implementing the monte carlo error analysis

            Acceptable to be used on non-linear Us-Up hugoniots
            """
            V = np.linspace(V1, self.V0, 1000)
            dV = np.diff(V).mean()

            F = fPhvV(V) - gamma/V * (fEhvV(V) - E1)

            _intF = dV/2 * ( F[:-1]*V[:-1]**gamma + F[1:]*V[1:]**gamma)
            intF = np.cumsum(np.append(0, _intF))

            Ps = F - gamma / np.power(V, gamma+1) * intF

            _Ups = np.sqrt(-np.diff(Ps)/dV)*dV
            Ups = np.cumsum(np.append(Up1, _Ups))

        elif method == "ode":
            """From Knudson 
            solved ODE for equation (8) and equation (9)
            """

            V = np.linspace(V1, self.V0, 1000)
            dV = np.diff(V).mean()

            def ddEdV(fE, v):
                return -(fPhvV(v)*(1 - gamma/2 * (self.V0/v - 1)) \
                        + gamma/v * fE)

            dE = odeint(ddEdV, (E1-self.E0),  V).flatten()
            Ps = -np.diff(dE)/dV
            
            _Ups = np.sqrt(-np.diff(Ps)/dV)*dV
            Ups = np.cumsum(np.append(Up1, _Ups))


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
                "../data/hugoniot_{}.dat".format(self.mat.eosnum), 
                sep="\s+", skiprows=3, index_col=False)
        hugoniot_pts = hugoniot_pts[hugoniot_pts.m == 1]

        Uph, Ph, rhoh = self.get_vars("Up", "P", "rho")

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(Uph, Ph, 'g-o')
        hugoniot_pts.plot.scatter("U", "P", ax=ax1)
        plt.ylim(-10, 800)
        plt.xlim(-.5, 15)
        plt.grid()

        fig = plt.figure()
        ax2 = fig.add_subplot(111)

        ax2.plot(rhoh, Ph, 'g-o')
        hugoniot_pts.plot.scatter("R", "P", ax=ax2)
        plt.ylim(-10, 800)
        plt.xlim(-.5, 15)
        plt.grid()


def quartz_UsvUp(model="knudson"):
    """
    Up must be array type
    """
    def knudson(up):
        """Us-Up relations for the calculation of Quartz hugoniot
        Both Us-Up relations from Knudson's paper are near identical for
        values of up < 25

        M. Knudson and M. Desjarlais Phys Rev B (88) 2013
            Using equation A1 with constants from table XII

        """
        return 6.278 + 1.193*up - 2.505*up*np.exp(-.3701*up)

    def knudson_cubic(up):
        A0 = 1.754
        A1 = 1.862
        A2 = -0.03364
        A3 = 0.0005666
        assert np.all(up < 30)
        return A0 + A1*up + A2*up**2 + A3*up**3

    def hicks(up):
        try:
            up[0]
        except:
            up = np.array(up)
            print("Up wasnt and array")
        us1 = 6.914 + 1.667*(up[up<6.358]-3.0244)
        us2 = 19.501 + 1.276*(up[up>=6.358]-11.865)
        return np.hstack((us1, us2))

    if model == "knudson":
        return knudson
    elif model == "knudson-cubic":
        return knudson_cubic
    elif model == "hicks":
        return hicks

def quartz_mglr(self, P1):
    """The MGLR release model of quarts
    """
    Us1 = self.YgivenX("Us", "P", P1)
    Up1 = self.YgivenX("Up", "P", P1)
    E1 = self.YgivenX("E", "P", P1)
    V1 = self.YgivenX("V", "P", P1)

    S = 1.197
    C = Us1 - S*Up1

    def _gamma(Us):
        """Taken from Marcus' email to Jim (Jun 15 2017)
        """
        a1 = 0.579
        a2 = 0.129
        a3 = 12.81

        if Us > 14.69:
            return a1*(1 - np.exp(-a2*(Us - a3)**1.5))
        else:
            return 0.11016009*Us - 1.4544581;

    def Peff(v):
        """Effective pressure along the hugoniot
        Assumes a linear Us-Up hugoniot with slope defined 
        at the shocked state of the true hugoniot
        """
        return self.rho0*C**2*(self.V0/v - 1)*(self.V0/v) \
            / (S-(S-1)*(self.V0/v))**2

    def Eeff(v):
        """Effective energy along the hugoniot
        """
        return .5*Peff(v)*(self.V0 - v)

    gamma = _gamma(Us1)

    Useff = lambda up : Co + S*up

    V = np.linspace(V1, self.V0, 1000)
    dV = np.diff(V).mean()

    F = Peff(V) - gamma/V * (Eeff(V) - E1)

    _intF = dV/2 * ( F[:-1]*V[:-1]**gamma + F[1:]*V[1:]**gamma)
    intF = np.cumsum(np.append(0, _intF))

    Ps = F - gamma / np.power(V, gamma+1) * intF

    _Ups = np.sqrt(-np.diff(Ps)/dV)*dV
    Ups = np.cumsum(np.append(Up1, _Ups))

    return interp1d(Ups, Ps, bounds_error=False)



def quartz_ode_mglr(self, P1):
    """The MGLR release model of quarts
    """
    Us1 = self.YgivenX("Us", "P", P1)
    Up1 = self.YgivenX("Up", "P", P1)
    E1 = self.YgivenX("E", "P", P1)
    V1 = self.YgivenX("V", "P", P1)

    S = 1.197
    C = Us1 - S*Up1

    def _gamma(Us):
        """Taken from Marcus' email to Jim (Jun 15 2017)
        """
        a1 = 0.579
        a2 = 0.129
        a3 = 12.81

        if Us > 14.69:
            return a1*(1 - np.exp(-a2*(Us - a3)**1.5))
        else:
            return 0.11016009*Us - 1.4544581;

    def Peff(v):
        """Effective pressure along the hugoniot
        Assumes a linear Us-Up hugoniot with slope defined 
        at the shocked state of the true hugoniot
        """
        return self.rho0*C**2*(self.V0/v - 1)*(self.V0/v) \
            / (S-(S-1)*(self.V0/v))**2

    def Eeff(v):
        """Effective energy along the hugoniot
        """
        return .5*Peff(v)*(self.V0 - v)

    gamma = _gamma(Us1)

    Useff = lambda up : Co + S*up

    V = np.linspace(V1, self.V0, 1000)
    dV = np.diff(V).mean()

    def ddEdV(fE, v):
        return -(Peff(v)*(1 - gamma/2 * (self.V0/v - 1)) + gamma/v * fE)

    dE = odeint(ddEdV, (E1-self.E0),  V).flatten()
    Ps = -np.diff(dE)/dV
    
    _Ups = np.sqrt(-np.diff(Ps)/dV)*dV
    Ups = np.cumsum(np.append(Up1, _Ups))

    return interp1d(Ups, Ps, bounds_error=False)


def dump_hyades_eos(mat):
    """I don't remember why this was important
    """
    p = np.vstack((
        np.zeros_like(mat._d), mat._p))
    p = np.hstack((
        np.zeros(shape=(len(mat._t)+1, 1)), p))
    p *= 1e10

    e = np.vstack((
        np.zeros_like(mat._d), mat._e))
    e = np.hstack((
        np.zeros(shape=(len(mat._t)+1, 1)), e))
    e *= 1e10

    d = np.concatenate((
        [len(mat._d)+1, len(mat._t)+1], 
        [0], 
        mat._d, 
        [0], 
        mat._t, 
        p.ravel(), 
        e.ravel()
        ))

    i=0;
    n_chunks = len(d)

    header = """POLYSTYRENE LANL SESAME #7592 DATED: 121488 121588
    32     3.50000000E+00 6.51000000E+00 1.04400000E+00    6102
"""
    with open("test.dat", "w") as f:
        f.write(header)
        for i in range(0, n_chunks, 5):
            chunk = d[i:i+5]
            line = ("{:15e}"*len(chunk) + "\n").format(*chunk)
            f.write(line)







