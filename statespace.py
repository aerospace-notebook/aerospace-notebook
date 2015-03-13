"""
A state space module.
"""

import sympy
import pylab as pl
import numpy
import scipy.integrate

# pylint: disable=no-member
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name

class Data(object):
    """
    A data structure for the state space.
    """

    def __init__(self):
        self.t = []
        self.x = []
        self.u = []
        self.y = []

    def update(self, t, x, u, y):
        """
        Update data.
        """
        self.t.append(t)
        self.x.append(x)
        self.u.append(u)
        self.y.append(y)

    def pack(self):
        """
        Pack as array.
        """
        self.t = pl.array(self.t)
        self.x = pl.array(self.x)
        self.u = pl.array(self.u)
        self.y = pl.array(self.y)

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()


class StateSpace(object):
    """
    A generic state space class.
    """

    def __init__(self, x, u, f, g):

        # create eval functions
        xd = sympy.DeferredVector('x')
        ud = sympy.DeferredVector('u')
        ss_sub = {}
        t = sympy.symbols('t')
        ss_sub.update({x[i]: xd[i] for i in range(len(x))})
        ss_sub.update({u[i]: ud[i] for i in range(len(u))})
        use_array = [{'ImmutableMatrix': numpy.array}, 'numpy']
        f_eval = sympy.lambdify((t, xd, ud), f.subs(ss_sub), use_array)
        g_eval = sympy.lambdify((t, xd, ud), g.subs(ss_sub), use_array)

        # class data
        self.x = x
        self.u = u
        self.f = f
        self.g = g
        self.ss_sub = ss_sub
        self.f_eval = f_eval
        self.g_eval = g_eval

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__dict__.__repr__()

    def simulate(self, x0, u0, dt=0.1, tf=10, contr_eval=None):
        """
        Simulate system.
        """
        sim = scipy.integrate.ode(self.f_eval)
        sim.set_initial_value(x0)
        u = u0
        data = Data()
        while sim.t + dt < tf:
            t = sim.t
            x = sim.y
            y = self.g_eval(t, x, u)[:, 0]
            if contr_eval is not None:
                u = contr_eval(y)
            sim.set_f_params(u)
            sim.integrate(t + dt)
            data.update(t, x, u, y)
        data.pack()
        return data

    def linearize(self):
        """
        Linearize system.
        """
        A = self.f.jacobian(self.x)
        B = self.f.jacobian(self.u)
        C = self.g.jacobian(self.x)
        D = self.g.jacobian(self.u)
        return LinearStateSpace(self.x, self.u, A, B, C, D)


class LinearStateSpace(StateSpace):
    """
    A linear state space.
    """

    def __init__(self, x, u, x0, u0, A, B, C, D):
        f = A*(x-x0) + B*(u-u0)
        g = C*(x-x0)+ D*(u-u0)
        super(LinearStateSpace, self).__init__(x, u, f, g)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def eval(self, x0, u0):
        """
        Evaluate at a point.
        """
        x0_sub = {self.x[i]:x0[i] for i in range(len(self.x))}
        u0_sub = {self.u[i]:u0[i] for i in range(len(self.u))}
        return [
            pl.array(self.A.subs(x0_sub).subs(u0_sub)).astype(float),
            pl.array(self.B.subs(x0_sub).subs(u0_sub)).astype(float),
            pl.array(self.C.subs(x0_sub).subs(u0_sub)).astype(float),
            pl.array(self.D.subs(x0_sub).subs(u0_sub)).astype(float)]

# vim: set et fenc=utf-8 ff=unix sts=0 sw=4 ts=4 ft=python :
