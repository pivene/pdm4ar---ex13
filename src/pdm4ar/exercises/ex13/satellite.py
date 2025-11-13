import sympy as spy

from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters


class SatelliteDyn:
    sg: SatelliteGeometry
    sp: SatelliteParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SatelliteGeometry, sp: SatelliteParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust_l thrust_r", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        extract the state from self.x the following way:
        0x 1y 2psi 3vx 4vy 5dpsi
        """
        # Extract variables
        x, y, psi, v_x, v_y, d_psi = self.x
        F_l, F_r = self.u
        t_f = self.p[0]

        # Extract parameters
        m = self.sp.m_v     # mass of vehicle
        Iz = self.sg.Iz     # rotational inertia
        l_m = self.sg.l_m   # lateral distance from CoG to thrusters

        # Initialize the dynamics vector
        f = spy.zeros(self.n_x, 1)

        # Position dynamics
        f[0] = t_f * v_x
        f[1] = t_f * v_y

        # Orientation dynamics
        f[2] = t_f * d_psi

        # Translational acceleration
        F_total = F_l + F_r
        f[3] = t_f * (F_total * spy.cos(psi) / m)
        f[4] = t_f * (F_total * spy.sin(psi) / m)

        # Angular acceleration
        f[5] = t_f * (l_m * (F_r - F_l) / Iz)

        # Calculate the Jacobians
        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        # Lambdified functions
        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func
