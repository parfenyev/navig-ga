import numpy as np
import h5py
from scipy.ndimage import map_coordinates

######################
#  Velocity Fields
######################

class VelocityField:
    def velX(self, x, y, t):
        raise NotImplementedError

    def velY(self, x, y, t):
        raise NotImplementedError

    def vort(self, x, y, t):
        raise NotImplementedError

# Example 1: Steady Vortex Flow
class VortexFlow(VelocityField):
    def __init__(self, omega=0.9):
        self.ω = omega

    def velX(self, x, y, t):
        return -self.ω * y

    def velY(self, x, y, t):
        return self.ω * x

    def vort(self, x, y, t):
        return 2*self.ω

# Example 2: Sink+Rotation Flow
class SRFlow(VelocityField):
    def __init__(self, a=-0.3, b=0.5):
        self.a = a
        self.b = b

    def velX(self, x, y, t):
        return self.a*x+(t-self.b)*y

    def velY(self, x, y, t):
        return self.a*y+(self.b-t)*x

    def vort(self, x, y, t):
        return 2*(self.b-t)

# Example 3: Taylor-Green Flow
class TGFlow(VelocityField):
    def __init__(self, u0=1.0, k=3.0):
        self.u0 = u0
        self.k = k

    def velX(self, x, y, t):
        return self.u0*np.sin(self.k*x)*np.cos(self.k*y)

    def velY(self, x, y, t):
        return -self.u0*np.cos(self.k*x)*np.sin(self.k*y)

    def vort(self, x, y, t):
        return 2*self.u0*self.k*np.sin(self.k*x)*np.sin(self.k*y)

# Examples 4 and 5: 2D Turbulence
class TurbFlow(VelocityField):
    def __init__(self, filename=None, *, u_data=None, v_data=None, steady=True):
        self.n = 512
        self.L = 2*np.pi
        self.t_step = 0.005
        self.frame_max = 801
        self.steady = steady

        # ---------------------------------------------------------------
        # Construct from existing NumPy arrays (used by worker processes)
        # ---------------------------------------------------------------
        if u_data is not None:
            self.u_data = u_data
            self.v_data = v_data
        # --------------------------------------------------------
        # Construct from HDF5 file (used only in the main process)
        # --------------------------------------------------------
        else:
            with h5py.File(filename, "r") as f:
                if steady:
                    # Only load first snapshot
                    self.u_data = np.asarray(f["vel_x"][0, :, :], dtype=np.float32)
                    self.v_data = np.asarray(f["vel_y"][0, :, :], dtype=np.float32)
                else:
                    # Load all time frames
                    self.u_data = np.asarray(f["vel_x"][0:self.frame_max, :, :], dtype=np.float32)  # shape (nt, n, n)
                    self.v_data = np.asarray(f["vel_y"][0:self.frame_max, :, :], dtype=np.float32)


        # Compute max velocity for first snapshot
        self.u0_max = np.max(np.sqrt(self.u_data[0]**2 + self.v_data[0]**2))
        
    def _interp_xyt(self, field, x, y, t):
        if self.steady:
            xi = self.n * x / self.L
            yi = self.n * y / self.L
            coords = np.array([[yi], [xi]])
            return map_coordinates(
                field,
                coords,
                order=1,
                mode="wrap"
            )[0]

        # --- time cutoff ---
        if (t < 0.0) or (t > (self.frame_max - 1) * self.t_step):
            return 0.0

        ti = t / self.t_step
        xi = self.n * x / self.L
        yi = self.n * y / self.L

        coords = np.array([[ti], [yi], [xi]])  # axes: t, y, x
        return map_coordinates(
            field,
            coords,
            order=1,
            mode="wrap"
        )[0]

    def velX(self, x, y, t):
        return self._interp_xyt(self.u_data, x, y, t)

    def velY(self, x, y, t):
        return self._interp_xyt(self.v_data, x, y, t)

    def vort(self, x, y, t):
        # Approximate derivatives numerically using small delta
        dx = self.L / self.n
        dy = self.L / self.n

        u_y_plus = self.velX(x, (y + dy) % self.L, t)
        u_y_minus = self.velX(x, (y - dy) % self.L, t)
        du_dy = (u_y_plus - u_y_minus) / (2 * dy)

        v_x_plus = self.velY((x + dx) % self.L, y, t)
        v_x_minus = self.velY((x - dx) % self.L, y, t)
        dv_dx = (v_x_plus - v_x_minus) / (2 * dx)

        return dv_dx - du_dy
      
######################
#  Configs
######################

VORTEX_CONFIG = dict(
    steady_env=True,
    Va=1.0, # agent's velocity
    Δt=0.012, # reaction time
    T_max=5.0, # max duration of an episode
    n_steps=10, # num of integration steps

    # start zone
    xA=0.5,
    yA=np.sqrt(3)/2,
    rA=0.01,

    # target zone
    xB=1.0,
    yB=0.0,
    rB=0.01,

    # domain
    x_min=-0.3,
    x_max=1.2,
    y_min=-0.3,
    y_max=1.2,
)

SR_CONFIG = dict(
    steady_env=False,
    Va=1.0, # agent's velocity
    Δt=0.012, # reaction time
    T_max=5.0, # max duration of an episode
    n_steps=10, # num of integration steps

    # start zone
    xA=0.5,
    yA=np.sqrt(3)/2,
    rA=0.01,

    # target zone
    xB=1.0,
    yB=0.0,
    rB=0.01,

    # domain
    x_min=-0.3,
    x_max=1.2,
    y_min=-0.3,
    y_max=1.2,
)

TG_CONFIG = dict(
    steady_env=True,
    Va=0.1, # agent's velocity
    Δt=np.pi/30, # reaction time
    T_max=90.0, # max duration of an episode
    n_steps=10, # num of integration steps

    # start zone
    xA=2*np.pi/3,
    yA=np.pi/3,
    rA=0.2,

    # target zone
    xB=3*np.pi/2,
    yB=3*np.pi/2,
    rB=0.2,

    # domain
    x_min=0,
    x_max=2*np.pi,
    y_min=0,
    y_max=2*np.pi,
)

TURB_STEADY_P1 = dict(
    steady_env=True,
    T_max=4.0, # max duration of an episode
    n_steps=10, # num of integration steps
    Va=3.8185675,
    Δt=0.008227151,
    # start zone
    xA=1.2,
    yA=5.5,
    rA=0.2,
    # target zone
    xB=5.2,
    yB=1.8,
    rB=0.2,
    # domain
    x_min=0,
    x_max=2*np.pi,
    y_min=0,
    y_max=2*np.pi,
)

TURB_STEADY_P2 = dict(
    steady_env=True,
    T_max=4.0, # max duration of an episode
    n_steps=10, # num of integration steps
    Va=3.8185675,
    Δt=0.008227151,
    # start zone
    xA=5.8,
    yA=5.5,
    rA=0.2,
    # target zone
    xB=1.0,
    yB=1.0,
    rB=0.2,
    # domain
    x_min=0,
    x_max=2*np.pi,
    y_min=0,
    y_max=2*np.pi,
)

TURB_TIME_P1 = dict(
    steady_env=False,
    T_max=4.0, # max duration of an episode
    n_steps=10, # num of integration steps
    Va=3.8185675,
    Δt=0.008227151,
    # start zone
    xA=5.5,
    yA=2.5,
    rA=0.2,
    # target zone
    xB=1.0,
    yB=5.0,
    rB=0.2,
    # domain
    x_min=0,
    x_max=2*np.pi,
    y_min=0,
    y_max=2*np.pi,
)

TURB_TIME_P2 = dict(
    steady_env=False,
    T_max=4.0, # max duration of an episode
    n_steps=10, # num of integration steps
    Va=3.8185675,
    Δt=0.008227151,
    # start zone
    xA=1.4,
    yA=3.0,
    rA=0.2,
    # target zone
    xB=5.2,
    yB=1.8,
    rB=0.2,
    # domain
    x_min=0,
    x_max=2*np.pi,
    y_min=0,
    y_max=2*np.pi,
)