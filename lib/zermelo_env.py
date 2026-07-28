import numpy as np
import gymnasium as gym
from typing import Optional
import matplotlib.pyplot as plt

from lib.velocity_field import *

######################
#  GymEnv  
###################### 
class ZermeloEnv(gym.Env):

    def __init__(self, cfg: dict, vel_field: VelocityField):
        super().__init__()
       
        self.vel_field = vel_field # external current
        self.trajectory = [] # save trajectory for visualization
       
        self.Va = cfg["Va"] # agent's velocity
        self.Δt = cfg["Δt"] # reaction time
        self.T_max = cfg["T_max"] # max duration of an episode
        self.steady = cfg["steady_env"] # observation=(x,y) (steady_env=True) or observation=(x,y,t) (steady_env=False) 

        # integration time-step
        self.n_steps = cfg["n_steps"]
        self.dt = self.Δt / self.n_steps 

        # start and target zones
        self.xA, self.yA, self.rA = cfg["xA"], cfg["yA"], cfg["rA"]
        self.xB, self.yB, self.rB = cfg["xB"], cfg["yB"], cfg["rB"]

        # domain size
        self.x_min, self.x_max = cfg["x_min"], cfg["x_max"]
        self.y_min, self.y_max = cfg["y_min"], cfg["y_max"]

        if self.steady:
            self.observation_space = gym.spaces.Box(
                low=np.array([self.x_min, self.y_min], dtype=np.float32),
                high=np.array([self.x_max, self.y_max], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array([self.x_min, self.y_min, 0.0], dtype=np.float32),
                high=np.array([self.x_max, self.y_max, self.T_max], dtype=np.float32),
                dtype=np.float32
            )

        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        self._agent_location = np.array([self.xA, self.yA], dtype=np.float32)
        self.time = 0.0

    def next_state(self, x, y, t, theta):
        # integrate kinematic eqs to obtain next state:
        p = np.array([x, y], dtype=np.float32)
        u = np.array([np.cos(theta), np.sin(theta)])
        t_star = t

        for _ in range(self.n_steps):
            v = np.array([self.vel_field.velX(p[0], p[1], t_star), self.vel_field.velY(p[0], p[1], t_star)])
            p += self.dt * (v + self.Va * u)
            t_star += self.dt

        return p[0], p[1]
    
    # -----------------------------
    # Gym API
    # -----------------------------
    def _get_obs(self):
        if self.steady:
            return self._agent_location
        else:
            return np.array(
                [self._agent_location[0],
                 self._agent_location[1],
                 self.time], dtype=np.float32
            )
        
    def _get_info(self):
        return {"time": self.time}

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        
        # Randomly place the agent in the start zone
        phi = 2 * np.pi * self.np_random.random()
        r = self.rA * np.sqrt(self.np_random.random())

        self._agent_location[:] = (
            self.xA + r * np.cos(phi),
            self.yA + r * np.sin(phi),
        )

        # clean step counter and trajectory
        self.time = 0.0
        self.trajectory = [self._agent_location.copy()]

        return self._get_obs(), self._get_info()

    def is_terminal(self):
        # check collision with boundaries or target 
        x, y = self._agent_location

        if x <= self.x_min or y <= self.y_min or x >= self.x_max or y >= self.y_max:
            return True

        dx = self.xB - x
        dy = self.yB - y
        
        return (dx * dx + dy * dy) <= self.rB*self.rB

    def step(self, action):
        x, y = self._agent_location
        x_next, y_next = self.next_state(x, y, self.time, action.item())

        self._agent_location[:] = (x_next, y_next)
        self.time += self.Δt
        self.trajectory.append(self._agent_location.copy())

        reward = (
            -self.Δt
            #+ np.linalg.norm([self.xB-x, self.yB-y])/self.Va
            #- np.linalg.norm([self.xB-x_next, self.yB-y_next])/self.Va
        )

        # additional reward for hitting the boundary
        if (x_next<=self.x_min) or (y_next<=self.y_min) or (x_next>=self.x_max) or (y_next>=self.x_max):
            reward -= 2 * self.T_max

        terminated = self.is_terminal()
        truncated = self.time > self.T_max

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # -----------------------------
    # Visualization
    # -----------------------------
    def plot_trajectory(self, n_vort=200, n_vel=36, cmap="viridis"):
        traj = np.array(self.trajectory)
        
        fig, ax = plt.subplots(figsize=(6, 6))

        # Vorticity field
        x = np.linspace(self.x_min, self.x_max, n_vort)
        y = np.linspace(self.y_min, self.y_max, n_vort)
        X, Y = np.meshgrid(x, y)

        VORT = np.vectorize(self.vel_field.vort)(X, Y, self.time)
        vmax = np.max(np.abs(VORT))

        im = ax.pcolormesh(X, Y, VORT, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax, alpha=0.85)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Vorticity")

        # Velocity field (quiver)
        xq = np.linspace(self.x_min, self.x_max, n_vel)
        yq = np.linspace(self.y_min, self.y_max, n_vel)
        Xq, Yq = np.meshgrid(xq, yq)

        U = np.vectorize(self.vel_field.velX)(Xq, Yq, self.time)
        V = np.vectorize(self.vel_field.velY)(Xq, Yq, self.time)

        ax.quiver(Xq, Yq, U, V, color="k", alpha=0.8)

        # Trajectory
        ax.plot(traj[:, 0], traj[:, 1], color="tomato", lw=2.0)

        ax.add_patch(plt.Circle((self.xA, self.yA), self.rA, color="tomato", alpha=1))
        ax.add_patch(plt.Circle((self.xB, self.yB), self.rB, color="tomato", alpha=1))
        ax.text(self.xA, self.yA, "A", color="black", fontsize=14, ha="center", va="center", zorder=5)
        ax.text(self.xB, self.yB, "B", color="black", fontsize=14, ha="center", va="center", zorder=5)

        # Formatting
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        plt.tight_layout()
        plt.show()