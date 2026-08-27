import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt

import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from lib.zermelo_env import *
from lib.shared_memory_utils import *

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

def plot_rewards(generation_best_rewards, generation_mean_rewards, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(generation_best_rewards, dtype=torch.float)
    mean_rewards_t = torch.tensor(generation_mean_rewards, dtype=torch.float)

    plt.xlabel('Generation')
    plt.ylabel('Best and Mean Reward')
    plt.plot(rewards_t.numpy())
    plt.plot(mean_rewards_t.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())    

def plot_trajectories(env, trajectories, time, n_vort=200, n_vel=36, cmap="viridis"):
        
        fig, ax = plt.subplots(figsize=(6, 6))

        # Vorticity field
        x = np.linspace(env.x_min, env.x_max, n_vort)
        y = np.linspace(env.y_min, env.y_max, n_vort)
        X, Y = np.meshgrid(x, y)

        VORT = np.vectorize(env.vel_field.vort)(X, Y, time)
        vmax = np.max(np.abs(VORT))

        im = ax.pcolormesh(X, Y, VORT, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax, alpha=0.85, zorder=1)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Vorticity")

        # Velocity field (quiver)
        xq = np.linspace(env.x_min, env.x_max, n_vel)
        yq = np.linspace(env.y_min, env.y_max, n_vel)
        Xq, Yq = np.meshgrid(xq, yq)

        U = np.vectorize(env.vel_field.velX)(Xq, Yq, time)
        V = np.vectorize(env.vel_field.velY)(Xq, Yq, time)

        ax.quiver(Xq, Yq, U, V, color="k", alpha=0.8, zorder=2)

        # Trajectory
        #ax.plot(traj[:, 0], traj[:, 1], color="tomato", lw=2.0)
        colors = list(plt.cm.jet(np.linspace(0, 1, len(trajectories)-1)))
        colors.append("white")
    
        for traj, color in zip(trajectories, colors):
            ax.plot(traj[:, 0], traj[:, 1], lw=2, color=color, zorder=3)

        ax.add_patch(plt.Circle((env.xA, env.yA), env.rA, color="tomato", alpha=1, zorder=4))
        ax.add_patch(plt.Circle((env.xB, env.yB), env.rB, color="tomato", alpha=1, zorder=4))
        ax.text(env.xA, env.yA, "A", color="black", fontsize=14, ha="center", va="center", zorder=5)
        ax.text(env.xB, env.yB, "B", color="black", fontsize=14, ha="center", va="center", zorder=5)

        # Formatting
        ax.set_xlim(env.x_min, env.x_max)
        ax.set_ylim(env.y_min, env.y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        plt.tight_layout()
        plt.show()
        plt.close(fig)
    
# Policy Network (Forward-only)
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, act_dim),
        )

    def forward(self, x):
        return self.net(x)

# Genetic Algorithm Utils
def get_flat_params(model):
    return torch.cat([p.data.flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    # Ensure flat_params is a torch.Tensor
    if isinstance(flat_params, np.ndarray):
        flat_params = torch.from_numpy(flat_params).float()
        
    idx = 0
    for p in model.parameters():
        size = p.numel()
        p.data.copy_(flat_params[idx:idx + size].view_as(p))
        idx += size

def mutate(params, sigma, prob=1.0):
    mask = (torch.rand_like(params) < prob).float()
    noise = sigma * torch.randn_like(params)
    return params + noise * mask

# Fitness Evaluation
def evaluate(env, model, episodes):
    total_reward = 0.0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = model(obs_tensor).numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

    return float(total_reward / episodes)

_worker_env = None
_worker_model = None

def worker_init(cfg, flow, shared, obs_dim, act_dim): 
    global _worker_env
    global _worker_model

    if shared:
        vel = attach_shared_velocity_field(flow)
    else:
        vel = flow
    _worker_env = ZermeloEnv(cfg, vel)
    _worker_model = PolicyNet(obs_dim, act_dim)

def evaluate_genome(genome, episodes):
    global _worker_env
    global _worker_model
    
    set_flat_params(_worker_model, genome)
    return evaluate(_worker_env, _worker_model, episodes)

def evaluate_population(executor, population, episodes):
    fitness = list(
        executor.map(
            evaluate_genome,
            population,
            [episodes] * len(population),
        )
    )
    return np.array(fitness, dtype=np.float32)

def train_ga(cfg, flow, shared=True, generations=200, population_size=320, elite_frac=0.2, immigrant_frac=0.2, start_episodes=2, n_workers=8):
    if n_workers is None:
        n_workers = os.cpu_count()

    # only for determining dimensions
    if shared:
        env = ZermeloEnv(cfg, attach_shared_velocity_field(flow))
    else:
        env = ZermeloEnv(cfg, flow)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = PolicyNet(obs_dim, act_dim)
    param_size = len(get_flat_params(model))
    elite_size = int(population_size * elite_frac)
    immigrant_size = int(population_size * immigrant_frac)
    parent_size = elite_size // 2

    rank_weights = torch.arange(parent_size, 0, -1, dtype=torch.float32)
    selection_probs = rank_weights / rank_weights.sum()
       
    population = torch.zeros(population_size, param_size)
 
    for i in range(population_size):
        model = PolicyNet(obs_dim, act_dim)
        population[i] = get_flat_params(model)
 
    reward_history = []
    mean_reward_history = []
   
    with ProcessPoolExecutor(max_workers=n_workers, initializer=worker_init, initargs=(cfg, flow, shared, obs_dim, act_dim)) as executor: 
        
        for gen in range(generations): 

            episodes = min(start_episodes*5, int(start_episodes*(1.02**gen)))
                
            fitness = evaluate_population(executor, population, episodes)
            elite_idx = torch.topk(torch.from_numpy(fitness), elite_size).indices
            elites = population[elite_idx]

            elite_fitness_1 = fitness[elite_idx.numpy()]
            elite_fitness_2 = evaluate_population(executor, elites, episodes * 2)
            combined_fitness = elite_fitness_1 * (1/3) + elite_fitness_2 * (2/3)

            current_best = combined_fitness.max()
            current_mean = combined_fitness.mean()
            reward_history.append(current_best)
            mean_reward_history.append(current_mean)
            plot_rewards(reward_history, mean_reward_history)

            parent_idx = torch.topk(torch.from_numpy(combined_fitness), parent_size).indices
            parents = elites[parent_idx]

            # annealing mutations
            mutation_sigma = max(0.05, 0.5 * (0.99 ** gen))
            mutation_prob = 0.1 #max(0.05, 0.5 * (0.985 ** gen))

            # Create next generation
            new_population = torch.empty_like(population)
            new_population[:parent_size] = parents

            # Generate offspring
            idx = parent_size
            while idx < population_size - immigrant_size:
                p = torch.multinomial(selection_probs, 1).item()
                #p = torch.randint(parent_size, (1,)).item()
                new_population[idx] = mutate(parents[p], mutation_sigma, mutation_prob)
                idx += 1

            # Add immigrants
            for idx in range(population_size - immigrant_size, population_size):
                p = torch.randint(parent_size, (1,)).item()
                new_population[idx] = mutate(parents[p], 0.5, 0.3)
                #random_model = PolicyNet(obs_dim, act_dim)
                #new_population[idx] = get_flat_params(random_model)

            population = new_population

    # best genome
    set_flat_params(model, parents[0])

    return model, reward_history, mean_reward_history

def evaluate_policy(model, env, n_eval=5000):
    model.eval()
    results = []

    for episode in range(n_eval):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action = model(obs_tensor).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # определяем, достигнута ли цель
        x, y = env._agent_location
        success = ((x - env.xB) ** 2 + (y - env.yB) ** 2 <= env.rB ** 2)

        results.append(
            {
                "reward": total_reward,
                "time": env.time,
                "trajectory": np.array(env.trajectory),
                "success": success,
            }
        )

    return results