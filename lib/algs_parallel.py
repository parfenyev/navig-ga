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

import cma  # Ensure you run: pip install pycma
import math

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
        #outputs = self.net(x)
        #return torch.atan2(outputs[..., 0], outputs[..., 1])

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

def mutate(params, sigma):
    #return params + sigma * torch.randn_like(params)
    return params + sigma * torch.randn_like(params) * (params.abs() + 0.1)

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

def train_ga(cfg, flow, shared=True,  generations=150, population_size=300, elite_frac=0.15, immigrant_frac=0.1, episodes=4, n_workers=8):
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
    
    rank_weights = torch.arange(elite_size, 0, -1, dtype=torch.float32)
    #rank_weights = torch.pow(rank_weights, 2.0)
    selection_probs = rank_weights / rank_weights.sum()
    
    population = torch.zeros(population_size, param_size)
 
    for i in range(population_size):
        model = PolicyNet(obs_dim, act_dim)
        population[i] = get_flat_params(model)
 
    reward_history = []
    mean_reward_history = []
   
    with ProcessPoolExecutor(max_workers=n_workers, initializer=worker_init, initargs=(cfg, flow, shared, obs_dim, act_dim)) as executor: 
        
        for gen in range(generations): 
            if gen==60:
                episodes *= 5
                
            fitness = evaluate_population(executor, population, episodes)
            elite_idx = torch.topk(torch.from_numpy(fitness), elite_size).indices
            elites = population[elite_idx]   

            current_best = fitness.max().item()
            current_mean = fitness[elite_idx.numpy()].mean()
            reward_history.append(current_best)
            mean_reward_history.append(current_mean)
            plot_rewards(reward_history, mean_reward_history)

            # annealing mutations
            mutation_sigma = max(0.1, 1.0 * (0.985 ** gen))

            # Create next generation
            new_population = elites.clone()

            while len(new_population) < population_size - immigrant_size:
                #i = random.randrange(elite_size)
                i = torch.multinomial(selection_probs, num_samples=1).item()
                child = elites[i].clone()
                child = mutate(child, mutation_sigma)
                new_population = torch.vstack([new_population, child.unsqueeze(0)])

            for _ in range(immigrant_size):
                random_model = PolicyNet(obs_dim, act_dim)
                genome = get_flat_params(random_model)
                new_population = torch.vstack([new_population, genome.unsqueeze(0)])

            population = new_population

    best_genome = elites[fitness[elite_idx].argmax()]
    set_flat_params(model, best_genome)

    return model, reward_history, mean_reward_history

def train_cma_es(cfg, flow, shared, init_model, generations=50, episodes=50, n_workers=8):
    if n_workers is None:
        n_workers = os.cpu_count()

    # Only for determining dimensions and initial parameters
    if shared:
        env = ZermeloEnv(cfg, attach_shared_velocity_field(flow))
    else:
        env = ZermeloEnv(cfg, flow)
        
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Grab parameter sizes and initial values
    initial_params = get_flat_params(init_model).detach().cpu().numpy()
    sigma = float(np.std(initial_params) * 0.005)

    # Initialize CMA-ES optimizer
    opts = {
        'CMA_diagonal': False,  
    } 
    es = cma.CMAEvolutionStrategy(initial_params, sigma0=sigma, inopts=opts)
 
    reward_history = []
    mean_reward_history = []
   
    with ProcessPoolExecutor(max_workers=n_workers, initializer=worker_init, initargs=(cfg, flow, shared, obs_dim, act_dim)) as executor: 
        
        for gen in range(generations):
            if es.stop():
                print("CMA-ES convergence criteria reached early.")
                break
                
            # 1. Ask CMA-ES for a generation of candidate weights (list of numpy arrays)
            population = es.ask()
            
            # 2. Parallel Evaluation
            rewards = evaluate_population(executor, population, episodes)
            
            # 3. CMA-ES minimizes, so convert max rewards to a loss/cost function (negation)
            costs = -rewards
            
            # 4. Update the internal covariance matrix and mean vector
            es.tell(population, costs)

            # Logging & Tracking performance
            current_best = rewards.max().item()
            current_mean = rewards.mean().item()
            reward_history.append(current_best)
            mean_reward_history.append(current_mean)
            plot_rewards(reward_history, mean_reward_history)

    # Extract the absolute best parameter set found over the entire run
    best_genome = es.result.xbest
    
    # Apply to a final model instance
    final_model = PolicyNet(obs_dim, act_dim)
    set_flat_params(final_model, best_genome)

    return final_model, reward_history, mean_reward_history

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