import os
import functools
import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from dataclasses import dataclass
import tyro

from brax.io import html
from mbd import envs as mbd_envs

@dataclass
class Args:
    env_name: str = "halfcheetah"
    num_trajectories: int = 200
    rollout_horizon: int = 50
    save_path: str = "proposal_dataset.npz"

args = tyro.cli(Args)

# Get environment and training setup
env = mbd_envs.get_env(args.env_name)

# Choose training config based on env_name
def train_fn(environment):
    from brax.training.agents.sac.train import train as sac_train
    return sac_train(
        environment=environment,
        num_timesteps=1_000_000,
        num_evals=5,
        episode_length=1000,
        normalize_observations=True,
        reward_scaling=30,
        action_repeat=1,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=512,
        grad_updates_per_step=64,
        seed=1,
    )

rng = jax.random.PRNGKey(seed=0)

# Train the policy
print("Starting training")
make_inference_fn, params, _ = train_fn(environment=env)
print("Finished training")
inference_fn = make_inference_fn(params)
jit_inference = jax.jit(inference_fn)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Collect dataset
contexts = []
trajectories = []

for i in range(args.num_trajectories):
    rng, rollout_rng = jax.random.split(rng)
    state = jit_reset(rollout_rng)
    trajectory = []

    for t in range(args.rollout_horizon):
        rollout_rng, act_rng = jax.random.split(rollout_rng)
        action, _ = jit_inference(state.obs, act_rng)
        trajectory.append(action)
        state = jit_step(state, action)

    trajectory = jnp.stack(trajectory)  # [H, action_dim]
    context = state.obs  # or initial obs if you prefer
    contexts.append(np.array(context))
    trajectories.append(np.array(trajectory))

    if i % 10 == 0:
        print(f"Collected {i}/{args.num_trajectories} trajectories")

# Save dataset
contexts_np = np.stack(contexts)
trajectories_np = np.stack(trajectories)
np.savez(args.save_path, contexts=contexts_np, trajectories=trajectories_np)

print(f"✅ Saved dataset to {args.save_path}")
print(f"Contexts shape: {contexts_np.shape}")
print(f"Trajectories shape: {trajectories_np.shape}")
import os
import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from dataclasses import dataclass
import tyro

from brax.io import html
from mbd import envs as mbd_envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.training import inference as inference_lib

@dataclass
class Args:
    env_name: str = "halfcheetah"
    num_trajectories: int = 200
    rollout_horizon: int = 50
    save_path: str = "proposal_dataset.npz"

args = tyro.cli(Args)

# Get environment and training setup
env = mbd_envs.get_env(args.env_name)

# Choose training config based on env_name
train_fn = {
    "halfcheetah": functools.partial(
        ppo,
        num_timesteps=1_000_000,
        num_evals=5,
        episode_length=1000,
        normalize_observations=True,
        reward_scaling=1.0,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=0.001,
        num_envs=2048,
        batch_size=512,
        seed=0,
    ),
    "hopper": functools.partial(
        sac,
        num_timesteps=1_000_000,
        num_evals=5,
        episode_length=1000,
        normalize_observations=True,
        reward_scaling=30,
        action_repeat=1,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=512,
        grad_updates_per_step=64,
        seed=1,
    ),
}[args.env_name]

rng = jax.random.PRNGKey(seed=0)

# Train the policy
make_inference_fn, params, _ = train_fn(environment=env)
inference_fn = make_inference_fn(params)
jit_inference = jax.jit(inference_fn)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Collect dataset
contexts = []
trajectories = []

for i in range(args.num_trajectories):
    rng, rollout_rng = jax.random.split(rng)
    state = jit_reset(rollout_rng)
    trajectory = []

    for t in range(args.rollout_horizon):
        rollout_rng, act_rng = jax.random.split(rollout_rng)
        action, _ = jit_inference(state.obs, act_rng)
        trajectory.append(action)
        state = jit_step(state, action)

    trajectory = jnp.stack(trajectory)  # [H, action_dim]
    context = state.obs  # or initial obs if you prefer
    contexts.append(np.array(context))
    trajectories.append(np.array(trajectory))

    if i % 10 == 0:
        print(f"Collected {i}/{args.num_trajectories} trajectories")

# Save dataset
contexts_np = np.stack(contexts)
trajectories_np = np.stack(trajectories)
np.savez(args.save_path, contexts=contexts_np, trajectories=trajectories_np)

print(f"✅ Saved dataset to {args.save_path}")
print(f"Contexts shape: {contexts_np.shape}")
print(f"Trajectories shape: {trajectories_np.shape}")
