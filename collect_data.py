import time
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import tyro
from mbd import envs as mbd_envs

@dataclass
class Args:
    env_name: str = "hopper"
    num_trajectories: int = 200
    rollout_horizon: int = 50
    save_path: str = "proposal_dataset.npz"

args = tyro.cli(Args)
last_print = time.time()

# Get environment and training setup
env = mbd_envs.get_env(args.env_name)

def progress_fn(num_steps, metrics):
    global last_print
    now = time.time()
    if now - last_print > 5:  # print every 5 seconds
        reward = metrics.get("eval/episode_reward", float('nan'))
        print(f"[{num_steps} steps] Eval reward: {reward:.2f}")
        last_print = now

# Choose training config based on env_name
def train_fn(environment, progress_fn=None):
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
        progress_fn=progress_fn,
    )

rng = jax.random.PRNGKey(seed=0)

# Train the policy
print("Start training")
start_time = time.time()
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress_fn)
print(f"Finished training in {time.time()-start_time:.1f}s")
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

print(f"Saved dataset to {args.save_path}")
print(f"Contexts shape: {contexts_np.shape}")
print(f"Trajectories shape: {trajectories_np.shape}")