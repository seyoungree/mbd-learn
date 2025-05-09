import functools
import os
import jax
from jax import numpy as jnp
from dataclasses import dataclass
import tyro
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import numpy as np

import mbd
import flax.linen as nn
import jax.numpy as jnp

class ConditionalProposalNet(nn.Module):
    hidden_dim: int = 512
    H: int = 50
    d: int = 3
    time_embed_dim: int = 16
    max_t: int = 50
    ctx_dim: int = 16

    @nn.compact
    def __call__(self, Yi, context, timesteps):
        B = Yi.shape[0]

        # Embed scalar timestep t_i -> (B, time_embed_dim)
        t_embed = nn.Embed(self.max_t + 1, self.time_embed_dim)(timesteps)
        t_embed = jnp.tile(t_embed[:, None, :], (1, self.H, 1))  # (B, H, time_embed_dim)

        # Expand context -> (B, H, ctx_dim)
        ctx_exp = jnp.tile(context[:, None, :], (1, self.H, 1))

        # Combine inputs: (B, H, d + time_embed_dim + ctx_dim)
        x = jnp.concatenate([Yi, t_embed, ctx_exp], axis=-1)
        x = x.reshape((B, -1))  # Flatten over time

        # Feedforward MLP
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.H * self.d)(x)

        return x.reshape((B, self.H, self.d))

    
@dataclass
class Args:
    seed: int = 0
    disable_recommended_params: bool = False
    not_render: bool = False
    env_name: str = "ant"
    Nsample: int = 2048
    Hsample: int = 50
    Ndiffuse: int = 100
    temp_sample: float = 0.1
    beta0: float = 1e-4
    betaT: float = 1e-2
    enable_demo: bool = False

def run_diffusion(args: Args):
    rng = jax.random.PRNGKey(seed=args.seed)

    temp_recommend = {
        "ant": 0.1,
        "halfcheetah": 0.4,
        "hopper": 0.1,
        "humanoidstandup": 0.1,
        "humanoidrun": 0.1,
        "walker2d": 0.1,
        "pushT": 0.2,
    }
    Ndiffuse_recommend = {"pushT": 200, "humanoidrun": 300}
    Nsample_recommend = {"humanoidrun": 8192}
    Hsample_recommend = {"pushT": 40}

    if not args.disable_recommended_params:
        args.temp_sample = temp_recommend.get(args.env_name, args.temp_sample)
        args.Ndiffuse = Ndiffuse_recommend.get(args.env_name, args.Ndiffuse)
        args.Nsample = Nsample_recommend.get(args.env_name, args.Nsample)
        args.Hsample = Hsample_recommend.get(args.env_name, args.Hsample)
        print(f"override temp_sample to {args.temp_sample}")

    env = mbd.envs.get_env(args.env_name)
    Nx = env.observation_size
    Nu = env.action_size

    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(functools.partial(mbd.utils.rollout_us, step_env_jit))

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)

    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)
    Sigmas_cond = (1 - alphas) * (1 - jnp.sqrt(jnp.roll(alphas_bar, 1))) / (1 - alphas_bar)
    sigmas_cond = jnp.sqrt(Sigmas_cond)
    sigmas_cond = sigmas_cond.at[0].set(0.0)
    print(f"init sigma = {sigmas[-1]:.2e}")

    YN = jnp.zeros([args.Hsample, Nu])

    # Load ConditionalProposalNet
    with open("traj_model_0.075422.pkl", "rb") as f:
        proposal_params = pickle.load(f)
    proposal_model = ConditionalProposalNet(H=args.Hsample, d=Nu, ctx_dim=12, max_t=args.Ndiffuse-1)
    proposal_apply_fn = proposal_model.apply

    dummy_context = state_init.obs
    
    @jax.jit
    def reverse_once(carry, unused):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        rng, noise_rng = jax.random.split(rng)

        # Proposal-based sampling
        Yi_input = Yi[None, :, :]  # (1, H, d)
        ctx_input = dummy_context[None, :]  # (1, ctx_dim)
        t_input = jnp.array([i], dtype=jnp.int32)  # (1,)

        # Single proposal prediction
        Y0_pred = proposal_apply_fn(proposal_params, Yi_input, ctx_input, t_input)  # (1, H, d)
        Y0s = jnp.repeat(Y0_pred, args.Nsample, axis=0)  # (Nsample, H, d)

        noise = jax.random.normal(noise_rng, Y0s.shape)
        Y0s = Y0s + sigmas[i] * noise
        Y0s = jnp.clip(Y0s, -1.0, 1.0)

        rewss, qs = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=-1)
        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        rew_mean = rews.mean()
        logp0 = (rews - rew_mean) / rew_std / args.temp_sample

        if args.enable_demo:
            xref_logpds = jax.vmap(env.eval_xref_logpd)(qs)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = (xref_logpds + env.rew_xref - rew_mean) / rew_std / args.temp_sample
            demo_mask = logpdemo > logp0
            logp0 = jnp.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("n,nij->ij", weights, Y0s)

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        return (i - 1, rng, Ybar_im1), rews.mean()

    def reverse(YN, rng):
        Yi = YN
        Ybars = []
        with tqdm(range(args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                carry_once = (i, rng, Yi)
                (i, rng, Yi), rew = reverse_once(carry_once, None)
                Ybars.append(Yi)
                pbar.set_postfix({"rew": f"{rew:.2e}"})
        return jnp.array(Ybars)

    rng_exp, rng = jax.random.split(rng)
    Yi = reverse(YN, rng_exp)

    if not args.not_render:
        path = f"{mbd.__path__[0]}/../results/{args.env_name}"
        os.makedirs(path, exist_ok=True)
        jnp.save(f"{path}/mu_0ts.npy", Yi)
        if args.env_name == "car2d":
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            xs = jnp.array([state_init.pipeline_state])
            state = state_init
            for t in range(Yi.shape[1]):
                state = step_env_jit(state, Yi[-1, t])
                xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
            env.render(ax, xs)
            if args.enable_demo:
                ax.plot(env.xref[:, 0], env.xref[:, 1], "g--", label="RRT path")
            ax.legend()
            plt.savefig(f"{path}/rollout.png")
        else:
            render_us = functools.partial(
                mbd.utils.render_us,
                step_env_jit,
                env.sys.tree_replace({"opt.timestep": env.dt}),
            )
            webpage = render_us(state_init, Yi[-1])
            with open(f"{path}/rollout.html", "w") as f:
                f.write(webpage)

    rewss_final, _ = rollout_us(state_init, Yi[-1])
    return rewss_final.mean()

if __name__ == "__main__":
    rew_final = run_diffusion(args=tyro.cli(Args))
    print(f"final reward = {rew_final:.2e}")
