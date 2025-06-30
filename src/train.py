import os

from hyperparams import BaseArgs

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
from flax.training import train_state

from fast_td3 import Actor, Critic
from utils.replay_buffer import ReplayBuffer
from utils.reward_normalizer import RewardNormalizer
from utils.empirical_normalization import EmpiricalNormalize

prng = jax.random.PRNGKey(0)
deterministic = False

def main():
    args = BaseArgs()
    env = gym.make("Pendulum-v1")

    obs, _ = env.reset()
    obs_dim = obs.shape[0]
    assert env.action_space.shape is not None, "Action space must have a shape"
    act_dim = env.action_space.shape[0]

    actor = Actor(
        n_obs=obs_dim,
        n_act=act_dim,
        n_envs=args.num_envs,
        init_scale=args.init_scale,
        hidden_dim=args.actor_hidden_dim,
        std_min=args.std_min,
        std_max=args.std_max,
    )

    critic = Critic(
        n_obs=obs_dim,
        n_act=act_dim,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=args.critic_hidden_dim,
    )

    normalize = EmpiricalNormalize(obs_dim)
    obs = normalize(jnp.array([obs]))

    actor_params = actor.init(prng, obs)
    critic_params = critic.init(prng, obs, jnp.zeros((1, act_dim)))

    actor_tx = optax.adamw(learning_rate=args.actor_learning_rate)
    critic_tx = optax.adamw(learning_rate=args.critic_learning_rate)

    noise_scales = jnp.ones((args.num_envs, 1)) * (args.std_min + args.std_max) / 2

    actor_state = train_state.TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=actor_tx,
    )

    critic_state = train_state.TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=critic_tx,
    )

    rb = ReplayBuffer.create(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=obs_dim,
        n_act=act_dim,
    )

    for step in range(1):
        # for step in range(args.total_timesteps):
        action, noise_scales = actor.explore(actor_params, obs, prng, noise_scales, deterministic=deterministic)
        next_obs, rewards, dones, truncations, infos = env.step(action)

        print("INFO: ", infos)
        # _next_obs = jnp.where(dones[:, None] > 0, next_obs, _next_obs)

        print("Step:  ")
        print({
            "obs": obs.shape,
            "action": action.shape,
            "next_obs": next_obs.shape,
            "rewards": rewards,
            "dones": dones,
            "truncations": truncations,
        })

    #     rb = rb.extend({
    #         'observations': obs,
    #         'actions': action,
    #         'next': {
    #             'observations': next_obs,
    #             'rewards': rewards,
    #             'dones': dones,
    #             'truncations': truncations,
    #         }
    #     })

    #     batch_size = args.batch_size // args.num_envs

    #     data = rb.sample(prng, batch_size)
    #     obs = normalize(data['observations'])
    #     next_obs = normalize(data['next']['observations'])

if __name__ == "__main__":
    main()