import os

from hyperparams import BaseArgs

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import jax
import jax.numpy as jnp
from optax import adam, apply_updates, linear_schedule
import gymnasium as gym

from fast_td3 import Actor, Critic
from utils.replay_buffer import ReplayBuffer
from utils.reward_normalizer import RewardNormalizer
from utils.empirical_normalization import EmpiricalNormalization

prng = jax.random.PRNGKey(0)

def main():
    args = BaseArgs()
    env = gym.make("Pendulum-v1")

    obs, info = env.reset()
    obs_dim = obs.shape[0]
    assert env.action_space.shape is not None, "Action space must have a shape"
    act_dim = env.action_space.shape[0]

    rb = ReplayBuffer.create(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=obs_dim,
        n_act=act_dim,
        playground_mode=False,
    )

    normalizer = EmpiricalNormalization(obs_dim)


    obs = jnp.array([obs])
    obs = normalizer(obs)

    action = env.action_space.sample()

    next_obs, rewards, dones, truncations, _ = env.step(action)


    rb = rb.extend({
        'observations': obs,
        'actions': action,
        'next': {
            'observations': next_obs,
            'rewards': rewards,
            'dones': dones,
            'truncations': truncations,
        }
    })

    print(f'Stored obs: {obs}')
    print(f'Stored action: {action}')
    print(f'Stored next_obs: {next_obs}')
    print(f'Stored rewards: {rewards}')

    batch_size = args.batch_size // args.num_envs

    print(f'batch_size: {batch_size}')
    print(f'buffer ptr: {rb.ptr}')
    print(f'buffer size: {rb.buffer["buffer_size"]}')

    data = rb.sample(prng, batch_size)

    print(f'sampled data shape: {data["observations"].shape}')
    print(f'first obs: {data["observations"][0]}')
    print(f'first action: {data["actions"][0]}')

if __name__ == "__main__":
    main()