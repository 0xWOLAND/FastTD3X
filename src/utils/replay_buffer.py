import jax.numpy as jnp
from jax import random, jit
from flax import struct
from dataclasses import replace

@jit
def _extend(buffer, data):
    i = buffer.ptr % buffer.buffer_size
    buf = replace(buffer, ptr=buffer.ptr + 1)
    buf = replace(
        buf,
        observations=buf.observations.at[:, i].set(data['observations']),
        actions=buf.actions.at[:, i].set(data['actions']),
        rewards=buf.rewards.at[:, i].set(data['next']['rewards']),
        dones=buf.dones.at[:, i].set(data['next']['dones']),
        truncations=buf.truncations.at[:, i].set(data['next']['truncations']),
        next_observations=buf.next_observations.at[:, i].set(data['next']['observations'])
    )

    p = data['critic_observations'][..., buffer.n_obs:]
    q = data['next']['critic_observations'][..., buffer.n_obs:]
    buf = replace(
        buf,
        privileged_observations=buf.privileged_observations.at[:, i].set(p),
        next_privileged_observations=buf.next_privileged_observations.at[:, i].set(q),
        critic_observations=buf.critic_observations.at[:, i].set(data['critic_observations']),
        next_critic_observations=buf.next_critic_observations.at[:, i].set(data['next']['critic_observations'])
    )
    return buf


@jit
def _sample(buffer, key, batch_size):
    def g(X):
        return jnp.take(X, idx, axis=1).reshape(-1, *X.shape[2:])

    m = jnp.minimum(buffer.ptr, buffer.buffer_size)
    key, sub = random.split(key)
    idx = random.randint(sub, (buffer.n_env, batch_size), 0, m)
    obs = g(buffer.observations)
    act = g(buffer.actions)
    nxt = g(buffer.next_observations)
    r = g(buffer.rewards[..., None]).squeeze(-1)
    d = g(buffer.dones[..., None]).squeeze(-1)
    t = g(buffer.truncations[..., None]).squeeze(-1)

    out = {
        'observations': obs,
        'actions': act,
        'next': {
            'observations': nxt,
            'rewards': r,
            'dones': d,
            'truncations': t,
            'effective_n_steps': jnp.ones_like(r)
        }
    }
    p = g(buffer.privileged_observations)
    q = g(buffer.next_privileged_observations)
    c = g(buffer.critic_observations)
    nc = g(buffer.next_critic_observations)
    out['critic_observations'] = jnp.where(
        buffer.playground_mode, jnp.concatenate([obs, p], -1), c
    )
    out['next']['critic_observations'] = jnp.where(
        buffer.playground_mode, jnp.concatenate([nxt, q], -1), nc
    )
    return out

@struct.dataclass
class ReplayBuffer:
    n_env: int
    buffer_size: int
    n_obs: int
    n_act: int
    n_critic_obs: int
    asymmetric_obs: bool
    playground_mode: bool
    ptr: jnp.ndarray = struct.field(pytree_node=False)
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    truncations: jnp.ndarray
    next_observations: jnp.ndarray
    privileged_observations: jnp.ndarray
    next_privileged_observations: jnp.ndarray
    critic_observations: jnp.ndarray
    next_critic_observations: jnp.ndarray

    @classmethod
    def create(cls, n_env, buffer_size, n_obs, n_act, n_critic_obs,
               asymmetric_obs=False, playground_mode=False):
        def Z(shape):
            return jnp.zeros((n_env, buffer_size) + shape)

        pm = asymmetric_obs and playground_mode
        return cls(
            n_env, buffer_size, n_obs, n_act, n_critic_obs, asymmetric_obs, pm,
            jnp.array(0, jnp.int32),
            Z((n_obs,)), Z((n_act,)), Z(()), Z(()), Z(()), Z((n_obs,)),
            Z((n_critic_obs - n_obs,)), Z((n_critic_obs - n_obs,)),
            Z((n_critic_obs,)), Z((n_critic_obs,))
        )