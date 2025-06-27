import jax
import jax.numpy as jnp


def init_buffer(
    n_env: int,
    buffer_size: int,
    n_obs: int,
    n_act: int,
    n_critic_obs: int,
    playground_mode: bool = False,
    gamma: float = 0.99,
) -> dict:
    return {
        'n_env': n_env,
        'buffer_size': buffer_size,
        'n_obs': n_obs,
        'n_act': n_act,
        'n_critic_obs': n_critic_obs,
        'playground_mode': playground_mode,
        'gamma': gamma,
        'observations': jnp.zeros((n_env, buffer_size, n_obs), jnp.float32),
        'actions': jnp.zeros((n_env, buffer_size, n_act), jnp.float32),
        'rewards': jnp.zeros((n_env, buffer_size), jnp.float32),
        'dones': jnp.zeros((n_env, buffer_size), jnp.int32),
        'truncations': jnp.zeros((n_env, buffer_size), jnp.int32),
        'next_observations': jnp.zeros((n_env, buffer_size, n_obs), jnp.float32),
        'critic_observations': jnp.zeros((n_env, buffer_size, n_critic_obs), jnp.float32),
        'next_critic_observations': jnp.zeros((n_env, buffer_size, n_critic_obs), jnp.float32),
        'privileged_observations': jnp.zeros((n_env, buffer_size, n_critic_obs - n_obs), jnp.float32),
        'next_privileged_observations': jnp.zeros((n_env, buffer_size, n_critic_obs - n_obs), jnp.float32),
        'ptr': jnp.array(0, jnp.int32),
    }


@jax.jit
def extend(buffer: dict, data: dict) -> dict:
    """
    Functional update: insert one batch of transitions into the circular buffer.
    """
    idx = buffer['ptr'] % buffer['buffer_size']
    # base updates
    obs = buffer['observations'].at[:, idx].set(data['observations'])
    acts = buffer['actions'].at[:, idx].set(data['actions'])
    rwds = buffer['rewards'].at[:, idx].set(data['next']['rewards'])
    dns = buffer['dones'].at[:, idx].set(data['next']['dones'])
    trc = buffer['truncations'].at[:, idx].set(data['next']['truncations'])
    next_obs = buffer['next_observations'].at[:, idx].set(data['next']['observations'])
    # critic updates
    full_crit = buffer['critic_observations'].at[:, idx].set(data['critic_observations'])
    full_next_crit = buffer['next_critic_observations'].at[:, idx].set(data['next']['critic_observations'])
    priv = data['critic_observations'][..., buffer['n_obs']:]
    next_priv = data['next']['critic_observations'][..., buffer['n_obs']:]
    priv_upd = buffer['privileged_observations'].at[:, idx].set(priv)
    next_priv_upd = buffer['next_privileged_observations'].at[:, idx].set(next_priv)
    # select mode
    crit_obs = jax.lax.select(buffer['playground_mode'], priv_upd, full_crit)
    next_crit_obs = jax.lax.select(buffer['playground_mode'], next_priv_upd, full_next_crit)
    return {
        **buffer,
        'observations': obs,
        'actions': acts,
        'rewards': rwds,
        'dones': dns,
        'truncations': trc,
        'next_observations': next_obs,
        'critic_observations': crit_obs,
        'next_critic_observations': next_crit_obs,
        'privileged_observations': priv_upd,
        'next_privileged_observations': next_priv_upd,
        'ptr': buffer['ptr'] + 1,
    }


@jax.jit
def sample(buffer: dict, batch_size: int) -> dict:
    size = jnp.minimum(buffer['ptr'], buffer['buffer_size'])
    idx = jax.random.randint(0, (buffer['n_env'], batch_size), 0, size) # TODO: use key for randomness
    # expand idx for 3D gathers
    idx3 = idx[..., None]
    
    def gather(arr: jnp.ndarray) -> jnp.ndarray:
        # arr shape [n_env, buffer_size, D]
        D = arr.shape[-1]
        idx_expanded = idx3.repeat(D, axis=-1)
        return jnp.take_along_axis(arr, idx_expanded, axis=1).reshape(-1, D)

    obs = gather(buffer['observations'])
    acts = gather(buffer['actions'])
    next_obs = gather(buffer['next_observations'])
    rwds = gather(buffer['rewards'][..., None]).squeeze(-1)
    dns = gather(buffer['dones'][..., None]).squeeze(-1)
    trc = gather(buffer['truncations'][..., None]).squeeze(-1)
    full_crit = gather(buffer['critic_observations'])
    next_full_crit = gather(buffer['next_critic_observations'])
    priv = gather(buffer['privileged_observations'])
    next_priv = gather(buffer['next_privileged_observations'])
    # select mode
    crit = jax.lax.select(buffer['playground_mode'], jnp.concatenate([obs, priv], -1), full_crit)
    next_crit = jax.lax.select(buffer['playground_mode'], jnp.concatenate([next_obs, next_priv], -1), next_full_crit)
    return {
        'observations': obs,
        'actions': acts,
        'critic_observations': crit,
        'next': {
            'observations': next_obs,
            'critic_observations': next_crit,
            'rewards': rwds,
            'dones': dns,
            'truncations': trc,
            'effective_n_steps': jnp.ones_like(rwds),
        }
    }


class ReplayBuffer:
    """A simple replay buffer for storing and sampling transitions."""
    
    @staticmethod
    def create(
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False
    ):
        """Create a new replay buffer."""
        buffer = init_buffer(
            n_env=n_env,
            buffer_size=buffer_size,
            n_obs=n_obs,
            n_act=n_act,
            n_critic_obs=n_critic_obs,
            playground_mode=playground_mode
        )
        return ReplayBuffer(buffer)
    
    def __init__(self, buffer: dict):
        self.buffer = buffer
    
    @property
    def ptr(self):
        return self.buffer['ptr']
    
    def extend(self, data: dict):
        """Add transitions to the buffer."""
        new_buffer = extend(self.buffer, data)
        return ReplayBuffer(new_buffer)
    
    def sample(self, key: jax.Array, batch_size: int):
        """Sample a batch of transitions."""
        return sample(self.buffer, batch_size)
