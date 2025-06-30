import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial

def init_buffer(
    n_env: int,
    buffer_size: int,
    n_obs: int,
    n_act: int,
    playground_mode: bool = False,
    gamma: float = 0.99,
) -> dict:
    return {
        'n_env': n_env,
        'buffer_size': buffer_size,
        'n_obs': n_obs,
        'n_act': n_act,
        'playground_mode': playground_mode,
        'gamma': gamma,
        'observations': jnp.zeros((n_env, buffer_size, n_obs), jnp.float32),
        'actions': jnp.zeros((n_env, buffer_size, n_act), jnp.float32),
        'rewards': jnp.zeros((n_env, buffer_size), jnp.float32),
        'dones': jnp.zeros((n_env, buffer_size), jnp.int32),
        'truncations': jnp.zeros((n_env, buffer_size), jnp.int32),
        'next_observations': jnp.zeros((n_env, buffer_size, n_obs), jnp.float32),
        'ptr': jnp.array(0, jnp.int32),
    }


@jax.jit
def extend(buffer: dict, data: dict) -> dict:
    """Extend the replay buffer with new data.
    
    Args:
        buffer: Replay buffer dict containing the current buffer state
        data: Dict containing the new data to add:
            - observations: Array of shape [n_env, n_obs] - current observations
            - actions: Array of shape [n_env, n_act] - actions taken
            - next: Dict containing:
                - observations: Array of shape [n_env, n_obs] - next observations
                - rewards: Array of shape [n_env] - rewards received
                - dones: Array of shape [n_env] - done flags
                - truncations: Array of shape [n_env] - truncation flags
    
    Returns:
        dict: Updated replay buffer with new data added
    """
    idx = buffer['ptr'] % buffer['buffer_size']
    
    obs_data = data['observations']
    act_data = data['actions']
    
    next_obs_data = data['next']['observations']
    next_rewards_data = data['next']['rewards']
    next_dones_data = data['next']['dones']
    next_truncations_data = data['next']['truncations']
    
    obs = buffer['observations'].at[:, idx].set(obs_data)
    acts = buffer['actions'].at[:, idx].set(act_data)
    rwds = buffer['rewards'].at[:, idx].set(next_rewards_data)
    dns = buffer['dones'].at[:, idx].set(next_dones_data)
    trc = buffer['truncations'].at[:, idx].set(next_truncations_data)
    next_obs = buffer['next_observations'].at[:, idx].set(next_obs_data)
    
    
    return {
        **buffer,
        'observations': obs,
        'actions': acts,
        'rewards': rwds,
        'dones': dns,
        'truncations': trc,
        'next_observations': next_obs,
        'ptr': buffer['ptr'] + 1,
    }


@partial(jax.jit, static_argnums=(2,3))
# TODO: Replay buffer boundary edge case logic (e.g. rolling back truncation flags if full)
def sample(buffer: dict, key: jax.Array, n_env: int, batch_size: int) -> dict:
    size = jnp.minimum(buffer['ptr'], buffer['buffer_size'])
    idx = jax.random.randint(key, (n_env, batch_size), 0, size)
    # expand idx for 3D gathers
    idx3 = idx[..., None]
    
    def gather(arr: jnp.ndarray) -> jnp.ndarray:
        D = arr.shape[-1]
        idx_expanded = idx3.repeat(D, axis=-1)
        return jnp.take_along_axis(arr, idx_expanded, axis=1).reshape(-1, D)

    obs = gather(buffer['observations'])
    acts = gather(buffer['actions'])
    next_obs = gather(buffer['next_observations'])
    rwds = gather(buffer['rewards'][..., None]).squeeze(-1)
    dns = gather(buffer['dones'][..., None]).squeeze(-1)
    trc = gather(buffer['truncations'][..., None]).squeeze(-1)
    return {
        'observations': obs,
        'actions': acts,
        'next': {
            'observations': next_obs,
            'rewards': rwds,
            'dones': dns,
            'truncations': trc,
            'effective_n_steps': jnp.ones_like(rwds),
        }
    }
class ReplayBuffer(nn.Module):
    """A simple replay buffer for storing and sampling transitions."""
    
    @staticmethod
    def create(
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        playground_mode: bool = False
    ):
        """Create a new replay buffer."""
        buffer = init_buffer(
            n_env=n_env,
            buffer_size=buffer_size,
            n_obs=n_obs,
            n_act=n_act,
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
        return sample(self.buffer, key, int(self.buffer['n_env']), batch_size)
