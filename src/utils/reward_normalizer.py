import jax
import jax.numpy as jnp
from flax import linen as nn


def init_buffer(
    n_env: int,
    buffer_size: int,
    n_obs: int,
    n_act: int,
    n_critic_obs: int,
    playground_mode: bool = False,
    gamma: float = 0.99,
) -> dict:
    return dict(
        n_env=n_env,
        buffer_size=buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        playground_mode=playground_mode,
        gamma=gamma,
        ptr=jnp.array(0, jnp.int32),
        observations=jnp.zeros((n_env, buffer_size, n_obs), jnp.float32),
        actions=jnp.zeros((n_env, buffer_size, n_act), jnp.float32),
        rewards=jnp.zeros((n_env, buffer_size), jnp.float32),
        dones=jnp.zeros((n_env, buffer_size), jnp.int32),
        truncations=jnp.zeros((n_env, buffer_size), jnp.int32),
        next_observations=jnp.zeros((n_env, buffer_size, n_obs), jnp.float32),
        critic_observations=jnp.zeros((n_env, buffer_size, n_critic_obs), jnp.float32),
        next_critic_observations=jnp.zeros((n_env, buffer_size, n_critic_obs), jnp.float32),
        privileged_observations=jnp.zeros((n_env, buffer_size, n_critic_obs - n_obs), jnp.float32),
        next_privileged_observations=jnp.zeros((n_env, buffer_size, n_critic_obs - n_obs), jnp.float32)
    )

@jax.jit
def extend(buffer: dict, data: dict) -> dict:
    idx = buffer['ptr'] % buffer['buffer_size']
    def update(arr, val):
        return arr.at[:, idx].set(val)
    updates = {
        'observations': update(buffer['observations'], data['observations']),
        'actions': update(buffer['actions'], data['actions']),
        'rewards': update(buffer['rewards'], data['next']['rewards']),
        'dones': update(buffer['dones'], data['next']['dones']),
        'truncations': update(buffer['truncations'], data['next']['truncations']),
        'next_observations': update(buffer['next_observations'], data['next']['observations']),
    }

    full = update(buffer['critic_observations'], data['critic_observations'])
    priv = update(buffer['privileged_observations'], data['critic_observations'][..., buffer['n_obs']:])
    updates['critic_observations'] = jax.lax.select(buffer['playground_mode'], priv, full)
    full_n = update(buffer['next_critic_observations'], data['next']['critic_observations'])
    priv_n = update(buffer['next_privileged_observations'], data['next']['critic_observations'][..., buffer['n_obs']:])
    updates['next_critic_observations'] = jax.lax.select(buffer['playground_mode'], priv_n, full_n)
    updates['privileged_observations'] = priv
    updates['next_privileged_observations'] = priv_n
    updates['ptr'] = buffer['ptr'] + 1
    return {**buffer, **updates}

@jax.jit
def sample(buffer: dict, batch_size: int) -> dict:
    size = jnp.minimum(buffer['ptr'], buffer['buffer_size'])
    idx = jax.random.randint(0, (buffer['n_env'], batch_size), 0, size) # TODO: use key for randomness
    idx3 = idx[..., None]
    # dimension-agnostic gather
    def gather(arr):
        if arr.ndim == 2:
            return jnp.take_along_axis(arr, idx, axis=1).reshape(-1)
        D = arr.shape[-1]
        ie = idx3.repeat(D, axis=-1)
        return jnp.take_along_axis(arr, ie, axis=1).reshape(-1, D)
    obs = gather(buffer['observations'])
    acts = gather(buffer['actions'])
    next_obs = gather(buffer['next_observations'])
    rwds = gather(buffer['rewards'])
    dns = gather(buffer['dones'])
    trc = gather(buffer['truncations'])
    full = gather(buffer['critic_observations'])
    priv = gather(buffer['privileged_observations'])
    crit = jax.lax.select(buffer['playground_mode'], jnp.concatenate([obs, priv], -1), full)
    next_full = gather(buffer['next_critic_observations'])
    next_priv = gather(buffer['next_privileged_observations'])
    next_crit = jax.lax.select(buffer['playground_mode'], jnp.concatenate([next_obs, next_priv], -1), next_full)
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
            'effective_n_steps': jnp.ones_like(rwds)
        }
    }

class RewardNormalizer(nn.Module):
    @staticmethod
    def create(gamma: float, shape: tuple, g_max: float = 10.0, epsilon: float = 1e-8):
        return RewardNormalizer(
            gamma=gamma,
            shape=shape,
            g_max=g_max,
            epsilon=epsilon,
            count=jnp.array(0, jnp.int32),
            mean=jnp.zeros(shape, jnp.float32),
            M2=jnp.zeros(shape, jnp.float32),
            std=jnp.ones(shape, jnp.float32),
            returns=jnp.zeros(shape, jnp.float32)
        )
    
    def __init__(self, gamma: float, shape: tuple, g_max: float, epsilon: float, 
                 count: jax.Array, mean: jax.Array, M2: jax.Array, std: jax.Array, returns: jax.Array):
        self.gamma = gamma
        self.shape = shape
        self.g_max = g_max
        self.epsilon = epsilon
        self.count = count
        self.mean = mean
        self.M2 = M2
        self.std = std
        self.returns = returns
    
    def apply(self, rewards: jax.Array, dones: jax.Array):
        """Apply reward normalization and update statistics."""
        # Update returns
        self.returns = self.returns * self.gamma + rewards
        
        # Update statistics
        batch_n = rewards.shape[0]
        total_n = self.count + batch_n
        batch_mean = jnp.mean(self.returns, axis=0)
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_n / total_n)
        batch_var = jnp.mean((self.returns - batch_mean) ** 2, axis=0)
        delta2 = batch_mean - new_mean
        M2_total = self.M2 + batch_var * batch_n + delta * delta2 * self.count
        new_M2 = M2_total
        new_std = jnp.sqrt(new_M2 / jnp.maximum(total_n - 1, 1))
        new_count = total_n
        
        # Create new normalizer with updated stats
        new_normalizer = RewardNormalizer(
            gamma=self.gamma,
            shape=self.shape,
            g_max=self.g_max,
            epsilon=self.epsilon,
            count=new_count,
            mean=new_mean,
            M2=new_M2,
            std=new_std,
            returns=jnp.where(dones, jnp.zeros_like(self.returns), self.returns)
        )
        
        # Normalize rewards
        normalized_rewards = jnp.clip(
            (rewards - new_mean) / (new_std + self.epsilon),
            -self.g_max, self.g_max
        )
        
        return new_normalizer, normalized_rewards
