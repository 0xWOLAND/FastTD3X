import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import replace

from empirical_normalization import EmpiricalNormalization

@jax.jit
def _update(
    gamma: float,
    G: jnp.ndarray,
    G_r_max: jnp.ndarray,
    G_rms: EmpiricalNormalization,
    rewards: jnp.ndarray,
    dones: jnp.ndarray
):
    """Compute new discounted returns and update stats."""
    G_new = gamma * (1 - dones) * G + rewards
    G_rms_new = G_rms.update(G_new[..., None])
    G_r_max_new = jnp.maximum(G_r_max, jnp.max(jnp.abs(G_new)))
    return G_new, G_r_max_new, G_rms_new

@jax.jit
def _scale(
    std: jnp.ndarray,
    G_r_max: jnp.ndarray,
    g_max: float,
    epsilon: float,
    rewards: jnp.ndarray
) -> jnp.ndarray:
    """Normalize rewards by variance and clipped max."""
    denom = jnp.maximum(std + epsilon, G_r_max / g_max)
    return rewards / denom

@struct.dataclass
class RewardNormalizer:
    """Online reward normalization."""
    gamma: float
    g_max: float
    epsilon: float
    G: jnp.ndarray
    G_r_max: jnp.ndarray
    G_rms: EmpiricalNormalization

    @classmethod
    def create(
        cls,
        gamma: float,
        shape: tuple,
        g_max: float = 10.0,
        epsilon: float = 1e-8
    ) -> "RewardNormalizer":
        G = jnp.zeros(shape)
        G_r_max = jnp.zeros(shape)
        G_rms = EmpiricalNormalization.create(shape)
        return cls(gamma, g_max, epsilon, G, G_r_max, G_rms)

    def apply(
        self,
        rewards: jnp.ndarray,
        dones: jnp.ndarray
    ) -> tuple["RewardNormalizer", jnp.ndarray]:
        G_new, G_r_max_new, G_rms_new = _update(
            self.gamma, self.G, self.G_r_max, self.G_rms, rewards, dones
        )
        normed = _scale(
            G_rms_new.std, G_r_max_new, self.g_max, self.epsilon, rewards
        )
        new_state = replace(
            self,
            G=G_new,
            G_r_max=G_r_max_new,
            G_rms=G_rms_new
        )
        return new_state, normed
