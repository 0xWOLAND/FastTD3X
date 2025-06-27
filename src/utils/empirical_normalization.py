import jax
import jax.numpy as jnp
from jax import jit, lax
from flax import struct
from dataclasses import replace
from typing import Optional, Tuple

@jit
def _update_normalizer(count, mean, var, std, until, eps, x):
    """Update running mean, variance, and std with batch x."""
    def _update():
        batch_size = x.shape[0]
        new_count = count + batch_size
        batch_mean = jnp.mean(x, axis=0)
        delta = batch_mean - mean
        new_mean = mean + delta * (batch_size / new_count)

        batch_var = jnp.mean((x - batch_mean) ** 2, axis=0)
        delta2 = batch_mean - new_mean
        m_a = var * count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta2**2 * (count * batch_size / new_count)
        new_var = M2 / new_count
        new_std = jnp.sqrt(new_var)

        return new_count, new_mean, new_var, new_std

    def _identity():
        return count, mean, var, std

    cond = (until is not None) & (count >= until)
    return lax.cond(cond, _identity, _update)

@jit
def _normalize(mean, std, eps, x, center):
    """Normalize x given current mean and std."""
    denom = std + eps
    return (x - mean) / denom if center else x / denom

@struct.dataclass
class EmpiricalNormalization:
    """Online normalization with empirical mean and variance."""
    eps: float
    until: Optional[int]
    count: jnp.ndarray = struct.field(pytree_node=False)
    mean: jnp.ndarray
    var: jnp.ndarray
    std: jnp.ndarray

    @classmethod
    def create(cls, shape: Tuple[int, ...], eps: float = 1e-8, until: Optional[int] = None) -> "EmpiricalNormalization":
        mean = jnp.zeros(shape)
        var = jnp.ones(shape)
        std = jnp.ones(shape)
        count = jnp.array(0, dtype=jnp.int32)
        return cls(eps, until, count, mean, var, std)

    def update(self, x: jnp.ndarray) -> "EmpiricalNormalization":
        """Update running mean, variance, and std with batch x."""
        new_count, new_mean, new_var, new_std = _update_normalizer(
            self.count, self.mean, self.var, self.std, self.until, self.eps, x
        )
        return replace(
            self,
            count=new_count,
            mean=new_mean,
            var=new_var,
            std=new_std
        )

    def normalize(self, x: jnp.ndarray, center: bool = True) -> jnp.ndarray:
        """Normalize x given current mean and std."""
        return _normalize(self.mean, self.std, self.eps, x, center)
