import jax
import jax.numpy as jnp
from flax import linen as nn


def init(
    shape,
    eps: float = 1e-8,
    until: int = -1,
) -> dict:
    return dict(
        eps=eps,
        until=until,
        count=jnp.array(0, jnp.int32),
        mean=jnp.zeros(shape, jnp.float32),
        var=jnp.ones(shape, jnp.float32),
        std=jnp.ones(shape, jnp.float32),
    )

# @jax.jit
def update(normalizer: dict, x: jnp.ndarray) -> jnp.ndarray:
    count = normalizer['count']
    eps = normalizer['eps']
    until = normalizer['until']
    mean = normalizer['mean']
    var = normalizer['var']
    
    batch_size = x.shape[0]
    batch_mean = jnp.mean(x, axis=0, keepdims=True)
    new_count = count + batch_size
    
    delta = batch_mean - mean
    new_mean = mean + (batch_size / new_count) * delta
    
    batch_var = jnp.mean((x - batch_mean) ** 2, axis=0, keepdims=True)
    delta2 = batch_mean - new_mean
    
    m_a = var * count
    m_b = batch_var * batch_size
    M2 = m_a + m_b + (delta2**2) * (count * batch_size / new_count)
    
    new_var = M2 / new_count
    new_std = jnp.sqrt(new_var)
    
    should_update = ~((until >= 0) & (count >= until))

    return jax.lax.cond(
        should_update,
        lambda _: dict(
            eps=eps,
            until=until,
            count=new_count,
            mean=new_mean,
            var=new_var,
            std=new_std,
        ),
        lambda _: normalizer,
        operand=None
    )

@jax.jit
def normalize(normalizer: dict, x: jnp.ndarray, center: bool = True) -> jnp.ndarray:
    mean = normalizer['mean']
    std = normalizer['std']
    eps = normalizer['eps']
    denom = std + eps

    return jnp.where(center, (x - mean) / denom, x / denom)

class EmpiricalNormalize(nn.Module):
    def __init__(self, shape, eps: float = 1e-8, until: int = -1):
        super().__init__()
        self.state = init(shape, eps, until)

    def __call__(self, x: jnp.ndarray, center: bool = True) -> jnp.ndarray:
        return normalize(self.state, x, center)
    
    def update(self, x: jnp.ndarray) -> jnp.ndarray:
        return update(self.state, x)