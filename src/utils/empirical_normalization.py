import jax
import jax.numpy as jnp


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
        M2=jnp.zeros(shape, jnp.float32),
        std=jnp.ones(shape, jnp.float32),
    )

@jax.jit
def update(normalizer: dict, x: jnp.ndarray) -> dict:
    count = normalizer['count']
    eps = normalizer['eps']
    until = normalizer['until']
    mean = normalizer['mean']
    M2 = normalizer['M2']
    std = normalizer['std']

    # compute new stats
    batch_n = x.shape[0]
    total_n = count + batch_n
    batch_mean = jnp.mean(x, axis=0)
    delta = batch_mean - mean
    new_mean = mean + delta * (batch_n / total_n)
    batch_var = jnp.mean((x - batch_mean) ** 2, axis=0)
    delta2 = batch_mean - new_mean
    M2_total = M2 + batch_var * batch_n + delta * delta2 * count
    new_M2 = M2_total
    new_std = jnp.sqrt(new_M2 / jnp.maximum(total_n - 1, 1))
    new_count = total_n

    # cond: skip update if until>=0 and count>=until
    cond = (until >= 0) & (count >= until)
    return jax.lax.cond(
        cond,
        lambda _: normalizer,
        lambda _: dict(
            eps=eps,
            until=until,
            count=new_count,
            mean=new_mean,
            M2=new_M2,
            std=new_std,
        ),
        operand=None,
    )

@jax.jit
def normalize(normalizer: dict, x: jnp.ndarray, center: bool = True) -> jnp.ndarray:
    mean = normalizer['mean']
    std = normalizer['std']
    eps = normalizer['eps']
    denom = std + eps
    return jnp.where(center, (x - mean) / denom, x / denom)
