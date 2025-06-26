import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Optional, Tuple
class DistributionalQNetwork(nn.Module):
    n_obs: int
    n_act: int
    num_atoms: int
    v_min: float
    v_max: float
    hidden_dim: int

    @nn.compact
    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, act], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 4)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_atoms)(x)
        return x
    
    def projection(
        self,
        obs: jax.Array,           
        actions: jax.Array,        
        rewards: jax.Array,        
        bootstrap: jax.Array,      
        discount: float,           
        q_support: jax.Array,      
    ):
        B = rewards.shape[0]
        N = self.num_atoms
        delta_z = (self.v_max - self.v_min) / (N - 1)

        z = rewards[:, None] + (bootstrap * discount)[:, None] * q_support[None, :]
        z = jnp.clip(z, self.v_min, self.v_max)

        b_idx = (z - self.v_min) / delta_z                
        l = jnp.floor(b_idx).astype(jnp.int32)             
        u = jnp.ceil(b_idx).astype(jnp.int32)              

        l = jnp.where((u > 0) & (l == u), l - 1, l)
        u = jnp.where((l < N - 1) & (l == u), u + 1, u)

        logits = self(obs, actions)
        prob = jax.nn.softmax(logits, axis=-1)

        w_l = prob * (u.astype(jnp.float32) - b_idx)
        w_u = prob * (b_idx - l.astype(jnp.float32))

        batch_idx = jnp.arange(B)[:, None].repeat(N, axis=1)

        proj = jnp.zeros((B, N), dtype=jnp.float32)
        proj = proj.at[batch_idx, l].add(w_l)
        proj = proj.at[batch_idx, u].add(w_u)

        return proj

class Critic(nn.Module):
    n_obs:      int
    n_act:      int
    num_atoms:  int
    v_min:      float
    v_max:      float
    hidden_dim: int

    def setup(self):
        self.qnet1 = DistributionalQNetwork(
            self.n_obs, self.n_act, self.num_atoms,
            self.v_min, self.v_max, self.hidden_dim
        )
        self.qnet2 = DistributionalQNetwork(
            self.n_obs, self.n_act, self.num_atoms,
            self.v_min, self.v_max, self.hidden_dim
        )
        self.q_support = jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    @nn.compact
    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return self.qnet1(obs, act), self.qnet2(obs, act)

    def projection(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        bootstrap: jnp.ndarray,
        discount: float,
    ) -> Tuple[jax.Array, jax.Array]:
        q1_proj = self.qnet1.projection(
            obs, actions, rewards, bootstrap, discount, self.q_support
        )
        q2_proj = self.qnet2.projection(
            obs, actions, rewards, bootstrap, discount, self.q_support
        )
        return q1_proj, q2_proj

    def get(self, atoms: jnp.ndarray) -> jnp.ndarray:
        return (atoms * self.q_support).sum(axis=-1)

class Actor(nn.Module):
    n_obs: int
    n_act: int
    n_envs: int
    init_scale: float
    hidden_dim: int
    std_min: float
    std_max: float

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 4)(x)
        x = nn.relu(x)

        mu = nn.Dense(
            self.n_act,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=nn.initializers.zeros
        )(x)
        return jnp.tanh(mu)


    def explore(
        self,
        obs: jax.Array,
        rng: jax.Array,
        noise_scales: jax.Array,
        dones: Optional[jax.Array],
        deterministic: bool,
        std_min: float,
        std_max: float,
    ) -> jax.Array:
        rng_resample, rng_noise = jax.random.split(rng)

        dones_mask = (
            dones.reshape(-1, 1).astype(bool)
            if dones is not None else
            jnp.zeros_like(noise_scales, dtype=bool)
        )

        new_scales = jax.random.uniform(
            rng_resample, noise_scales.shape,
            minval=std_min, maxval=std_max,
        )
        scales = jnp.where(dones_mask, new_scales, noise_scales)

        mu = self(obs)
        noise = jax.random.normal(rng_noise, mu.shape)
        return jnp.where(deterministic, mu, mu + noise * scales)

class MultiTaskActor(Actor):
    num_tasks:      int
    task_embed_dim: int

    def setup(self):
        super().setup()
        self.task_embed = self.param(
            "task_embed",
            nn.initializers.orthogonal(),
            (self.num_tasks, self.task_embed_dim),
        )

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        core    = obs[:, :-self.num_tasks]
        one_hot = obs[:, -self.num_tasks:]
        task_emb = one_hot @ self.task_embed
        obs = jnp.concatenate([core, task_emb], axis=-1)

        return super().__call__(obs)

class MultiTaskCritic(Critic):
    num_tasks:      int
    task_embed_dim: int

    def setup(self):
        super().setup()

        self.task_embed = self.param(
            "task_embed",
            nn.initializers.orthogonal(),
            (self.num_tasks, self.task_embed_dim),
        )
    
    @nn.compact
    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        core    = obs[:, :-self.num_tasks]
        one_hot = obs[:, -self.num_tasks:]
        task_emb = one_hot @ self.task_embed
        obs = jnp.concatenate([core, task_emb], axis=-1)

        return super().__call__(obs, act)

    def projection(
        self,
        obs: jax.Array,
        actions: jax.Array,
        rewards: jax.Array,
        bootstrap: jax.Array,
        discount: float,
    ) -> Tuple[jax.Array, jax.Array]:
        core    = obs[..., :-self.num_tasks]
        one_hot = obs[..., -self.num_tasks:]
        task_emb = one_hot @ self.task_embed
        obs = jnp.concatenate([core, task_emb], axis=-1)

        return super().projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
        )
