import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

    
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
        net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim // 2),
            nn.relu,
            nn.Dense(self.hidden_dim // 4),
            nn.relu,
        ])
        x = net(x)
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
    def __call__(self, obs: jnp.ndarray, act: jnp.ndarray):
        return self.qnet1(obs, act), self.qnet2(obs, act)

    def projection(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        bootstrap: jnp.ndarray,
        discount: float,
    ):
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
    n_obs:      int
    n_act:      int
    n_envs:     int
    init_scale: float
    hidden_dim: int
    std_min:    float = 0.05
    std_max:    float = 0.8

    def setup(self):
        self.net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim // 2),
            nn.relu,
            nn.Dense(self.hidden_dim // 4),
            nn.relu,
        ])

        self.fc_mu = nn.Dense(
            self.n_act,
            kernel_init=nn.initializers.normal(stddev=self.init_scale),
            bias_init=nn.initializers.zeros,
        )

        init_scales = jax.random.uniform(
            self.make_rng('noise_init'),
            (self.n_envs, 1),
            minval=self.std_min,
            maxval=self.std_max
        )
        self.noise_scales = self.variable(
            'explore', 'noise_scales', lambda: init_scales
        )

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = self.net(obs)
        mu = jnp.tanh(self.fc_mu(x))
        return mu

    def explore(
        self,
        obs: jnp.ndarray,
        deterministic: bool = False,
        dones: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        mask = (
            dones.reshape(-1, 1)
            if dones is not None
            else jnp.zeros((self.n_envs, 1), dtype=bool)
        )

        rng = self.make_rng('explore')
        rng_new, rng_noise = jax.random.split(rng)

        new_scales = jax.random.uniform(
            rng_new, (self.n_envs, 1),
            minval=self.std_min,
            maxval=self.std_max
        )

        noise = jax.random.normal(rng_noise, (obs.shape[0], self.n_act))

        scales = jnp.where(mask, new_scales, self.noise_scales.value)
        self.noise_scales.value = scales

        mu = self(obs)

        return jnp.where(
            deterministic,
            mu,
            mu + noise * scales
        )
