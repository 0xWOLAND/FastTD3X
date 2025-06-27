import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import jax
import jax.numpy as jnp
from optax import adam, apply_updates, linear_schedule
import gymnasium as gym

from fast_td3 import Actor, Critic
from hyperparams import BaseArgs, get_args

TOTAL_STEPS = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 10000

def main():
    args = get_args()
    print(args)
    
    # Environment setup
    env = gym.make("Pendulum-v1")    
    obs, info = env.reset()
    
    # Dimensions
    obs_dim = 3
    action_dim = 1
    hidden_dim = 256
    
    # Create networks
    actor = Actor(
        n_obs=obs_dim, 
        n_act=action_dim, 
        n_envs=1, 
        init_scale=args.init_scale, 
        hidden_dim=hidden_dim, 
        std_min=args.std_min, 
        std_max=args.std_max
    )
    
    critic = Critic(
        n_obs=obs_dim,
        n_act=action_dim,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=hidden_dim
    )
    
    # Initialize networks
    rng = jax.random.key(0)
    obs_batch = jnp.expand_dims(obs, axis=0)
    action_batch = jnp.expand_dims(jnp.zeros(action_dim), axis=0)
    
    actor_params = actor.init(rng, obs_batch)
    critic_params = critic.init(rng, obs_batch, action_batch)
    target_critic_params = critic_params
    
    # Create optimizers with schedulers
    actor_lr_schedule = linear_schedule(
        init_value=args.actor_learning_rate,
        end_value=args.actor_learning_rate_end,
        transition_steps=TOTAL_STEPS
    )
    critic_lr_schedule = linear_schedule(
        init_value=args.critic_learning_rate,
        end_value=args.critic_learning_rate_end,
        transition_steps=TOTAL_STEPS
    )
    
    actor_optimizer = adam(learning_rate=actor_lr_schedule)
    critic_optimizer = adam(learning_rate=critic_lr_schedule)
    
    actor_opt_state = actor_optimizer.init(actor_params)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Simple replay buffer (in-memory)
    buffer = {
        'obs': jnp.zeros((BUFFER_SIZE, obs_dim)),
        'actions': jnp.zeros((BUFFER_SIZE, action_dim)),
        'rewards': jnp.zeros(BUFFER_SIZE),
        'next_obs': jnp.zeros((BUFFER_SIZE, obs_dim)),
        'dones': jnp.zeros(BUFFER_SIZE, dtype=bool),
        'ptr': 0,
        'size': 0
    }

    @jax.jit
    def update_critic(critic_params, target_critic_params, actor_params, batch):
        obs, actions, rewards, next_obs, dones = batch
        
        # Target Q-values
        next_actions = actor.apply(actor_params, next_obs)
        q1_target_atoms, q2_target_atoms = critic.apply(target_critic_params, next_obs, next_actions)
        q1_target = critic.apply(target_critic_params, q1_target_atoms, method=Critic.get)
        q2_target = critic.apply(target_critic_params, q2_target_atoms, method=Critic.get)
        target_q = jnp.minimum(q1_target, q2_target)
        target_q = rewards + args.gamma * (1 - dones) * target_q
        
        # Current Q-values
        q1_atoms, q2_atoms = critic.apply(critic_params, obs, actions)
        q1 = critic.apply(critic_params, q1_atoms, method=Critic.get)
        q2 = critic.apply(critic_params, q2_atoms, method=Critic.get)
        q1_loss = jnp.mean((q1 - target_q) ** 2)
        q2_loss = jnp.mean((q2 - target_q) ** 2)
        critic_loss = q1_loss + q2_loss
        
        return critic_loss
    
    @jax.jit
    def update_critic_with_q_values(critic_params, target_critic_params, actor_params, batch):
        obs, actions, rewards, next_obs, dones = batch
        
        # Target Q-values
        next_actions = actor.apply(actor_params, next_obs)
        q1_target_atoms, q2_target_atoms = critic.apply(target_critic_params, next_obs, next_actions)
        q1_target = critic.apply(target_critic_params, q1_target_atoms, method=Critic.get)
        q2_target = critic.apply(target_critic_params, q2_target_atoms, method=Critic.get)
        target_q = jnp.minimum(q1_target, q2_target)
        target_q = rewards + args.gamma * (1 - dones) * target_q
        
        # Current Q-values
        q1_atoms, q2_atoms = critic.apply(critic_params, obs, actions)
        q1 = critic.apply(critic_params, q1_atoms, method=Critic.get)
        q2 = critic.apply(critic_params, q2_atoms, method=Critic.get)
        q1_loss = jnp.mean((q1 - target_q) ** 2)
        q2_loss = jnp.mean((q2 - target_q) ** 2)
        critic_loss = q1_loss + q2_loss
        
        return critic_loss, (q1, q2)
    
    @jax.jit
    def update_actor(actor_params, critic_params, batch):
        obs, _, _, _, _ = batch
        
        actions = actor.apply(actor_params, obs)
        q1_atoms, q2_atoms = critic.apply(critic_params, obs, actions)
        q1 = critic.apply(critic_params, q1_atoms, method=Critic.get)
        q2 = critic.apply(critic_params, q2_atoms, method=Critic.get)
        q_value = jnp.minimum(q1, q2)
        actor_loss = -jnp.mean(q_value)
        
        return actor_loss
    
    @jax.jit
    def update_target(target_params, params, tau=0.005):
        return jax.tree.map(lambda t, p: tau * p + (1 - tau) * t, target_params, params)
    
    for step in range(TOTAL_STEPS):
        obs_batch = jnp.expand_dims(obs, axis=0)
        action = actor.apply(actor_params, obs_batch) + jax.random.normal(jax.random.key(step), (1, action_dim)) * 0.05
        action = jnp.clip(action, -1, 1)
        
        next_obs, reward, done, truncated, info = env.step(action[0])
        
        # Store in buffer
        idx = buffer['ptr']
        buffer['obs'] = buffer['obs'].at[idx].set(obs)
        buffer['actions'] = buffer['actions'].at[idx].set(action[0])
        buffer['rewards'] = buffer['rewards'].at[idx].set(reward)
        buffer['next_obs'] = buffer['next_obs'].at[idx].set(next_obs)
        buffer['dones'] = buffer['dones'].at[idx].set(done)
        
        buffer['ptr'] = (buffer['ptr'] + 1) % BUFFER_SIZE
        buffer['size'] = min(buffer['size'] + 1, BUFFER_SIZE)
        
        obs = next_obs
        
        # Update networks if enough data
        if buffer['size'] > BATCH_SIZE:
            # Sample batch
            batch_idx = jax.random.randint(jax.random.key(step), (BATCH_SIZE,), 0, buffer['size'])
            batch = (
                buffer['obs'][batch_idx],
                buffer['actions'][batch_idx],
                buffer['rewards'][batch_idx],
                buffer['next_obs'][batch_idx],
                buffer['dones'][batch_idx]
            )
            
            # Update critic
            critic_loss, (q1, q2) = update_critic_with_q_values(critic_params, target_critic_params, actor_params, batch)
            critic_grads = jax.grad(update_critic, argnums=0)(critic_params, target_critic_params, actor_params, batch)
            critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state)
            critic_params = apply_updates(critic_params, critic_updates)
            
            # Update actor
            actor_loss = update_actor(actor_params, critic_params, batch)
            actor_grads = jax.grad(update_actor, argnums=0)(actor_params, critic_params, batch)
            actor_updates, actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state)
            actor_params = apply_updates(actor_params, actor_updates)
            
            # Update target network
            target_critic_params = update_target(target_critic_params, critic_params, args.tau)
            
            # Logging
            if step % 100 == 0:
                current_actor_lr = actor_lr_schedule(step)
                current_critic_lr = critic_lr_schedule(step)
                print(f'Step {step}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}')
                print(f'Q1: {q1.mean():.4f}, Q2: {q2.mean():.4f}')
                print(f'Actor LR: {current_actor_lr:.6f}, Critic LR: {current_critic_lr:.6f}')
                print(f'Reward: {reward:.4f}')
                print('---')

if __name__ == "__main__":
    main()