import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import jax
import jax.numpy as jnp
from optax import adam, apply_updates, linear_schedule
import gymnasium as gym

from fast_td3 import Actor, Critic
from utils.replay_buffer import ReplayBuffer
from utils.reward_normalizer import RewardNormalizer


TOTAL_STEPS = 10000
BATCH_SIZE = 256
BUFFER_SIZE = 50000


# def main():

#     # Configuration
#     v_min = -10.0
#     v_max = 10.0
#     num_atoms = 51
#     gamma = 0.99
#     tau = 0.005
#     actor_learning_rate = 3e-4
#     critic_learning_rate = 3e-4
#     actor_learning_rate_end = 3e-4
#     critic_learning_rate_end = 3e-4
#     std_min = 0.1
#     std_max = 0.3
#     init_scale = 0.01
    
#     # Environment setup
#     env = gym.make("Pendulum-v1")    
#     obs, info = env.reset()
    
#     # Dimensions
#     obs_dim = 3
#     action_dim = 1
#     hidden_dim = 256
    
#     # Create networks
#     actor = Actor(
#         n_obs=obs_dim, 
#         n_act=action_dim, 
#         n_envs=1, 
#         init_scale=init_scale, 
#         hidden_dim=hidden_dim, 
#         std_min=std_min, 
#         std_max=std_max
#     )
    
#     critic = Critic(
#         n_obs=obs_dim,
#         n_act=action_dim,
#         num_atoms=num_atoms,
#         v_min=v_min,
#         v_max=v_max,
#         hidden_dim=hidden_dim
#     )
    
#     # Initialize networks
#     rng = jax.random.key(0)
#     obs_batch = jnp.expand_dims(obs, axis=0)
#     action_batch = jnp.expand_dims(jnp.zeros(action_dim), axis=0)
    
#     actor_params = actor.init(rng, obs_batch)
#     critic_params = critic.init(rng, obs_batch, action_batch)
#     target_critic_params = critic_params
    
#     # Create optimizers with schedulers
#     actor_lr_schedule = linear_schedule(
#         init_value=actor_learning_rate,
#         end_value=actor_learning_rate_end,
#         transition_steps=TOTAL_STEPS
#     )
#     critic_lr_schedule = linear_schedule(
#         init_value=critic_learning_rate,
#         end_value=critic_learning_rate_end,
#         transition_steps=TOTAL_STEPS
#     )
    
#     actor_optimizer = adam(learning_rate=actor_lr_schedule)
#     critic_optimizer = adam(learning_rate=critic_lr_schedule)
    
#     actor_opt_state = actor_optimizer.init(actor_params)
#     critic_opt_state = critic_optimizer.init(critic_params)
    
#     # Create replay buffer
#     replay_buffer = ReplayBuffer.create(
#         n_env=1,
#         buffer_size=BUFFER_SIZE,
#         n_obs=obs_dim,
#         n_act=action_dim,
#         n_critic_obs=obs_dim,
#         asymmetric_obs=False,
#         playground_mode=False
#     )
    
#     # Create reward normalizer
#     reward_normalizer = RewardNormalizer.create(
#         gamma=gamma,
#         shape=(1,),
#         g_max=10.0,
#         epsilon=1e-8
#     )

#     @jax.jit
#     def update_critic(critic_params, target_critic_params, actor_params, batch):
#         obs, actions, rewards, next_obs, dones = batch
        
#         # Target Q-values
#         next_actions = actor.apply(actor_params, next_obs)
#         q1_target_atoms, q2_target_atoms = critic.apply(target_critic_params, next_obs, next_actions)
#         q1_target = critic.apply(target_critic_params, q1_target_atoms, method=critic.get)
#         q2_target = critic.apply(target_critic_params, q2_target_atoms, method=critic.get)
#         target_q = jnp.minimum(q1_target, q2_target)
#         target_q = rewards + gamma * (1 - dones) * target_q
        
#         # Current Q-values
#         q1_atoms, q2_atoms = critic.apply(critic_params, obs, actions)
#         q1 = critic.apply(critic_params, q1_atoms, method=critic.get)
#         q2 = critic.apply(critic_params, q2_atoms, method=critic.get)
#         q1_loss = jnp.mean((q1 - target_q) ** 2)
#         q2_loss = jnp.mean((q2 - target_q) ** 2)
#         critic_loss = q1_loss + q2_loss
        
#         return critic_loss
    
#     @jax.jit
#     def update_critic_with_q_values(critic_params, target_critic_params, actor_params, batch):
#         obs, actions, rewards, next_obs, dones = batch
        
#         # Target Q-values
#         next_actions = actor.apply(actor_params, next_obs)
#         q1_target_atoms, q2_target_atoms = critic.apply(target_critic_params, next_obs, next_actions)
#         q1_target = critic.apply(target_critic_params, q1_target_atoms, method=critic.get)
#         q2_target = critic.apply(target_critic_params, q2_target_atoms, method=critic.get)
#         target_q = jnp.minimum(q1_target, q2_target)
#         target_q = rewards + gamma * (1 - dones) * target_q
        
#         # Current Q-values
#         q1_atoms, q2_atoms = critic.apply(critic_params, obs, actions)
#         q1 = critic.apply(critic_params, q1_atoms, method=critic.get)
#         q2 = critic.apply(critic_params, q2_atoms, method=critic.get)
#         q1_loss = jnp.mean((q1 - target_q) ** 2)
#         q2_loss = jnp.mean((q2 - target_q) ** 2)
#         critic_loss = q1_loss + q2_loss
        
#         return critic_loss, (q1, q2)
    
#     @jax.jit
#     def update_actor(actor_params, critic_params, batch):
#         obs, _, _, _, _ = batch
        
#         actions = actor.apply(actor_params, obs)
#         q1_atoms, q2_atoms = critic.apply(critic_params, obs, actions)
#         q1 = critic.apply(critic_params, q1_atoms, method=critic.get)
#         q2 = critic.apply(critic_params, q2_atoms, method=critic.get)
#         q_value = jnp.minimum(q1, q2)
#         actor_loss = -jnp.mean(q_value)
        
#         return actor_loss
    
#     @jax.jit
#     def update_target(target_params, params, tau=0.005):
#         return jax.tree.map(lambda t, p: tau * p + (1 - tau) * t, target_params, params)
    
#     # Training loop
#     total_reward = 0.0
#     episode_count = 0
    
#     for step in range(TOTAL_STEPS):
#         obs_batch = jnp.expand_dims(obs, axis=0)
#         action = actor.apply(actor_params, obs_batch) + jax.random.normal(jax.random.key(step), (1, action_dim)) * 0.1
#         action = jnp.clip(action, -1, 1)
        
#         next_obs, reward, done, truncated, info = env.step(action[0])
#         total_reward += float(reward)
        
#         # Normalize reward
#         reward_normalizer, normalized_reward = reward_normalizer.apply(
#             jnp.array([reward]), jnp.array([done])
#         )
        
#         # Store in replay buffer
#         data = {
#             'observations': obs_batch,
#             'actions': action,
#             'critic_observations': obs_batch,
#             'next': {
#                 'observations': jnp.expand_dims(next_obs, axis=0),
#                 'rewards': normalized_reward,
#                 'dones': jnp.array([done]),
#                 'truncations': jnp.array([truncated]),
#                 'critic_observations': jnp.expand_dims(next_obs, axis=0)
#             }
#         }
#         replay_buffer = replay_buffer.extend(data)
        
#         obs = next_obs
        
#         # Reset environment if episode is done
#         if done or truncated:
#             obs, info = env.reset()
#             episode_count += 1
#             total_reward = 0
        
#         # Update networks if enough data
#         if replay_buffer.ptr > BATCH_SIZE:
#             # Sample batch
#             batch = replay_buffer.sample(jax.random.key(step), BATCH_SIZE)
            
#             # Extract batch data
#             batch_obs = batch['observations']
#             batch_actions = batch['actions']
#             batch_rewards = batch['next']['rewards']
#             batch_next_obs = batch['next']['observations']
#             batch_dones = batch['next']['dones']
            
#             # Update critic
#             critic_loss, (q1, q2) = update_critic_with_q_values(
#                 critic_params, target_critic_params, actor_params, 
#                 (batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)
#             )
#             critic_grads = jax.grad(update_critic, argnums=0)(
#                 critic_params, target_critic_params, actor_params,
#                 (batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)
#             )
#             critic_updates, critic_opt_state = critic_optimizer.update(critic_grads, critic_opt_state)
#             critic_params = apply_updates(critic_params, critic_updates)
            
#             # Update actor
#             actor_loss = update_actor(
#                 actor_params, critic_params,
#                 (batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)
#             )
#             actor_grads = jax.grad(update_actor, argnums=0)(
#                 actor_params, critic_params,
#                 (batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)
#             )
#             actor_updates, actor_opt_state = actor_optimizer.update(actor_grads, actor_opt_state)
#             actor_params = apply_updates(actor_params, actor_updates)
            
#             # Update target network
#             target_critic_params = update_target(target_critic_params, critic_params, tau)
            
#             # Logging
#             if step % 500 == 0:
#                 current_actor_lr = actor_lr_schedule(step)
#                 current_critic_lr = critic_lr_schedule(step)
#                 print(f'Step {step}, Episode {episode_count}')
#                 print(f'Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}')
#                 print(f'Q1: {q1.mean():.4f}, Q2: {q2.mean():.4f}')
#                 print(f'Actor LR: {current_actor_lr:.6f}, Critic LR: {current_critic_lr:.6f}')
#                 print(f'Reward: {reward:.4f}, Normalized: {normalized_reward[0]:.4f}')
#                 print('---')

# if __name__ == "__main__":
#     main()