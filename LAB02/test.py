import torch
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def run_episode(env, policy):
    """Run a single episode and collect trajectory data."""
    observations = []
    actions = []
    log_probs = []
    rewards = []
    
    obs, _ = env.reset(seed=123)
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        action_probs = policy(obs_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        observations.append(obs)
        actions.append(action.item())
        log_probs.append(log_prob)
        rewards.append(reward)
        
        obs = next_obs

    # Convert to tensors
    observations = torch.tensor(np.array(observations), dtype=torch.float32)
    log_probs = torch.stack(log_probs).squeeze(-1)
    
    return observations, actions, log_probs, rewards

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

def reinforce_with_baseline(policy, value_function, env, env_render=None, 
                           gamma=0.99, num_episodes=50, check_freq=20, 
                           policy_lr=0.001, value_lr=0.001, standardize=True):
    """
    REINFORCE algorithm with value function baseline.
    Keeps policy and value function updates completely separate.
    """
    
    # Optimizers
    policy_opt = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    value_opt = torch.optim.Adam(value_function.parameters(), lr=value_lr)
    
    # Loss function for value function
    mse_loss = torch.nn.MSELoss()
    
    running_rewards = [0.0]
    writer = SummaryWriter(log_dir="runs/reinforce_baseline")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        # Collect episode data
        observations, actions, log_probs, rewards = run_episode(env, policy)
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        
        # Update running average
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])
        
        # ===== VALUE FUNCTION UPDATE =====
        # Train value function to predict returns
        value_opt.zero_grad()
        
        # Forward pass through value function
        state_values = value_function(observations).squeeze(-1)
        
        # Value function loss
        value_loss = mse_loss(state_values, returns)
        
        # Backward pass for value function
        value_loss.backward()
        value_opt.step()
        
        # ===== POLICY UPDATE =====
        policy_opt.zero_grad()
        
        # Compute baseline (no gradients needed)
        with torch.no_grad():
            baseline = value_function(observations).squeeze(-1)
        
        # Compute advantages
        advantages = returns - baseline
        
        # Standardize advantages if requested
        if standardize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -(log_probs * advantages).mean()
        
        print(f"Policy Loss: {policy_loss.item():.6f}, Value Loss: {value_loss.item():.6f}")
        
        # Backward pass for policy
        policy_loss.backward()
        policy_opt.step()
        
        # ===== LOGGING =====
        total_reward = sum(rewards)
        
        if episode % check_freq == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Running Avg: {running_rewards[-1]:.2f}")
            
            writer.add_scalar("Reward/Episode", total_reward, episode)
            writer.add_scalar("Reward/Running_Avg", running_rewards[-1], episode)
            writer.add_scalar("Loss/Policy", policy_loss.item(), episode)
            writer.add_scalar("Loss/Value", value_loss.item(), episode)
            writer.add_scalar("Episode_Length", len(rewards), episode)
        
        # Render occasionally
        if episode % 100 == 0 and env_render:
            print(f'Running reward: {running_rewards[-1]:.2f}')
            policy.eval()
            run_episode(env_render, policy)
            policy.train()
    
    writer.close()
    policy.eval()
    return running_rewards

def reinforce_vanilla(policy, env, env_render=None, gamma=0.99, num_episodes=50, 
                     check_freq=20, policy_lr=0.001, standardize=True):
    """
    Vanilla REINFORCE algorithm without baseline.
    """
    
    policy_opt = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    running_rewards = [0.0]
    writer = SummaryWriter(log_dir="runs/reinforce_vanilla")
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        # Collect episode data
        observations, actions, log_probs, rewards = run_episode(env, policy)
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        
        # Update running average
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])
        
        # Standardize returns if requested
        if standardize and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy update
        policy_opt.zero_grad()
        policy_loss = -(log_probs * returns).mean()
        
        print(f"Policy Loss: {policy_loss.item():.6f}")
        
        policy_loss.backward()
        policy_opt.step()
        
        # Logging
        total_reward = sum(rewards)
        
        if episode % check_freq == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Running Avg: {running_rewards[-1]:.2f}")
            
            writer.add_scalar("Reward/Episode", total_reward, episode)
            writer.add_scalar("Reward/Running_Avg", running_rewards[-1], episode)
            writer.add_scalar("Loss/Policy", policy_loss.item(), episode)
        
        if episode % 100 == 0 and env_render:
            print(f'Running reward: {running_rewards[-1]:.2f}')
            policy.eval()
            run_episode(env_render, policy)
            policy.train()
    
    writer.close()
    policy.eval()
    return running_rewards