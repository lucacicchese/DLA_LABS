import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter


def run_episode(env, policy):

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


    log_probs = torch.stack(log_probs)
    log_probs = log_probs.squeeze(-1)

    observations = torch.tensor(observations, dtype=torch.float32)
    return observations, actions, log_probs, rewards


def reinforce(policy, env, env_render=None, gamma=0.99, num_episodes=50, check_freq=20, standardize=True, value_function=None, device=torch.device("cpu")):
    #torch.autograd.set_detect_anomaly(True)

    print(policy is value_function)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=0.001)
    policy.train()
    
    if value_function is not None:
        value_opt = torch.optim.Adam(value_function.parameters(), lr=0.001)
        mse_loss = torch.nn.MSELoss()
        value_function.train()

    running_rewards = [0.0]
    policy.train()

    writer = SummaryWriter(log_dir="runs/reinforce_cartpole")

    for episode in range(num_episodes):

        print(f"Episode {episode + 1}/{num_episodes}")

        G = 0
        returns = []
        
        (observations, actions, log_probs, rewards) = run_episode(env, policy)


        for t in reversed(rewards):
            G = t + gamma * G
            returns.insert(0, G)

            
        returns = torch.tensor(returns, dtype=torch.float32)

        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        if value_function is not None:
            # Clone returns for value function training to avoid shared tensors
            returns_for_value = returns.clone().detach()
            
            values = value_function(observations)
            values = values.view(-1)
            returns_for_value = returns_for_value.view(-1)


            values_loss = mse_loss(values, returns_for_value)
            
            value_opt.zero_grad()
            values_loss.backward()
            value_opt.step()

            # Compute baseline after value function update
            with torch.no_grad():
                baseline = value_function(observations)
                baseline = baseline.view(-1)
            
            # Create advantages (new tensor, not in-place modification)
            advantages = returns - baseline
            returns = advantages

        if standardize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)


        policy_opt.zero_grad()
        log_probs = log_probs.view(-1)
        returns = returns.view(-1)

        loss = (-log_probs * returns).mean()

        loss.backward()
        policy_opt.step()

        total_reward = sum(rewards)

        ep_reward = sum(rewards)


        if episode % check_freq == 0:
            print(f"Episode {episode + 1}, Total Reward: {ep_reward}, Running Avg: {running_rewards[-1]}")

            # Average total reward for the episode
            avg_reward = total_reward / len(rewards)

            # Average episode length
            avg_length = len(rewards)

            writer.add_scalar("Reward/Episode", avg_reward, episode)
            writer.add_scalar("Length/Episode", avg_length, episode)

            writer.add_scalar("Reward/Running_Avg", running_rewards[-1], episode)

        if not episode % 100:
            if env_render:
                policy.eval()
                run_episode(env_render, policy)
                policy.train()
            print(f'Running reward: {running_rewards[-1]}')
    
    
    policy.eval()
    return running_rewards
