import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import os
from gym.wrappers import RecordVideo
import wandb


def run_episode(env, policy, device=torch.device("cpu")):

    observations = []
    actions = []
    log_probs = []
    rewards = []
    obs, _ = env.reset(seed=123)
    done = False

    while not done:

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
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
    log_probs = log_probs.squeeze(-1).to(device)

    observations = torch.tensor(observations, dtype=torch.float32, device=device)
    return observations, actions, log_probs, rewards


def reinforce(policy, env, env_render=None, gamma=0.99, num_episodes=20, check_freq=20, standardize=False, value_function=None, device=torch.device("cpu"), record=True, winning_score=195, config=None):

    if config is not None and config["logging"]["wandb"]:
        wandb.init(project=config["project_name"])
        wandb.watch(policy, log="all")

    if record:
        video_dir = "assets/videos"
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: episode_id % 50 == 0,
            name_prefix="reinforce"
        )

    policy.to(device)

    if config["training"]["optimizer"] == "adamW":
        policy_opt = torch.optim.AdamW(policy.parameters(), lr=config["training"]["learning_rate"], weight_decay=0.0001)
    else:
        policy_opt = torch.optim.Adam(policy.parameters(), lr=config["training"]["learning_rate"])
    

    policy.train()
    
    if value_function is not None:
        value_function.to(device)
        if config["training"]["optimizer"] == "adamW":
            value_opt = torch.optim.AdamW(value_function.parameters(), lr=config["training"]["learning_rate"], weight_decay=0.0001)
        else:
            value_opt = torch.optim.Adam(value_function.parameters(), lr=config["training"]["learning_rate"])
        mse_loss = torch.nn.MSELoss()
        value_function.train()

    if os.path.exists(config["logging"]["save_dir"]):
        latest_checkpoint = f"{config['project_name']}_latest_checkpoint.pth"
        checkpoint_path = os.path.join(config["logging"]["save_dir"], latest_checkpoint)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            policy_opt.load_state_dict(checkpoint['optimizer_state_dict'])
            if value_function is not None:
                value_function.load_state_dict(checkpoint['value_function_state_dict'])
                value_opt.load_state_dict(checkpoint['value_optimizer_state_dict'])
            print(f"Checkpoint {latest_checkpoint} loaded.")
        else:
            print("No checkpoint found. Starting from scratch.")

    running_rewards = [0.0]
    policy.train()

    writer = SummaryWriter(log_dir="runs/reinforce_cartpole")

    for episode in range(num_episodes):

        print(f"Episode {episode + 1}/{num_episodes}")

        G = 0
        returns = []
        
        (observations, actions, log_probs, rewards) = run_episode(env, policy, device)


        for t in reversed(rewards):
            G = t + gamma * G
            returns.insert(0, G)

            
        returns = torch.tensor(returns, dtype=torch.float32)
        
        ep_reward = sum(rewards)

        running_rewards.append(0.05 * ep_reward + 0.95 * running_rewards[-1])

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

        if config is not None and config["logging"]["wandb"]:
            wandb.log({
                "Episode": episode + 1,
                "Total Reward": ep_reward,
                "Running Avg Reward": running_rewards[-1],
                "Loss": loss.item()
            })




        if episode % check_freq == 0:
            print(f"Episode {episode + 1}, Total Reward: {ep_reward}, Running Avg: {running_rewards[-1]}")

            # Average total reward for the episode
            avg_reward = ep_reward / len(rewards)

            # Average episode length
            avg_length = len(rewards)

            writer.add_scalar("Reward/Episode", avg_reward, episode)
            writer.add_scalar("Length/Episode", avg_length, episode)

            writer.add_scalar("Reward/Running_Avg", running_rewards[-1], episode)

        if running_rewards[-1] >= winning_score:
            print(f"The agent won! Environment solved in {episode + 1} episodes.")
            break

        if episode % config["logging"]["save_frequency"] == 0:
            checkpoint_path = os.path.join(config["logging"]["save_dir"], f"{config['project_name']}_checkpoint_{episode + 1}.pth")
            latest_checkpoint_path = os.path.join(config["logging"]["save_dir"], f"{config['project_name']}_latest_checkpoint.pth")
            os.makedirs(config["logging"]["save_dir"], exist_ok=True)

            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': policy_opt.state_dict(),
                'value_function_state_dict': value_function.state_dict() if value_function else None,
                'value_optimizer_state_dict': value_opt.state_dict() if value_function else None
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, latest_checkpoint_path)
            #print(f"Checkpoint saved at {checkpoint_path}")

        if not episode % 100:
            if env_render:
                policy.eval()
                run_episode(env_render, policy, device)
                policy.train()
            #print(f'Running reward: {running_rewards[-1]}')
    
    
    policy.eval()
    return running_rewards
