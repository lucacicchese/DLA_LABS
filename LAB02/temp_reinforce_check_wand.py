import os
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from gym.wrappers import RecordVideo


def reinforce(policy, env, env_render=None, gamma=0.99, num_episodes=50, check_freq=20, standardize=True, value_function=None, device=torch.device("cpu"), record=False, checkpoint_freq=10, checkpoint_dir="logs/checkpoints", wandb_project="reinforce_cartpole"):

    # Initialize wandb
    wandb.init(project=wandb_project)
    wandb.watch(policy, log="all")
    
    if record:
        video_dir = "cartpole_videos"
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder="assets/videos",
            episode_trigger=lambda episode_id: episode_id % 200 == 0,
            name_prefix="reinforce"
        )

    policy.to(device)
    policy_opt = Adam(policy.parameters(), lr=0.001)
    policy.train()
    
    if value_function is not None:
        value_function.to(device)
        value_opt = Adam(value_function.parameters(), lr=0.001)
        mse_loss = torch.nn.MSELoss()
        value_function.train()

    # Check if checkpoint exists and load it
    if os.path.exists(checkpoint_dir):
        checkpoint_files = os.listdir(checkpoint_dir)
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))  # Assuming filename format like 'checkpoint_100.pth'
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
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

        # Calculate total reward for this episode
        ep_reward = sum(rewards)

        # Update the running reward (using total episode reward)
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

        # Check if the agent has solved the environment
        if running_rewards[-1] >= 195:
            print(f"The agent won! Environment solved in {episode + 1} episodes.")
            break

        if episode % check_freq == 0:
            print(f"Episode {episode + 1}, Total Reward: {ep_reward}, Running Avg: {running_rewards[-1]}")

            # Log to wandb
            wandb.log({
                "Episode": episode + 1,
                "Total Reward": ep_reward,
                "Running Avg Reward": running_rewards[-1],
                "Loss": loss.item()
            })

            # Average total reward for the episode
            avg_reward = ep_reward / len(rewards)

            # Average episode length
            avg_length = len(rewards)

            writer.add_scalar("Reward/Episode", avg_reward, episode)
            writer.add_scalar("Length/Episode", avg_length, episode)
            writer.add_scalar("Reward/Running_Avg", running_rewards[-1], episode)

        # Save checkpoint every 'checkpoint_freq' episodes
        if episode % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode + 1}.pth")
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': policy_opt.state_dict(),
                'value_function_state_dict': value_function.state_dict() if value_function else None,
                'value_optimizer_state_dict': value_opt.state_dict() if value_function else None
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        if not episode % 100:
            if env_render:
                policy.eval()
                run_episode(env_render, policy, device)
                policy.train()
            print(f'Running reward: {running_rewards[-1]}')

    # Finish wandb logging
    wandb.finish()
    
    policy.eval()
    return running_rewards
