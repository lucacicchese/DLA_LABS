"""
LAB02
Exercise 1

Policy Gradient Method applied to the cartpole environment
"""

# Import my modules
from reinforce import reinforce
import models

# Import external libraries
import gymnasium as gym
import torch
import torch.nn.functional as F

if __name__ == "__main__":

    config = {
        "project_name": "LAB02_Exercise1",
        "dataset_name": "CartPole-v1",
        "training": {
            "learning_rate": 0.001,
            "optimizer": "adam",
            "epochs": 2000,
            "batch_size": 64,
            "resume": False,
            "layers": [4, 64, 64, 2],
            "loss_function": "crossentropy"
        },
        "model": {
            "type": "mlp",
            "layers": [4, 64, 64, 2]
        },
        "logging": {
            "tensorboard": True,
            "weightsandbiases": True,
            "wandb": True,
            "tb_logs": "tensorboard_runs",
            "save_dir": "checkpoints",
            "save_frequency": 1
        },
        "env_settings" : {
            "record": True,
            "winning_score": 475
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env_render = gym.make('CartPole-v1', render_mode='human')

    policy = models.Policy(input_size=env.observation_space.shape[0], actions=env.action_space.n).to(device)

    rewards = reinforce(policy, env, env_render=env_render, gamma=0.99, standardize=True, num_episodes=config['training']['epochs'], record=config['env_settings']['record'], winning_score=config['env_settings']['winning_score'], config=config)

    print("Training completed. Final running reward:", rewards[-1])
    env.close()
    env_render.close()
    

   
