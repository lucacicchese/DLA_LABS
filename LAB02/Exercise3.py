"""
LAB02
Exercise 3.1

Policy Gradient Method + baseline applied to the lunar lander environment
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
        "project_name": "LAB02_Exercise3",
        "dataset_name": "LunarLander-v3",
        "training": {
            "learning_rate": 0.01,
            "optimizer": "adamW",
            "epochs": 10000,
            "batch_size": 64,
            "resume": False,
        },
        "logging": {
            "tensorboard": True,
            "weightsandbiases": True,
            "wandb": True,
            "tb_logs": "tensorboard_runs",
            "save_dir": "checkpoints",
            "save_frequency": 100
        },
        "env_settings" : {
            "record": False,
            "winning_score": 200,
            "wind": False,
            "wind_power": 13,
            "turbulence": 1.3
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    if config['env_settings']['wind']:
        print("Wind enabled in the environment.")
        env = gym.make('LunarLander-v3', enable_wind=config['env_settings']['wind'], wind_power=config['env_settings']['wind_power'], turbulence_power=config['env_settings']['turbulence'], render_mode='rgb_array')
        env_render = gym.make('LunarLander-v3', enable_wind=config['env_settings']['wind'], wind_power=config['env_settings']['wind_power'], turbulence_power=config['env_settings']['turbulence'], render_mode='human')

    else:
        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        env_render = gym.make('LunarLander-v3', render_mode='human')

    policy = models.Policy_Lander(input_size=env.observation_space.shape[0], actions=env.action_space.n).to(device)
    value_function = models.Policy_Lander(input_size=env.observation_space.shape[0], actions=1, softmax=False).to(device)

    rewards = reinforce(policy, env, env_render=env_render, gamma=0.99, num_episodes=config['training']['epochs'], value_function=value_function, standardize=True, winning_score=config['env_settings']['winning_score'], record=config['env_settings']['record'], config=config)

    print("Training completed. Final running reward:", rewards[-1])
    env.close()
    env_render.close()
    

   