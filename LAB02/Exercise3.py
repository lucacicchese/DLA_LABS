from reinforce import reinforce
import models

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
            "epochs": 10000,
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
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    env = gym.make('LunarLander-v3')
    env_render = gym.make('LunarLander-v3', render_mode='human')

    policy = models.Policy(input_size=env.observation_space.shape[0], actions=env.action_space.n).to(device)

    rewards = reinforce(policy, env, env_render=env_render, gamma=0.99, num_episodes=config['training']['epochs'])

    print("Training completed. Final running reward:", rewards[-1])
    env.close()
    env_render.close()
    

   