from reinforce import reinforce
import models
import test

import gymnasium as gym
import torch

import torch.nn.functional as F

if __name__ == "__main__":

    config = {
        "project_name": "LAB02_Exercise2",
        "dataset_name": "CartPole-v1",
        "training": {
            "learning_rate": 0.001,
            "optimizer": "adam",
            "epochs": 3000,
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

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env_render = gym.make('CartPole-v1', render_mode='human')

    policy = models.Policy(input_size=env.observation_space.shape[0], actions=env.action_space.n).to(device)

    #rewards_false = reinforce(policy, env, env_render=env_render, gamma=0.99, num_episodes=config['training']['epochs'], standardize=False)
    #rewards_true = reinforce(policy, env, env_render=env_render, gamma=0.99, num_episodes=config['training']['epochs'], standardize=True)

    #print("Training completed. Final running reward (without standardization):", rewards_false[-1])
    #print("Training completed. Final running reward (with standardization):", rewards_true[-1])

    value = models.Policy(input_size=env.observation_space.shape[0], actions=1, softmax=False).to(device)

    #rewards = test.reinforce_with_baseline(policy=value, env=env, env_render=env_render, gamma=0.99, num_episodes=config['training']['epochs'], value_function=value)
    rewards = reinforce(policy=policy, env=env, env_render=env_render, gamma=0.99, num_episodes=config['training']['epochs'], value_function=value, config = config)

    env.close()
    env_render.close()
    

   
