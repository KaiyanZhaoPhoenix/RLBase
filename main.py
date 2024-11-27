import os
import numpy as np
import random
import time
import wandb
import argparse
import torch
from envs.atari_env import create_atari_env
from algo import PPO

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def initialize_wandb(args):
    return wandb.init(
        project="gen",  # Specify your WandB project name here
        name = f"{args.env}_{args.policy_type}_{args.seed}",
        config=vars(args),  # Automatically logs arguments as config
        sync_tensorboard=True,  # Sync TensorBoard logs (optional)
        save_code=True,  # Optionally save the code used to this run
        notes="Training PPO on Atari environment",  # Optional notes for the run
        mode='online'  # Ensure that the run is being logged online
    )

def main(args):
    # Initialize configuration and seed
    env_id = args.env
    policy_type = args.policy_type
    total_timesteps = args.total_timesteps
    use_cuda = args.use_cuda

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)  # If no seed is specified, randomly generate one
    set_seed(args.seed)

    # Device setup (CUDA if available, else CPU)
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Weights and Biases for experiment tracking
    run = initialize_wandb(args)

    # Create and reset the Atari environment
    env = create_atari_env(env_id)
    env.reset(seed=args.seed)

    # Initialize PPO model
    model = PPO(policy_type, env, verbose=1, tensorboard_log=f"runs/ppo")

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Finish the wandb run and log the final model
    run.finish()

if __name__ == '__main__':
    # Argument parser for configurable options
    parser = argparse.ArgumentParser(description='Atari PPO Training')

    # General configurations
    parser.add_argument('--env', type=str, default="ALE/Breakout-v5", help='Atari environment name')
    parser.add_argument('--policy_type', type=str, default="CnnPolicy", help='Policy type for PPO')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total timesteps for training')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use CUDA for training')

    args = parser.parse_args()

    # Track training time and run the main function
    training_start_time = time.time()
    main(args)
    training_duration = time.time() - training_start_time
    print(f'Training time: {training_duration / 3600:.2f} hours')
