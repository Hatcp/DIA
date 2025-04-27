import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from datetime import datetime
import tarware

from hseac_tarware import HSEAC

def parse_args():
    """Parse command line arguments for HSEAC training and evaluation."""
    parser = argparse.ArgumentParser(description='Train and evaluate HSEAC on TA-RWARE environment')
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='tarware-tiny-3agvs-2pickers-globalobs-v1',
                      help='Name of the TA-RWARE environment')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=10000,
                      help='Number of episodes to train')
    parser.add_argument('--max_steps', type=int, default=500,
                      help='Maximum steps per episode')
    parser.add_argument('--update_interval', type=int, default=100,
                      help='Update interval for network training')
    
    # Algorithm hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--manager_lr', type=float, default=3e-4,
                      help='Learning rate for manager network')
    parser.add_argument('--worker_lr', type=float, default=3e-4,
                      help='Learning rate for worker networks')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                      help='Entropy coefficient for exploration')
    parser.add_argument('--lambda_coef', type=float, default=1.0,
                      help='Coefficient for shared experience')
    parser.add_argument('--goal_completion_steps', type=int, default=20,
                      help='Maximum steps before considering a goal complete')
    
    # Experiment settings
    parser.add_argument('--exp_name', type=str, default=None,
                      help='Experiment name for logging')
    parser.add_argument('--log_dir', type=str, default='runs',
                      help='Directory for tensorboard logs')
    parser.add_argument('--load_path', type=str, default=None,
                      help='Path to load models from')
    
    # Action flags
    parser.add_argument('--train', action='store_true',
                      help='Train the agents')
    parser.add_argument('--evaluate', action='store_true',
                      help='Evaluate the agents')
    parser.add_argument('--render', action='store_true',
                      help='Render during training/evaluation')
    parser.add_argument('--eval_episodes', type=int, default=10,
                      help='Number of episodes for evaluation')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_experiment(args):
    """Setup experiment directories and save configuration."""
    # Create experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.env_name}_{timestamp}"
    
    # Create directories
    log_dir = os.path.join(args.log_dir, args.exp_name)
    save_dir = os.path.join("models", args.exp_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    
    # Save experiment configuration
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        import json
        json.dump(vars(args), f, indent=4)
    
    print(f"Experiment: {args.exp_name}")
    print(f"Configuration saved to: {config_path}")
    
    return log_dir, save_dir

def create_environment(env_name, seed=None):
    """Create and initialize the TA-RWARE environment."""
    print(f"Creating environment: {env_name}")
    
    env = gym.make(env_name)
    
    if seed is not None:
        # Reset with seed for reproducibility
        env.reset(seed=seed)
    
    # Print environment information
    num_agvs = env.num_agvs if hasattr(env, 'num_agvs') else 0
    num_pickers = env.num_pickers if hasattr(env, 'num_pickers') else 0
    num_agents = env.num_agents if hasattr(env, 'num_agents') else len(env.agents)
    
    print(f"Number of AGVs: {num_agvs}")
    print(f"Number of Pickers: {num_pickers}")
    print(f"Total agents: {num_agents}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Grid size: {env.grid_size}")
    
    return env

def main():
    """Main function to run HSEAC training or evaluation."""
    args = parse_args()
    
    # Set random seeds
    set_seed(args.seed)
    
    # Setup experiment
    log_dir, save_dir = setup_experiment(args)
    
    # Create environment
    env = create_environment(args.env_name, args.seed)
    
    # Create HSEAC agent
    print("Initializing HSEAC agent...")
    agent = HSEAC(
        env=env,
        gamma=args.gamma,
        manager_lr=args.manager_lr,
        worker_lr=args.worker_lr,
        ent_coef=args.ent_coef,
        lambda_coef=args.lambda_coef,
        update_interval=args.update_interval,
        goal_completion_steps=args.goal_completion_steps,
        log_dir=log_dir
    )
    
    # Load existing model if path provided
    if args.load_path:
        try:
            agent.load(args.load_path)
            print(f"Successfully loaded model from {args.load_path}")
        except Exception as e:
            print(f"Failed to load model from {args.load_path}: {e}")
            if not args.train:  # If not training, exit on load failure
                return
    
    # Train the agents
    if args.train:
        print(f"\nTraining HSEAC on {args.env_name} for {args.n_episodes} episodes...")
        start_time = time.time()
        
        # Train agents
        rewards, eval_returns, eval_deliveries = agent.train(
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            render=args.render
        )
        
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        # Save the final model
        final_model_path = os.path.join(save_dir, "final_model")
        agent.save(final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        # Plot and save training curves
        plt.figure(figsize=(15, 5))
        
        # Training returns
        plt.subplot(1, 3, 1)
        plt.plot(range(args.n_episodes), rewards)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Training Returns')
        
        if eval_returns:
            # Evaluation returns
            plt.subplot(1, 3, 2)
            eval_interval = 100  # Assuming evaluation every 100 episodes
            plt.plot(range(eval_interval, args.n_episodes + 1, eval_interval), eval_returns)
            plt.xlabel('Episode')
            plt.ylabel('Evaluation Return')
            plt.title('Evaluation Returns')
            
            # Evaluation deliveries
            plt.subplot(1, 3, 3)
            plt.plot(range(eval_interval, args.n_episodes + 1, eval_interval), eval_deliveries)
            plt.xlabel('Episode')
            plt.ylabel('Deliveries')
            plt.title('Evaluation Deliveries')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "learning_curves.png"))
        
        # Print final statistics
        print("\nTraining Statistics:")
        print(f"Final training return (last episode): {rewards[-1]:.2f}")
        print(f"Average training return (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
        
        if eval_returns:
            print(f"Final evaluation return: {eval_returns[-1]:.2f}")
            print(f"Best evaluation return: {max(eval_returns):.2f}")
            print(f"Final evaluation deliveries: {eval_deliveries[-1]:.2f}")
            print(f"Best evaluation deliveries: {max(eval_deliveries):.2f}")
    
    # Evaluate the agents
    if args.evaluate:
        print(f"\nEvaluating agents over {args.eval_episodes} episodes...")
        eval_return, eval_deliveries = agent.evaluate(args.eval_episodes, render=args.render)
        print(f"Evaluation return: {eval_return:.2f}, Deliveries: {eval_deliveries:.2f}")
    
    # Close the environment
    env.close()
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()
