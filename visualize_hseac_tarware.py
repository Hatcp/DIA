"""
Script to visualize trained HSEAC agents in the TA-RWARE environment.
"""
import argparse
import time
import os
import numpy as np
import gymnasium as gym
import torch
from hseac_tarware import HSEAC
import tarware

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize HSEAC agent in TA-RWARE environment')
    parser.add_argument('--env_name', type=str, default='tarware-tiny-3agvs-2pickers-globalobs-v1',
                        help='Name of the environment')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between steps (seconds)')
    parser.add_argument('--record', action='store_true',
                        help='Record videos of the episodes')
    parser.add_argument('--output_dir', type=str, default='videos',
                        help='Directory to save videos')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()

def create_environment(env_name):
    """Create the environment without render_mode parameter"""
    try:
        env = gym.make(env_name)
        return env
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise

def load_agent(env, model_path, debug=False):
    """Load the trained HSEAC agent"""
    try:
        # Create agent with the same parameters as during training
        agent = HSEAC(
            env=env,
            gamma=0.99,
            manager_lr=3e-4,
            worker_lr=3e-4,
            ent_coef=0.01,
            lambda_coef=1.0,
            manager_hidden_dims=[128, 128, 128],
            worker_hidden_dims=[64, 64],
            clip_importance_ratio=5.0,
            update_interval=100,
            goal_completion_steps=20,
            debug=debug
        )
        
        # Load trained model weights
        success = agent.load(model_path)
        if not success:
            raise ValueError(f"Failed to load model from {model_path}")
        
        return agent
    except Exception as e:
        print(f"Error loading agent: {e}")
        raise

def one_hot_encode(value, size):
    """Convert a value to a one-hot encoded vector."""
    one_hot = np.zeros(size, dtype=np.float32)
    if value < size:  # Ensure value is within valid range
        one_hot[value] = 1.0
    return one_hot

def visualize_episodes(env, agent, num_episodes=5, delay=0.1, debug=False):
    """Visualize the trained agent for a number of episodes"""
    total_rewards = []
    total_steps = []
    total_deliveries = []
    
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode+1}/{num_episodes}")
        
        obs = env.reset()
        obs = agent.preprocess_obs(obs)
        episode_reward = 0
        step_count = 0
        deliveries = 0
        
        # Reset goals
        current_goals = [None] * agent.n_agents
        goals_completed = [True] * agent.n_agents
        goal_steps = [0] * agent.n_agents
        
        # Initialize agent states tracking
        prev_agent_states = [None] * agent.n_agents
        for i, agent_entity in enumerate(env.agents):
            prev_agent_states[i] = {
                "carrying_shelf": agent_entity.carrying_shelf is not None if hasattr(agent_entity, "carrying_shelf") else False,
                "busy": agent_entity.busy if hasattr(agent_entity, "busy") else False,
                "x": agent_entity.x,
                "y": agent_entity.y
            }
        
        # Combined observation for manager
        manager_obs = np.concatenate(obs)
        
        while True:
            # Render the environment
            env.render()
            time.sleep(delay)  # Add delay for visualization
            
            # Update previous agent states
            for i, agent_entity in enumerate(env.agents):
                prev_agent_states[i] = {
                    "carrying_shelf": agent_entity.carrying_shelf is not None if hasattr(agent_entity, "carrying_shelf") else False,
                    "busy": agent_entity.busy if hasattr(agent_entity, "busy") else False,
                    "x": agent_entity.x,
                    "y": agent_entity.y
                }
            
            # Determine which agents need new goals
            needs_goal = [completed for completed in goals_completed]
            
            # If any agent needs a goal, get new goals from manager
            if any(needs_goal):
                goals, _, _ = agent.manager_net.get_goals(manager_obs, deterministic=True)
                
                # Assign goals only to agents that need them
                for agent_idx in range(agent.n_agents):
                    if needs_goal[agent_idx]:
                        # Validate goal index before assignment
                        goal_idx = goals[agent_idx]
                        if goal_idx is None or goal_idx >= agent.n_zones:
                            if debug:
                                print(f"[WARNING] Invalid goal index {goal_idx} for agent {agent_idx}, assigning None")
                            current_goals[agent_idx] = None
                            goals_completed[agent_idx] = True
                        else:
                            current_goals[agent_idx] = goal_idx
                            goals_completed[agent_idx] = False
                            goal_steps[agent_idx] = 0
                            
                            if debug:
                                print(f"Agent {agent_idx} assigned to zone {goal_idx}")
            
            # Select actions for each agent based on their current goals
            actions = []
            
            for agent_idx in range(agent.n_agents):
                # One-hot encode the goal for the worker network
                if current_goals[agent_idx] is not None:
                    one_hot_goal = one_hot_encode(current_goals[agent_idx], agent.n_zones)
                else:
                    # If no goal assigned yet, use zeros
                    one_hot_goal = np.zeros(agent.n_zones, dtype=np.float32)
                
                # Get valid actions within assigned zone
                valid_actions = agent.get_valid_actions(agent_idx, current_goals[agent_idx])
                
                # Select action deterministically
                action, _, _ = agent.worker_nets[agent_idx].get_action(
                    obs[agent_idx], one_hot_goal, valid_actions, deterministic=True
                )
                actions.append(action)
            
            # Execute actions
            next_obs, rewards, terminated, truncated, info = env.step(tuple(actions))
            next_obs = agent.preprocess_obs(next_obs)
            
            # Track shelf deliveries if available
            if 'shelf_deliveries' in info:
                deliveries += info['shelf_deliveries']
                if info['shelf_deliveries'] > 0 and debug:
                    print(f"Step {step_count}: {info['shelf_deliveries']} shelves delivered!")
            
            # Check if any agent has completed its goal
            for agent_idx in range(agent.n_agents):
                # Increment step counter for this goal
                goal_steps[agent_idx] += 1
                
                # Check if goal is completed
                if not goals_completed[agent_idx] and agent.is_goal_completed(agent_idx, next_obs, current_goals[agent_idx]):
                    goals_completed[agent_idx] = True
                    
                    if debug:
                        print(f"Agent {agent_idx} completed goal in zone {current_goals[agent_idx]}")
            
            # Update observations
            obs = next_obs
            manager_obs = np.concatenate(next_obs)
            
            # Accumulate episode reward
            episode_reward += sum(rewards)
            step_count += 1
            
            # Check if episode is done
            if all(terminated) or all(truncated) or step_count >= 500:
                if debug:
                    print(f"Episode ended after {step_count} steps")
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(step_count)
        total_deliveries.append(deliveries)
        
        print(f"Episode {episode+1} - Reward: {episode_reward:.2f}, Steps: {step_count}, Deliveries: {deliveries}")
    
    # Print summary statistics
    print("\nVisualization Summary:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Steps: {np.mean(total_steps):.2f} ± {np.std(total_steps):.2f}")
    print(f"Average Deliveries: {np.mean(total_deliveries):.2f} ± {np.std(total_deliveries):.2f}")
    
    return np.mean(total_rewards), np.mean(total_deliveries)

def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Creating environment: {args.env_name}")
    env = create_environment(args.env_name)
    
    print(f"Loading agent from: {args.model_path}")
    agent = load_agent(env, args.model_path, debug=args.debug)
    
    print(f"Visualizing {args.num_episodes} episodes with {args.delay}s delay...")
    avg_reward, avg_deliveries = visualize_episodes(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        delay=args.delay,
        debug=args.debug
    )
    
    print("\nVisualization complete!")
    env.close()

if __name__ == "__main__":
    main()