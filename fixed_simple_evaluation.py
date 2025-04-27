import argparse
import os
import sys
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import csv
import tarware
import contextlib

from hseac_tarware import HSEAC
from seac_tarware import EqualizedSEAC

# Create a context manager to suppress stdout temporarily
@contextlib.contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = original_stdout

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced evaluation script for HSEAC and SEAC')
    
    parser.add_argument('--env_name', type=str, default='tarware-tiny-3agvs-2pickers-globalobs-v1',
                      help='Name of the environment')
    parser.add_argument('--hseac_model_path', type=str, default='comparison_results/models/hseac_model',
                      help='Path to HSEAC model')
    parser.add_argument('--seac_model_path', type=str, default='comparison_results/models/equalized_seac_model',
                      help='Path to SEAC model')
    parser.add_argument('--num_episodes', type=int, default=100,
                      help='Number of episodes for evaluation')
    parser.add_argument('--output_dir', type=str, default='enhanced_eval_results',
                      help='Directory to save results')
    parser.add_argument('--quiet', action='store_true',
                      help='Reduce terminal output')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    
    return parser.parse_args()

def one_hot_encode(value, size):
    """Convert a value to a one-hot encoded vector."""
    one_hot = np.zeros(size, dtype=np.float32)
    if value < size:  # Ensure value is within valid range
        one_hot[value] = 1.0
    return one_hot

def evaluate_model(model_type, model, env, num_episodes, quiet=False, debug=False):
    """Evaluate a model on the given environment."""
    print(f"\nEvaluating {model_type} for {num_episodes} episodes...")
    
    episode_returns = []
    episode_steps = []
    shelf_deliveries = []
    
    for episode in range(num_episodes):
        # Suppress stdout for HSEAC if quiet mode is enabled
        context = suppress_stdout() if model_type == 'HSEAC' and quiet else contextlib.nullcontext()
        
        with context:
            # Reset environment
            obs = env.reset()
            episode_return = 0
            episode_delivery = 0
            step = 0
            
            if model_type == 'HSEAC':
                # Reset HSEAC agent state
                model.current_goals = [None] * model.n_agents
                model.goals_completed = [True] * model.n_agents
                model.goal_steps = [0] * model.n_agents
                
                # Initialize agent tracking
                model._initialize_agent_states()
            
            done = False
            while not done:
                # Get actions based on model type
                if model_type == 'HSEAC':
                    # Preprocess observations
                    proc_obs = model.preprocess_obs(obs)
                    
                    # Get manager observation
                    try:
                        # Make sure observations are numpy arrays
                        flat_obs = []
                        for o in proc_obs:
                            if isinstance(o, np.ndarray):
                                flat_obs.append(o.flatten())
                            else:
                                flat_obs.append(np.array(o, dtype=np.float32).flatten())
                        
                        manager_obs = np.concatenate(flat_obs)
                    except Exception as e:
                        if debug:
                            print(f"Error concatenating observations: {e}")
                        break
                    
                    # Determine which agents need goals
                    needs_goal = model.goals_completed
                    
                    # Get goals from manager
                    if any(needs_goal):
                        goals, _, _ = model.manager_net.get_goals(manager_obs, deterministic=True)
                        
                        # Assign goals to agents
                        for agent_idx in range(model.n_agents):
                            if needs_goal[agent_idx]:
                                # Validate goal index
                                goal_idx = goals[agent_idx]
                                if goal_idx is None or goal_idx >= model.n_zones:
                                    model.current_goals[agent_idx] = None
                                    model.goals_completed[agent_idx] = True
                                else:
                                    model.current_goals[agent_idx] = goal_idx
                                    model.goals_completed[agent_idx] = False
                                    model.goal_steps[agent_idx] = 0
                    
                    # Get actions from workers
                    actions = []
                    for agent_idx in range(model.n_agents):
                        # One-hot encode the goal
                        if model.current_goals[agent_idx] is not None:
                            # Use the standalone one_hot_encode function
                            one_hot_goal = one_hot_encode(model.current_goals[agent_idx], model.n_zones)
                        else:
                            one_hot_goal = np.zeros(model.n_zones, dtype=np.float32)
                        
                        # Get valid actions
                        valid_actions = model.get_valid_actions(agent_idx, model.current_goals[agent_idx])
                        
                        # Get action
                        action, _, _ = model.worker_nets[agent_idx].get_action(
                            proc_obs[agent_idx], one_hot_goal, valid_actions, deterministic=True
                        )
                        actions.append(action)
                    
                else:  # SEAC
                    # Preprocess observations
                    proc_obs = model.preprocess_obs(obs)
                    
                    # Get actions
                    actions = []
                    for agent_idx in range(model.n_agents):
                        # Get valid actions
                        valid_actions = model.get_valid_actions(agent_idx)
                        
                        # Get action
                        action, _, _ = model.actor_critics[agent_idx].get_action(
                            proc_obs[agent_idx], valid_actions, deterministic=True
                        )
                        actions.append(action)
                
                # Execute actions
                try:
                    next_obs, rewards, terminated, truncated, info = env.step(tuple(actions))
                except Exception as e:
                    if debug:
                        print(f"Error executing actions: {e}")
                    break
                
                # Check if done
                done = all(terminated) or all(truncated)
                
                # Track shelf deliveries
                if 'shelf_deliveries' in info:
                    episode_delivery += info['shelf_deliveries']
                
                # Update agent goals for HSEAC
                if model_type == 'HSEAC':
                    # Update previous agent states
                    for i, agent_obj in enumerate(env.agents):
                        model.prev_agent_states[i] = {
                            "carrying_shelf": agent_obj.carrying_shelf is not None if hasattr(agent_obj, "carrying_shelf") else False,
                            "busy": agent_obj.busy if hasattr(agent_obj, "busy") else False,
                            "x": agent_obj.x,
                            "y": agent_obj.y
                        }
                    
                    # Update goal completion
                    for agent_idx in range(model.n_agents):
                        model.goal_steps[agent_idx] += 1
                        
                        if not model.goals_completed[agent_idx] and model.is_goal_completed(agent_idx, next_obs, model.current_goals[agent_idx]):
                            model.goals_completed[agent_idx] = True
                
                # Update state
                obs = next_obs
                episode_return += sum(rewards)
                step += 1
                
                # Limit episode length
                if step >= 500:
                    done = True
        
        # Record episode data
        episode_returns.append(episode_return)
        episode_steps.append(step)
        shelf_deliveries.append(episode_delivery)
        
        # Print progress update (always outside the suppression context)
        if (episode + 1) % 10 == 0:
            print(f"{model_type} - Episode {episode + 1}/{num_episodes}: Return = {episode_return:.2f}, Steps = {step}, Deliveries = {episode_delivery}")
    
    # Calculate metrics
    metrics = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_deliveries': np.mean(shelf_deliveries),
        'std_deliveries': np.std(shelf_deliveries),
        'total_deliveries': np.sum(shelf_deliveries),
        'mean_steps': np.mean(episode_steps),
    }
    
    print(f"\n{model_type} Evaluation Results:")
    print(f"Mean Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"Mean Deliveries: {metrics['mean_deliveries']:.2f} ± {metrics['std_deliveries']:.2f}")
    print(f"Total Deliveries: {metrics['total_deliveries']}")
    print(f"Mean Steps: {metrics['mean_steps']:.2f}")
    
    return episode_returns, shelf_deliveries, episode_steps, metrics

def save_metrics_to_csv(output_dir, hseac_metrics, seac_metrics):
    """Save metrics to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'comparison_metrics.csv')
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Metric', 'HSEAC', 'SEAC', 'Difference', 'Improvement (%)'])
        
        # Write metrics
        metrics_to_compare = [
            'mean_return', 'std_return', 'min_return', 'max_return', 
            'mean_deliveries', 'std_deliveries', 'total_deliveries', 'mean_steps'
        ]
        
        for metric in metrics_to_compare:
            hseac_value = hseac_metrics[metric]
            seac_value = seac_metrics[metric]
            difference = hseac_value - seac_value
            
            # Calculate improvement percentage
            if seac_value != 0:
                improvement = (difference / abs(seac_value)) * 100
                improvement_str = f"{improvement:.2f}%"
            else:
                improvement_str = "N/A"
                
            writer.writerow([metric, hseac_value, seac_value, difference, improvement_str])
    
    print(f"Metrics saved to {csv_path}")

def save_episode_data_to_csv(output_dir, hseac_data, seac_data):
    """Save episode data to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'episode_data.csv')
    
    hseac_returns, hseac_deliveries, hseac_steps = hseac_data
    seac_returns, seac_deliveries, seac_steps = seac_data
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Episode', 'HSEAC_Return', 'HSEAC_Deliveries', 'HSEAC_Steps', 
                         'SEAC_Return', 'SEAC_Deliveries', 'SEAC_Steps'])
        
        # Write episode data
        max_episodes = max(len(hseac_returns), len(seac_returns))
        for i in range(max_episodes):
            hseac_return = hseac_returns[i] if i < len(hseac_returns) else ""
            hseac_delivery = hseac_deliveries[i] if i < len(hseac_deliveries) else ""
            hseac_step = hseac_steps[i] if i < len(hseac_steps) else ""
            seac_return = seac_returns[i] if i < len(seac_returns) else ""
            seac_delivery = seac_deliveries[i] if i < len(seac_deliveries) else ""
            seac_step = seac_steps[i] if i < len(seac_steps) else ""
            
            writer.writerow([i+1, hseac_return, hseac_delivery, hseac_step,
                            seac_return, seac_delivery, seac_step])
    
    print(f"Episode data saved to {csv_path}")

def create_enhanced_plots(output_dir, hseac_data, seac_data, hseac_metrics, seac_metrics):
    """Create enhanced comparison plots with improved aesthetics and additional visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    hseac_returns, hseac_deliveries, hseac_steps = hseac_data
    seac_returns, seac_deliveries, seac_steps = seac_data
    
    # Set up professional plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define a modern color palette
    colors = {
        'hseac': '#4C72B0',  # Blue
        'seac': '#DD8452',   # Orange
        'bg': '#F5F5F5',     # Light gray background
        'grid': '#DDDDDD',   # Grid lines
        'text': '#333333'    # Text color
    }
    
    # Set font properties
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    episodes = range(1, len(hseac_deliveries) + 1)
    
    # 1. Improved Performance Metrics Plot
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Comparison: HSEAC vs SEAC', fontsize=20, fontweight='bold', y=0.98)
    fig.patch.set_facecolor(colors['bg'])
    
    # Returns histogram
    ax[0, 0].hist(hseac_returns, bins=15, alpha=0.7, color=colors['hseac'], label='HSEAC')
    ax[0, 0].hist(seac_returns, bins=15, alpha=0.7, color=colors['seac'], label='SEAC')
    ax[0, 0].set_xlabel('Episode Return')
    ax[0, 0].set_ylabel('Frequency')
    ax[0, 0].set_title('Return Distribution')
    ax[0, 0].legend(frameon=True, facecolor=colors['bg'])
    ax[0, 0].grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
    
    # Deliveries histogram
    ax[0, 1].hist(hseac_deliveries, bins=10, alpha=0.7, color=colors['hseac'], label='HSEAC')
    ax[0, 1].hist(seac_deliveries, bins=10, alpha=0.7, color=colors['seac'], label='SEAC')
    ax[0, 1].set_xlabel('Shelf Deliveries per Episode')
    ax[0, 1].set_ylabel('Frequency')
    ax[0, 1].set_title('Deliveries Distribution')
    ax[0, 1].legend(frameon=True, facecolor=colors['bg'])
    ax[0, 1].grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
    
    # Mean returns bar chart
    methods = ['HSEAC', 'SEAC']
    means = [hseac_metrics['mean_return'], seac_metrics['mean_return']]
    stds = [hseac_metrics['std_return'], seac_metrics['std_return']]
    
    bars = ax[1, 0].bar(methods, means, yerr=stds, alpha=0.8, color=[colors['hseac'], colors['seac']], 
             error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
    ax[1, 0].set_ylabel('Mean Return')
    ax[1, 0].set_title('Average Performance')
    ax[1, 0].grid(True, linestyle='--', alpha=0.7, color=colors['grid'], axis='y')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax[1, 0].text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.1,
                f'{means[i]:.2f} ± {stds[i]:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Mean deliveries bar chart
    mean_deliveries = [hseac_metrics['mean_deliveries'], seac_metrics['mean_deliveries']]
    std_deliveries = [hseac_metrics['std_deliveries'], seac_metrics['std_deliveries']]
    
    bars = ax[1, 1].bar(methods, mean_deliveries, yerr=std_deliveries, alpha=0.8, 
             color=[colors['hseac'], colors['seac']], 
             error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
    ax[1, 1].set_ylabel('Mean Deliveries per Episode')
    ax[1, 1].set_title('Average Deliveries')
    ax[1, 1].grid(True, linestyle='--', alpha=0.7, color=colors['grid'], axis='y')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax[1, 1].text(bar.get_x() + bar.get_width()/2., height + std_deliveries[i] + 0.1,
                f'{mean_deliveries[i]:.2f} ± {std_deliveries[i]:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"Performance metrics plot saved to {os.path.join(output_dir, 'performance_metrics.png')}")
    
    # 2. Enhanced Deliveries Per Episode Plot
    plt.figure(figsize=(14, 8))
    fig = plt.gcf()
    fig.patch.set_facecolor(colors['bg'])
    
    # Create a smoother line using moving average
    window_size = 5
    hseac_smooth = np.convolve(hseac_deliveries, np.ones(window_size)/window_size, mode='valid')
    seac_smooth = np.convolve(seac_deliveries, np.ones(window_size)/window_size, mode='valid')
    smooth_episodes = range(window_size, len(hseac_deliveries) + 1)
    
    # Plot raw data as scatter points
    plt.scatter(episodes, hseac_deliveries, s=30, alpha=0.4, color=colors['hseac'], label='HSEAC (raw)')
    plt.scatter(episodes, seac_deliveries, s=30, alpha=0.4, color=colors['seac'], label='SEAC (raw)')
    
    # Plot smoothed lines
    plt.plot(smooth_episodes, hseac_smooth, linewidth=3, color=colors['hseac'], label=f'HSEAC (moving avg n={window_size})')
    plt.plot(smooth_episodes, seac_smooth, linewidth=3, color=colors['seac'], label=f'SEAC (moving avg n={window_size})')
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Deliveries per Episode', fontsize=14)
    plt.title('Deliveries per Episode over Time with Moving Average', fontsize=18, fontweight='bold')
    plt.legend(loc='upper left', frameon=True, facecolor=colors['bg'])
    plt.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
    
    # Add improvement percentage annotation
    hseac_avg = np.mean(hseac_deliveries)
    seac_avg = np.mean(seac_deliveries)
    improvement_pct = ((hseac_avg - seac_avg) / seac_avg) * 100 if seac_avg != 0 else 0
    
    plt.annotate(f'HSEAC outperforms SEAC by {improvement_pct:.1f}% on average',
                xy=(len(episodes) * 0.5, max(max(hseac_deliveries), max(seac_deliveries)) * 0.9),
                bbox=dict(boxstyle="round,pad=0.5", fc=colors['bg'], alpha=0.8),
                fontsize=14, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deliveries_per_episode_enhanced.png'), dpi=300, bbox_inches='tight')
    print(f"Enhanced deliveries per episode plot saved to {os.path.join(output_dir, 'deliveries_per_episode_enhanced.png')}")
    
    # 3. Improved Cumulative Deliveries Plot
    plt.figure(figsize=(14, 8))
    fig = plt.gcf()
    fig.patch.set_facecolor(colors['bg'])
    
    hseac_cumulative = np.cumsum(hseac_deliveries)
    seac_cumulative = np.cumsum(seac_deliveries)
    
    # Create a gradient fill under the lines
    plt.fill_between(episodes, hseac_cumulative, alpha=0.2, color=colors['hseac'])
    plt.fill_between(episodes, seac_cumulative, alpha=0.2, color=colors['seac'])
    
    plt.plot(episodes, hseac_cumulative, linewidth=3, color=colors['hseac'], label='HSEAC')
    plt.plot(episodes, seac_cumulative, linewidth=3, color=colors['seac'], label='SEAC')
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Cumulative Deliveries', fontsize=14)
    plt.title('Cumulative Deliveries over Time', fontsize=18, fontweight='bold')
    plt.legend(loc='upper left', frameon=True, facecolor=colors['bg'])
    plt.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
    
    # Add final totals annotation
    final_hseac = hseac_cumulative[-1]
    final_seac = seac_cumulative[-1]
    difference = final_hseac - final_seac
    pct_diff = (difference / final_seac) * 100 if final_seac != 0 else 0
    
    plt.annotate(f'HSEAC Total: {final_hseac}',
                xy=(len(episodes), final_hseac),
                xytext=(-20, 10),
                textcoords='offset points',
                fontsize=14, fontweight='bold', color=colors['hseac'],
                ha='right')
    
    plt.annotate(f'SEAC Total: {final_seac}',
                xy=(len(episodes), final_seac),
                xytext=(-20, 10),
                textcoords='offset points',
                fontsize=14, fontweight='bold', color=colors['seac'],
                ha='right')
    
    plt.annotate(f'Difference: {difference} ({pct_diff:.1f}% improvement)',
                xy=(len(episodes) * 0.6, (final_hseac + final_seac) / 2),
                bbox=dict(boxstyle="round,pad=0.5", fc=colors['bg'], alpha=0.8),
                fontsize=14, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_deliveries_enhanced.png'), dpi=300, bbox_inches='tight')
    print(f"Enhanced cumulative deliveries plot saved to {os.path.join(output_dir, 'cumulative_deliveries_enhanced.png')}")
    
    # 4. Steps per Delivery Efficiency Plot
    plt.figure(figsize=(14, 8))
    fig = plt.gcf()
    fig.patch.set_facecolor(colors['bg'])
    
    # Calculate steps per delivery (efficiency)
    hseac_efficiency = [s/d if d > 0 else float('nan') for s, d in zip(hseac_steps, hseac_deliveries)]
    seac_efficiency = [s/d if d > 0 else float('nan') for s, d in zip(seac_steps, seac_deliveries)]
    
    # Remove NaN values for plotting
    hseac_valid_eps = [ep for ep, eff in zip(episodes, hseac_efficiency) if not np.isnan(eff)]
    seac_valid_eps = [ep for ep, eff in zip(episodes, seac_efficiency) if not np.isnan(eff)]
    hseac_valid_eff = [eff for eff in hseac_efficiency if not np.isnan(eff)]
    seac_valid_eff = [eff for eff in seac_efficiency if not np.isnan(eff)]
    
    plt.scatter(hseac_valid_eps, hseac_valid_eff, s=40, alpha=0.6, color=colors['hseac'], label='HSEAC')
    plt.scatter(seac_valid_eps, seac_valid_eff, s=40, alpha=0.6, color=colors['seac'], label='SEAC')
    
    # Add trendlines
    from scipy import stats
    
    if len(hseac_valid_eps) > 1:
        slope, intercept, _, _, _ = stats.linregress(hseac_valid_eps, hseac_valid_eff)
        plt.plot(hseac_valid_eps, [intercept + slope * ep for ep in hseac_valid_eps], 
                '--', color=colors['hseac'], alpha=0.8, linewidth=2)
    
    if len(seac_valid_eps) > 1:
        slope, intercept, _, _, _ = stats.linregress(seac_valid_eps, seac_valid_eff)
        plt.plot(seac_valid_eps, [intercept + slope * ep for ep in seac_valid_eps], 
                '--', color=colors['seac'], alpha=0.8, linewidth=2)
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Steps per Delivery (lower is better)', fontsize=14)
    plt.title('Agent Efficiency: Steps Needed per Successful Delivery', fontsize=18, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, facecolor=colors['bg'])
    plt.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
    
    # Add average values
    hseac_avg_eff = np.nanmean(hseac_efficiency)
    seac_avg_eff = np.nanmean(seac_efficiency)
    
    plt.axhline(y=hseac_avg_eff, color=colors['hseac'], linestyle='-', alpha=0.5)
    plt.axhline(y=seac_avg_eff, color=colors['seac'], linestyle='-', alpha=0.5)
    
    plt.annotate(f'HSEAC Avg: {hseac_avg_eff:.2f} steps/delivery',
                xy=(len(episodes) * 0.02, hseac_avg_eff),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=12, fontweight='bold', color=colors['hseac'])
    
    plt.annotate(f'SEAC Avg: {seac_avg_eff:.2f} steps/delivery',
                xy=(len(episodes) * 0.02, seac_avg_eff),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=12, fontweight='bold', color=colors['seac'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steps_per_delivery_efficiency.png'), dpi=300, bbox_inches='tight')
    print(f"Steps per delivery efficiency plot saved to {os.path.join(output_dir, 'steps_per_delivery_efficiency.png')}")
    
    # Additional visualization plots
    try:
        create_additional_plots(output_dir, hseac_data, seac_data, hseac_metrics, seac_metrics)
    except Exception as e:
        print(f"Error creating additional plots: {e}")


def create_additional_plots(output_dir, hseac_data, seac_data, hseac_metrics, seac_metrics):
    """Create additional suggested plots that provide deeper insights."""
    os.makedirs(output_dir, exist_ok=True)
    
    hseac_returns, hseac_deliveries, hseac_steps = hseac_data
    seac_returns, seac_deliveries, seac_steps = seac_data
    
    episodes = range(1, len(hseac_deliveries) + 1)
    
    # Set up professional plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define a modern color palette
    colors = {
        'hseac': '#4C72B0',  # Blue
        'seac': '#DD8452',   # Orange
        'bg': '#F5F5F5',     # Light gray background
        'grid': '#DDDDDD',   # Grid lines
        'text': '#333333'    # Text color
    }
    
    # Set font properties
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # 1. Box/Violin Plot Comparison
    plt.figure(figsize=(16, 8))
    fig = plt.gcf()
    fig.patch.set_facecolor(colors['bg'])
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Detailed Distribution Comparison: HSEAC vs SEAC', fontsize=20, fontweight='bold')
    fig.patch.set_facecolor(colors['bg'])
    
    # Prepare data for violin plots
    metrics = [
        (hseac_returns, seac_returns, 'Episode Returns', 'Return'),
        (hseac_deliveries, seac_deliveries, 'Deliveries per Episode', 'Deliveries'),
        (hseac_steps, seac_steps, 'Steps per Episode', 'Steps')
    ]
    
    # Create violin plots
    for i, (hseac_data, seac_data, title, ylabel) in enumerate(metrics):
        # Create violin plot
        parts = axs[i].violinplot(
            [hseac_data, seac_data],
            showmeans=False,
            showmedians=True
        )
        
        # Customize violin colors
        for pc, color in zip(parts['bodies'], [colors['hseac'], colors['seac']]):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        # Add box plot inside violin for additional statistics
        box_parts = axs[i].boxplot(
            [hseac_data, seac_data],
            positions=[1, 2],
            widths=0.15,
            patch_artist=True,
            showfliers=False,  # Hide outliers as they're shown in the violin
            medianprops={'color': 'black', 'linewidth': 2},
            boxprops={'color': 'black', 'linewidth': 1}
        )
        
        # Set box colors
        for box, color in zip(box_parts['boxes'], [colors['hseac'], colors['seac']]):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        # Add scatter points for individual data points with jitter
        for j, data in enumerate([hseac_data, seac_data]):
            # Add some jitter to x-position
            x = np.random.normal(j+1, 0.05, size=len(data))
            axs[i].scatter(x, data, alpha=0.4, s=15, 
                         color=colors['hseac'] if j == 0 else colors['seac'])
        
        # Set labels and title
        axs[i].set_title(title)
        axs[i].set_ylabel(ylabel)
        axs[i].set_xticks([1, 2])
        axs[i].set_xticklabels(['HSEAC', 'SEAC'])
        axs[i].grid(True, linestyle='--', alpha=0.7, color=colors['grid'], axis='y')
        
        # Add statistics as annotations
        stats = [
            (np.mean(hseac_data), np.median(hseac_data), np.std(hseac_data)),
            (np.mean(seac_data), np.median(seac_data), np.std(seac_data))
        ]
        
        for j, (mean, median, std) in enumerate(stats):
            y_pos = np.max([hseac_data, seac_data]) * 0.95
            axs[i].annotate(f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}',
                           xy=(j+1, y_pos),
                           ha='center', va='top',
                           bbox=dict(boxstyle="round,pad=0.3", fc=colors['bg'], alpha=0.8),
                           color=colors['hseac'] if j == 0 else colors['seac'],
                           fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'violin_plot_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Violin plot comparison saved to {os.path.join(output_dir, 'violin_plot_comparison.png')}")
    
    # 2. Joint Distribution Plot (Returns vs Deliveries)
    plt.figure(figsize=(12, 10))
    fig = plt.gcf()
    fig.patch.set_facecolor(colors['bg'])
    
    # Create scatter plot
    plt.scatter(hseac_deliveries, hseac_returns, s=70, alpha=0.7, 
               color=colors['hseac'], label='HSEAC', edgecolor='white')
    plt.scatter(seac_deliveries, seac_returns, s=70, alpha=0.7, 
               color=colors['seac'], label='SEAC', edgecolor='white')
    
    plt.xlabel('Deliveries per Episode', fontsize=14)
    plt.ylabel('Episode Return', fontsize=14)
    plt.title('Joint Distribution: Returns vs Deliveries', fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
    plt.legend(frameon=True, facecolor=colors['bg'])
    
    # Add regression lines
    from scipy import stats
    
    # HSEAC regression line
    slope, intercept, r_value, _, _ = stats.linregress(hseac_deliveries, hseac_returns)
    x_range = np.linspace(min(min(hseac_deliveries), min(seac_deliveries)),
                         max(max(hseac_deliveries), max(seac_deliveries)), 100)
    plt.plot(x_range, intercept + slope * x_range, '--', 
            color=colors['hseac'], linewidth=2, 
            label=f'HSEAC (r={r_value:.2f})')
    
    # SEAC regression line
    slope, intercept, r_value, _, _ = stats.linregress(seac_deliveries, seac_returns)
    plt.plot(x_range, intercept + slope * x_range, '--', 
            color=colors['seac'], linewidth=2, 
            label=f'SEAC (r={r_value:.2f})')
    
    plt.legend(frameon=True, facecolor=colors['bg'])
    
    # Try to add density contours if possible
    try:
        from scipy.stats import gaussian_kde
        
        # HSEAC density
        if len(hseac_deliveries) > 2:  # Need at least 3 points for KDE
            xy_hseac = np.vstack([hseac_deliveries, hseac_returns])
            kde_hseac = gaussian_kde(xy_hseac)
            
            # Create a grid and compute the density
            xgrid = np.linspace(min(hseac_deliveries), max(hseac_deliveries), 100)
            ygrid = np.linspace(min(hseac_returns), max(hseac_returns), 100)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            Z_hseac = kde_hseac(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
            Z_hseac = Z_hseac.reshape(Xgrid.shape)
            
            # Plot contours
            plt.contour(Xgrid, Ygrid, Z_hseac, colors=colors['hseac'], alpha=0.5, levels=5)
        
        # SEAC density
        if len(seac_deliveries) > 2:  # Need at least 3 points for KDE
            xy_seac = np.vstack([seac_deliveries, seac_returns])
            kde_seac = gaussian_kde(xy_seac)
            
            # Create a grid and compute the density
            xgrid = np.linspace(min(seac_deliveries), max(seac_deliveries), 100)
            ygrid = np.linspace(min(seac_returns), max(seac_returns), 100)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            Z_seac = kde_seac(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
            Z_seac = Z_seac.reshape(Xgrid.shape)
            
            # Plot contours
            plt.contour(Xgrid, Ygrid, Z_seac, colors=colors['seac'], alpha=0.5, levels=5)
    except Exception as e:
        # Skip density contours if there's an error
        print(f"Skipping density contours due to error: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_distribution_returns_deliveries.png'), dpi=300, bbox_inches='tight')
    print(f"Joint distribution plot saved to {os.path.join(output_dir, 'joint_distribution_returns_deliveries.png')}")
    
    # 3. Radar/Spider Chart comparing key metrics
    plt.figure(figsize=(10, 10))
    fig = plt.gcf()
    fig.patch.set_facecolor(colors['bg'])
    
    # Define categories and values
    categories = ['Mean Return', 'Mean Deliveries', 'Success Rate', 'Efficiency', 'Consistency']
    
    # Normalize all metrics to 0-1 scale for comparison
    max_return = max(hseac_metrics['mean_return'], seac_metrics['mean_return'])
    norm_hseac_return = hseac_metrics['mean_return'] / max_return if max_return > 0 else 0
    norm_seac_return = seac_metrics['mean_return'] / max_return if max_return > 0 else 0
    
    max_deliveries = max(hseac_metrics['mean_deliveries'], seac_metrics['mean_deliveries'])
    norm_hseac_deliveries = hseac_metrics['mean_deliveries'] / max_deliveries if max_deliveries > 0 else 0
    norm_seac_deliveries = seac_metrics['mean_deliveries'] / max_deliveries if max_deliveries > 0 else 0
    
    # Success rate (% of episodes with at least one delivery)
    hseac_success = sum(1 for d in hseac_deliveries if d > 0) / len(hseac_deliveries)
    seac_success = sum(1 for d in seac_deliveries if d > 0) / len(seac_deliveries)
    
    # Efficiency (inverse of steps per delivery, higher is better)
    hseac_efficiency = [s/d if d > 0 else float('nan') for s, d in zip(hseac_steps, hseac_deliveries)]
    seac_efficiency = [s/d if d > 0 else float('nan') for s, d in zip(seac_steps, seac_deliveries)]
    hseac_avg_steps_per_delivery = np.nanmean(hseac_efficiency) if not np.all(np.isnan(hseac_efficiency)) else float('inf')
    seac_avg_steps_per_delivery = np.nanmean(seac_efficiency) if not np.all(np.isnan(seac_efficiency)) else float('inf')
    
    # Normalize efficiency (higher is better, so invert)
    max_steps = max(hseac_avg_steps_per_delivery, seac_avg_steps_per_delivery)
    norm_hseac_efficiency = 1 - (hseac_avg_steps_per_delivery / max_steps) if max_steps < float('inf') and max_steps > 0 else 0
    norm_seac_efficiency = 1 - (seac_avg_steps_per_delivery / max_steps) if max_steps < float('inf') and max_steps > 0 else 0
    
    # Consistency (inverse of standard deviation, higher is better)
    max_std_deliveries = max(hseac_metrics['std_deliveries'], seac_metrics['std_deliveries'])
    norm_hseac_consistency = 1 - (hseac_metrics['std_deliveries'] / max_std_deliveries) if max_std_deliveries > 0 else 0
    norm_seac_consistency = 1 - (seac_metrics['std_deliveries'] / max_std_deliveries) if max_std_deliveries > 0 else 0
    
    # Combine all metrics
    hseac_values = [norm_hseac_return, norm_hseac_deliveries, hseac_success, 
                   norm_hseac_efficiency, norm_hseac_consistency]
    seac_values = [norm_seac_return, norm_seac_deliveries, seac_success, 
                  norm_seac_efficiency, norm_seac_consistency]
    
    # Close the polygon by appending the first value
    hseac_values.append(hseac_values[0])
    seac_values.append(seac_values[0])
    categories.append(categories[0])
    
    # Compute angle for each category
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles.append(angles[0])  # Close the polygon
    
    # Create radar chart
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor(colors['bg'])
    
    # Plot data
    ax.plot(angles, hseac_values, 'o-', linewidth=2, color=colors['hseac'], label='HSEAC')
    ax.fill(angles, hseac_values, color=colors['hseac'], alpha=0.25)
    
    ax.plot(angles, seac_values, 'o-', linewidth=2, color=colors['seac'], label='SEAC')
    ax.fill(angles, seac_values, color=colors['seac'], alpha=0.25)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], fontsize=14)
    
    # Remove radial labels and set grid
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, facecolor=colors['bg'])
    
    plt.title('Performance Radar Chart: HSEAC vs SEAC', fontsize=18, fontweight='bold', y=1.08)
    
    # Add actual metric values as annotations
    ann_text = (
        f"HSEAC Metrics:\n"
        f"• Mean Return: {hseac_metrics['mean_return']:.2f}\n"
        f"• Mean Deliveries: {hseac_metrics['mean_deliveries']:.2f}\n"
        f"• Success Rate: {hseac_success:.1%}\n"
        f"• Steps/Delivery: {hseac_avg_steps_per_delivery:.2f}\n"
        f"• Std Deviation: {hseac_metrics['std_deliveries']:.2f}\n\n"
        f"SEAC Metrics:\n"
        f"• Mean Return: {seac_metrics['mean_return']:.2f}\n"
        f"• Mean Deliveries: {seac_metrics['mean_deliveries']:.2f}\n"
        f"• Success Rate: {seac_success:.1%}\n"
        f"• Steps/Delivery: {seac_avg_steps_per_delivery:.2f}\n"
        f"• Std Deviation: {seac_metrics['std_deliveries']:.2f}"
    )
    
    plt.annotate(ann_text, xy=(0.97, 0.03), xycoords='figure fraction',
                ha='right', va='bottom', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc=colors['bg'], alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Radar chart comparison saved to {os.path.join(output_dir, 'radar_chart_comparison.png')}")

def one_hot_encode(value, size):
    """Convert a value to a one-hot encoded vector."""
    one_hot = np.zeros(size, dtype=np.float32)
    if value is not None and value < size:  # Ensure value is within valid range
        one_hot[value] = 1.0
    return one_hot

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environments
    env_hseac = gym.make(args.env_name)
    env_seac = gym.make(args.env_name)
    
    # Load models
    print("Loading HSEAC model...")
    hseac_agent = HSEAC(env=env_hseac, debug=False)  # Force debug off to reduce output
    if not hseac_agent.load(args.hseac_model_path):
        print(f"Failed to load HSEAC model from {args.hseac_model_path}")
        return
    
    print("Loading SEAC model...")
    seac_agent = EqualizedSEAC(env=env_seac, debug=False)  # Force debug off to reduce output
    if not seac_agent.load(args.seac_model_path):
        print(f"Failed to load SEAC model from {args.seac_model_path}")
        return
    
    # Evaluate models
    hseac_returns, hseac_deliveries, hseac_steps, hseac_metrics = evaluate_model(
        'HSEAC', hseac_agent, env_hseac, args.num_episodes, quiet=args.quiet, debug=args.debug
    )
    
    seac_returns, seac_deliveries, seac_steps, seac_metrics = evaluate_model(
        'SEAC', seac_agent, env_seac, args.num_episodes, quiet=args.quiet, debug=args.debug
    )
    
    # Save results
    save_metrics_to_csv(
        args.output_dir, 
        hseac_metrics, 
        seac_metrics
    )
    
    save_episode_data_to_csv(
        args.output_dir,
        (hseac_returns, hseac_deliveries, hseac_steps),
        (seac_returns, seac_deliveries, seac_steps)
    )
    
    create_enhanced_plots(
        args.output_dir,
        (hseac_returns, hseac_deliveries, hseac_steps),
        (seac_returns, seac_deliveries, seac_steps),
        hseac_metrics,
        seac_metrics
    )
    
    # Close environments
    env_hseac.close()
    env_seac.close()
    
    print("\nEvaluation complete!")
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main()