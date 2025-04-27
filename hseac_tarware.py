import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import time
from collections import deque, namedtuple
import tarware  # Import tarware instead of rware

# Named tuple for storing experience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'goal'])

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HierarchicalExperienceBuffer:
    """
    Buffer to store and retrieve hierarchical agent experiences.
    """
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.clear()
    
    def clear(self):
        """Reset the buffer by clearing all stored experiences."""
        self.manager_experiences = []
        self.worker_experiences = [[] for _ in range(self.n_agents)]
        self.size = 0
    
    def add_manager_experience(self, state, goals, reward, next_state, done, log_probs):
        """Add a manager transition to the buffer."""
        experience = {
            "state": state,
            "goals": goals,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "log_probs": log_probs
        }
        self.manager_experiences.append(experience)
        
    def add_worker_experience(self, states, goals, actions, rewards, next_states, done, log_probs):
        """Add worker transitions to the buffer."""
        for agent_idx in range(self.n_agents):
            experience = Experience(
                state=states[agent_idx],
                goal=goals[agent_idx],
                action=actions[agent_idx],
                reward=float(rewards[agent_idx]),
                next_state=next_states[agent_idx],
                done=done,
                log_prob=log_probs[agent_idx]
            )
            self.worker_experiences[agent_idx].append(experience)
        self.size += 1
    
    def get_manager_experiences(self):
        """Get all manager experiences."""
        return self.manager_experiences
    
    def get_worker_experiences(self, agent_idx):
        """Get all experiences for a specific worker agent."""
        return self.worker_experiences[agent_idx]
    
    def get_all_worker_experiences(self):
        """Get experiences for all worker agents."""
        return self.worker_experiences

def create_mlp(input_dim, output_dim, hidden_dims=[128, 128], activation=nn.ReLU):
    """
    Creates a Multi-Layer Perceptron with the specified architecture.
    """
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

class ManagerNetwork(nn.Module):
    """
    Manager network for hierarchical architecture that assigns goals to worker agents
    """
    def __init__(self, obs_dim, num_agents, num_zones, hidden_dims=[128, 128, 128]):
        super(ManagerNetwork, self).__init__()
        
        # Manager has a multi-headed network that outputs a goal (zone) for each agent
        self.shared_encoder = create_mlp(obs_dim, hidden_dims[-1], hidden_dims[:-1])
        
        # Create policy heads for each agent
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], num_zones) for _ in range(num_agents)
        ])
        
        # Value head for critic
        self.value_head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        """
        Forward pass through the manager network.
        Returns policy distributions for all agents and a value estimate.
        """
        shared_features = self.shared_encoder(x)
        
        # Get policy logits for each agent and convert to probabilities
        agent_policies = []
        for policy_head in self.policy_heads:
            logits = policy_head(shared_features)
            policy = F.softmax(logits, dim=-1)
            agent_policies.append(policy)
        
        # Get state value estimate
        value = self.value_head(shared_features)
        
        return agent_policies, value
    
    def get_goals(self, state, deterministic=False):
        """
        Select goals for all agents based on the policy.
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            policies, _ = self.forward(state_tensor)
            
            goals = []
            log_probs = []
            entropies = []
            
            for agent_idx, policy in enumerate(policies):
                policy = policy.cpu().squeeze()
                
                # Calculate entropy
                entropy = -(policy * torch.log(policy + 1e-10)).sum()
                
                if deterministic:
                    goal = torch.argmax(policy).item()
                else:
                    goal = torch.multinomial(policy, 1).item()
                
                # Get log probability of selected goal
                log_prob = torch.log(policy[goal] + 1e-10)
                
                goals.append(goal)
                log_probs.append(log_prob.item())
                entropies.append(entropy.item())
        
        return goals, log_probs, entropies

class WorkerActorCritic(nn.Module):
    """
    Worker Actor-Critic network for HSEAC algorithm
    """
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dims=[64, 64]):
        super(WorkerActorCritic, self).__init__()
        
        # Combine observation and goal dimensions
        combined_input_dim = obs_dim + goal_dim
        
        # Actor network (policy)
        self.actor = create_mlp(combined_input_dim, action_dim, hidden_dims)
        
        # Critic network (value function)
        self.critic = create_mlp(combined_input_dim, 1, hidden_dims)
    
    def forward(self, x, goal):
        """
        Forward pass through both networks.
        """
        # Combine observation and goal
        combined_input = torch.cat([x, goal], dim=-1)
        
        # Get action logits and convert to probabilities
        logits = self.actor(combined_input)
        policy = F.softmax(logits, dim=-1)
        
        # Get state value estimate
        value = self.critic(combined_input)
        
        return policy, value
    
    def get_action(self, state, goal, action_mask=None, deterministic=False):
        """
        Select an action based on the policy, conditioned on the goal.
        
        Args:
            state: Observation numpy array
            goal: Goal numpy array (one-hot encoded zone)
            action_mask: Boolean mask for valid actions (True = valid)
            deterministic: Whether to select actions deterministically
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            goal_tensor = torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(device)
            
            probs, _ = self.forward(state_tensor, goal_tensor)
            probs = probs.cpu().squeeze()
            
            # Apply action mask if provided
            if action_mask is not None:
                # Convert mask to tensor
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool)
                # Set probabilities of invalid actions to 0
                masked_probs = probs * mask_tensor.float()
                # Renormalize probabilities
                if masked_probs.sum() > 0:
                    probs = masked_probs / masked_probs.sum()
                else:
                    # If all actions are masked, use uniform distribution
                    print("[WARNING] All actions masked, using uniform distribution")
                    probs = torch.ones_like(probs) / probs.size(0)
            
            # Calculate distribution entropy for exploration bonus
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                # Sample action from distribution
                action = torch.multinomial(probs, 1).item()
            
            # Get log probability of selected action
            log_prob = torch.log(probs[action] + 1e-10)
        
        return action, log_prob.item(), entropy.item()
    
    def evaluate_actions(self, states, goals, actions, action_masks=None):
        """
        Evaluate log probabilities, values and entropy for given states, goals, and actions.
        """
        probs, values = self.forward(states, goals)
        
        # Apply action masks if provided
        if action_masks is not None:
            # Expand dimensions to match batch size
            if len(action_masks.shape) == 1:
                action_masks = action_masks.unsqueeze(0)
            
            # Apply masks to probabilities
            masked_probs = probs * action_masks.float()
            # Renormalize
            masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-10)
            probs = masked_probs
        
        # Gather action probabilities
        action_probs = probs.gather(1, actions)
        log_probs = torch.log(action_probs + 1e-10)
        
        # Calculate entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum(1, keepdim=True)
        
        return log_probs, values, entropy

def one_hot_encode(value, size):
    """
    Convert a value to a one-hot encoded vector.
    """
    one_hot = np.zeros(size, dtype=np.float32)
    if value < size:  # Ensure value is within valid range
        one_hot[value] = 1.0
    return one_hot

class HSEAC:
    """
    Hierarchical Shared Experience Actor-Critic implementation for TA-RWARE
    """
    def __init__(
        self, 
        env, 
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
        # Add annealing parameters
        det_action_start_prob=0.0,    # Starting probability of using deterministic actions
        det_action_final_prob=0.8,    # Final probability of using deterministic actions
        det_action_anneal_episodes=700,  # Episodes over which to anneal
        log_dir="runs/hseac_tarware",
        debug=False
    ):
        self.env = env
        self.gamma = gamma
        self.manager_lr = manager_lr
        self.worker_lr = worker_lr
        self.ent_coef = ent_coef
        self.lambda_coef = lambda_coef
        self.clip_importance_ratio = clip_importance_ratio
        self.update_interval = update_interval
        self.goal_completion_steps = goal_completion_steps
        self.debug = debug

        # Add deterministic action annealing parameters
        self.det_action_start_prob = det_action_start_prob
        self.det_action_final_prob = det_action_final_prob
        self.det_action_anneal_episodes = det_action_anneal_episodes
        self.det_action_prob = det_action_start_prob  # Current probability (will be updated)
        
        # Get agent counts
        self.n_agvs = self.env.num_agvs if hasattr(self.env, "num_agvs") else 0
        self.n_pickers = self.env.num_pickers if hasattr(self.env, "num_pickers") else 0
        self.n_agents = self.env.num_agents if hasattr(self.env, "num_agents") else len(self.env.agents)
        
        # Create zone mapping based on rack groups in the environment
        if hasattr(self.env, "rack_groups"):
            self.n_zones = len(self.env.rack_groups)
            self.zone_mapping = self._create_zone_mapping_from_rack_groups
        else:
            # If rack_groups not available, create a simple partitioning
            self.n_zones = max(10, self.n_agents * 2)  # Ensure enough zones
            self.zone_mapping = self._create_simple_zone_mapping
        
        # Validate that zone mapping is working properly for all zones
        self._validate_zone_mapping()
        
        # Setup tensorboard logging
        self.writer = SummaryWriter(log_dir)
        self._train_step = 0
        
        # Determine observation and action dimensions
        self.obs_dim = self._get_obs_dim()
        self.action_dim = self._get_action_dim()
        self.goal_dim = self.n_zones  # One-hot encoding of assigned zone
        
        # Create manager network
        self.manager_net = ManagerNetwork(
            self.obs_dim * self.n_agents,  # Manager sees all agents' observations
            self.n_agents,
            self.n_zones,
            manager_hidden_dims
        ).to(device)
        
        self.manager_optimizer = optim.AdamW(self.manager_net.parameters(), lr=manager_lr, weight_decay=0.01)

        self.manager_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.manager_optimizer, T_max=self.det_action_anneal_episodes, eta_min=1e-6)
        
        # Create worker networks for each agent
        self.worker_nets = []
        self.worker_optimizers = []
        self.worker_schedulers = []
        
        for i in range(self.n_agents):
            net = WorkerActorCritic(
                self.obs_dim, 
                self.goal_dim,  # One-hot encoded goal
                self.action_dim, 
                worker_hidden_dims
            ).to(device)
            
            self.worker_nets.append(net)
            
            # Optimizer for worker networks
            optimizer = optim.Adam(
                net.parameters(), 
                lr=worker_lr
            )
            self.worker_optimizers.append(optimizer)

            # Scheduler for worker networks (Cosine Annealing)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.det_action_anneal_episodes,  # Number of episodes to decay over
                eta_min=1e-6  # Small minimum learning rate
            )
            self.worker_schedulers.append(scheduler)
        
        # Create experience buffer
        self.buffer = HierarchicalExperienceBuffer(self.n_agents)
        
        # Track agent goals and completion status
        self.current_goals = [None] * self.n_agents
        self.goals_completed = [True] * self.n_agents  # Start with completed so new goals are assigned
        self.goal_steps = [0] * self.n_agents
        
        # For tracking previous states of agents
        self.prev_agent_states = [None] * self.n_agents
        
        # Initialize agent tracking info
        self._initialize_agent_states()
    
    def _validate_zone_mapping(self):
        """Validate that zone mapping is working for all zones"""
        invalid_zones = []
        for zone_id in range(self.n_zones):
            zone_locations = self.zone_mapping(zone_id)
            if not zone_locations:
                invalid_zones.append(zone_id)
        
        if invalid_zones:
            print(f"[WARNING] The following zones have no mapped locations: {invalid_zones}")
            # Create a fallback grid-based zone mapping for these zones
            self._create_fallback_zone_mapping(invalid_zones)
    
    def _create_fallback_zone_mapping(self, invalid_zones):
        """Create a fallback zone mapping by dividing the grid into sections"""
        print("[INFO] Creating fallback grid-based zone mapping")
        
        # Simple grid division - divide the warehouse into a grid of n_zones cells
        grid_height, grid_width = self.env.grid_size
        
        # Calculate approx dimensions for each zone
        zone_height = max(1, grid_height // int(np.sqrt(self.n_zones)))
        zone_width = max(1, grid_width // int(np.sqrt(self.n_zones)))
        
        # Store zone locations
        self.fallback_zone_mappings = {}
        
        for zone_id in invalid_zones:
            # Calculate zone boundaries 
            zone_row = (zone_id // int(np.sqrt(self.n_zones)))
            zone_col = (zone_id % int(np.sqrt(self.n_zones)))
            
            start_y = min(zone_row * zone_height, grid_height - 1)
            end_y = min(start_y + zone_height, grid_height)
            start_x = min(zone_col * zone_width, grid_width - 1)
            end_x = min(start_x + zone_width, grid_width)
            
            # Create list of all locations in this zone
            zone_locations = []
            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    # Skip highway locations if we can identify them
                    if hasattr(self.env, "_is_highway") and self.env._is_highway(x, y):
                        continue
                    zone_locations.append((x, y))
            
            # Store this mapping
            self.fallback_zone_mappings[zone_id] = zone_locations
            
            if self.debug:
                print(f"Zone {zone_id} fallback mapping: {len(zone_locations)} locations")
    
    def _initialize_agent_states(self):
        """Initialize tracking of agent states"""
        for i, agent in enumerate(self.env.agents):
            self.prev_agent_states[i] = {
                "carrying_shelf": agent.carrying_shelf is not None if hasattr(agent, "carrying_shelf") else False,
                "busy": agent.busy if hasattr(agent, "busy") else False,
                "x": agent.x,
                "y": agent.y
            }
    
    def _get_obs_dim(self):
        """Get the observation dimension for a single agent."""
        if isinstance(self.env.observation_space[0], gym.spaces.Box):
            return self.env.observation_space[0].shape[0]
        else:
            # If observation space is not a Box, flatten it
            return gym.spaces.flatdim(self.env.observation_space[0])
    
    def _get_action_dim(self):
        """Get the action dimension for a single agent."""
        if isinstance(self.env.action_space[0], gym.spaces.Discrete):
            return self.env.action_space[0].n
        else:
            # If action space is not Discrete, flatten it
            return gym.spaces.flatdim(self.env.action_space[0])
    
    def _create_zone_mapping_from_rack_groups(self, zone_id):
        """
        Create a mapping from zone IDs to locations based on rack_groups.
        
        Args:
            zone_id: Index of the zone to map
            
        Returns:
            List of (x,y) locations in the zone
        """
        # First check fallback mappings if they exist
        if hasattr(self, 'fallback_zone_mappings') and zone_id in self.fallback_zone_mappings:
            return self.fallback_zone_mappings[zone_id]
        
        if not hasattr(self.env, "rack_groups") or zone_id >= len(self.env.rack_groups):
            return []
        
        return self.env.rack_groups[zone_id]
    
    def _create_simple_zone_mapping(self, zone_id):
        """
        Create a simple mapping from zone IDs to warehouse locations.
        Used as a fallback when rack_groups is not available.
        
        Args:
            zone_id: Index of the zone to map
            
        Returns:
            List of (x,y) locations in the zone
        """
        # First check fallback mappings if they exist
        if hasattr(self, 'fallback_zone_mappings') and zone_id in self.fallback_zone_mappings:
            return self.fallback_zone_mappings[zone_id]
            
        if not hasattr(self.env, "action_id_to_coords_map"):
            return []
        
        # Get all non-goal locations
        all_locations = []
        for id_, coords in self.env.action_id_to_coords_map.items():
            if hasattr(self.env, "goals") and (coords[1], coords[0]) not in self.env.goals:
                all_locations.append((coords[1], coords[0]))
        
        # Divide locations into zones based on zone_id
        n_zones = self.n_zones
        locations_per_zone = max(1, len(all_locations) // n_zones)
        zone_start = zone_id * locations_per_zone
        zone_end = min(len(all_locations), (zone_id + 1) * locations_per_zone)
        
        # Return locations for the requested zone
        if zone_id < n_zones and zone_start < len(all_locations):
            return all_locations[zone_start:zone_end]
        return []
    
    def preprocess_obs(self, obs):
        """
        Convert observations to the format needed by the networks.
        """
        processed_obs = []
        for agent_idx in range(self.n_agents):
            if isinstance(self.env.observation_space[agent_idx], gym.spaces.Box):
                processed_obs.append(obs[agent_idx])
            else:
                # Flatten the observation
                flat_obs = gym.spaces.flatten(
                    self.env.observation_space[agent_idx], 
                    obs[agent_idx]
                )
                processed_obs.append(flat_obs)
        return processed_obs
    
    def is_goal_completed(self, agent_idx, obs, goal):
        """
        Check if an agent's goal has been completed.
        """
        # Get the agent's information
        agent = self.env.agents[agent_idx]
        
        # Check if the agent has delivered a shelf and RESET the flag
        if hasattr(agent, "has_delivered") and agent.has_delivered:
            # Reset the flag so it doesn't keep triggering goal completion
            agent.has_delivered = False
            if self.debug:
                print(f"Agent {agent_idx} goal completed: delivered shelf and reset flag")
            return True
        
        # Other existing checks remain unchanged
        # Basic check: has the agent been working on this goal for too long?
        if self.goal_steps[agent_idx] >= self.goal_completion_steps:
            if self.debug:
                print(f"Agent {agent_idx} goal completed: exceeded max steps ({self.goal_steps[agent_idx]})")
            return True
        
        # Check if agent is in the assigned zone
        if goal is not None:
            zone_locations = self.zone_mapping(goal)
            if (agent.x, agent.y) in zone_locations:
                # If we're in the zone for several steps, consider it reached
                if self.goal_steps[agent_idx] > 10:
                    if self.debug:
                        print(f"Agent {agent_idx} goal completed: in zone {goal} for {self.goal_steps[agent_idx]} steps")
                    return True
        
        # Check if agent was carrying a shelf but now isn't (delivered or returned)
        prev_state = self.prev_agent_states[agent_idx]
        if prev_state and prev_state.get("carrying_shelf", False) and not agent.carrying_shelf:
            if self.debug:
                print(f"Agent {agent_idx} goal completed: shelf unloaded")
            return True
        
        # Check if the agent was busy but now is not (likely completed a task)
        if prev_state and prev_state.get("busy", False) and not agent.busy:
            if self.debug:
                print(f"Agent {agent_idx} goal completed: busy -> not busy")
            return True
            
        return False
    
    def get_valid_actions(self, agent_idx, goal):
        """
        Get mask of valid actions for an agent within its assigned zone.
        """
        # If no goal assigned, all actions are valid (will be handled by env's action mask)
        if goal is None:
            return np.ones(self.action_dim, dtype=bool)
        
        # Get valid action masks from the environment if available
        if hasattr(self.env, "compute_valid_action_masks"):
            valid_action_masks = self.env.compute_valid_action_masks()
            return valid_action_masks[agent_idx]
        
        # Otherwise, create action mask based on zone assignment
        action_mask = np.zeros(self.action_dim, dtype=bool)
        
        # Always allow "no-op" (action 0)
        action_mask[0] = True
        
        # Get locations in the assigned zone
        zone_locations = self.zone_mapping(goal)
        
        # Map zone locations to action indices
        if hasattr(self.env, "action_id_to_coords_map"):
            coords_to_id = {v: k for k, v in self.env.action_id_to_coords_map.items()}
            for loc in zone_locations:
                # Check if we need to swap coordinates
                loc_tuple = (loc[0], loc[1])
                if loc_tuple in coords_to_id:
                    action_id = coords_to_id[loc_tuple]
                elif (loc[1], loc[0]) in coords_to_id:  # Try swapped coordinates
                    action_id = coords_to_id[(loc[1], loc[0])]
                else:
                    continue
                
                if action_id < self.action_dim:
                    action_mask[action_id] = True
        
        # Make sure we have at least one valid action
        if not np.any(action_mask):
            if self.debug:
                print(f"[WARNING] No valid actions in zone {goal} for agent {agent_idx}, using all actions")
            return np.ones(self.action_dim, dtype=bool)
        
        return action_mask
    
    def update_manager(self):
        """
        Update the manager's policy based on collected experiences.
        """
        if len(self.buffer.manager_experiences) == 0:
            return
        
        # Prepare data for network update
        states = np.array([e["state"] for e in self.buffer.manager_experiences], dtype=np.float32)
        goals = [e["goals"] for e in self.buffer.manager_experiences]
        rewards = np.array([e["reward"] for e in self.buffer.manager_experiences], dtype=np.float32)
        next_states = np.array([e["next_state"] for e in self.buffer.manager_experiences], dtype=np.float32)
        dones = np.array([e["done"] for e in self.buffer.manager_experiences], dtype=np.float32)
        log_probs_list = [e["log_probs"] for e in self.buffer.manager_experiences]
        
        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
        
        # Compute TD errors and advantages
        with torch.no_grad():
            _, next_values = self.manager_net.forward(next_states_tensor)
            _, values = self.manager_net.forward(states_tensor)
            
            next_values = next_values.squeeze()
            values = values.squeeze()
            
            # Compute TD targets and advantages
            td_targets = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)
            advantages = td_targets - values
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update manager policy
        total_policy_loss = 0
        total_value_loss = 0
        
        # Policy and value loss computation
        for agent_idx in range(self.n_agents):
            agent_log_probs = torch.tensor([log_probs[agent_idx] for log_probs in log_probs_list],
                                        dtype=torch.float32).to(device)
            
            # Policy loss
            policy_loss = -(agent_log_probs * advantages.detach())
            total_policy_loss += policy_loss.mean()
            
        # Value loss
        value_loss = F.mse_loss(values, td_targets.detach())
        total_value_loss = value_loss
        
        # Total loss (average policy loss across agents)
        total_policy_loss /= self.n_agents
        
        # Apply entropy bonus
        policies_list, _ = self.manager_net.forward(states_tensor)
        entropy_loss = 0
        
        for policies in policies_list:
            entropy = -(policies * torch.log(policies + 1e-10)).sum(dim=1).mean()
            entropy_loss += entropy
        
        entropy_loss /= self.n_agents
        
        # Total loss
        total_loss = total_policy_loss + total_value_loss - self.ent_coef * entropy_loss
        
        # Update manager network
        self.manager_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager_net.parameters(), 0.5)
        self.manager_optimizer.step()
        
        # Log metrics
        self.writer.add_scalar('manager/policy_loss', total_policy_loss.item(), self._train_step)
        self.writer.add_scalar('manager/value_loss', total_value_loss.item(), self._train_step)
        self.writer.add_scalar('manager/entropy', entropy_loss.item(), self._train_step)
    
    def update_workers(self):
        """
        Update all worker agents' networks using collected experiences with experience sharing.
        """
        # Get experiences for all agents
        all_experiences = self.buffer.get_all_worker_experiences()
        
        # Update each agent
        for agent_idx in range(self.n_agents):
            # Get agent's own experiences
            agent_experiences = all_experiences[agent_idx]
            
            if len(agent_experiences) == 0:
                continue
            
            # Prepare data for network update
            states = np.array([e.state for e in agent_experiences], dtype=np.float32)
            goals = np.array([e.goal for e in agent_experiences], dtype=np.float32)
            actions = np.array([e.action for e in agent_experiences], dtype=np.int64)
            rewards = np.array([e.reward for e in agent_experiences], dtype=np.float32)
            next_states = np.array([e.next_state for e in agent_experiences], dtype=np.float32)
            dones = np.array([float(e.done) for e in agent_experiences], dtype=np.float32)
            log_probs = np.array([e.log_prob for e in agent_experiences], dtype=np.float32)
            
            # Convert to tensors
            states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            goals_tensor = torch.tensor(goals, dtype=torch.float32).to(device)
            actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
            log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32).to(device)
            
            # Compute TD errors and advantages
            with torch.no_grad():
                _, next_values = self.worker_nets[agent_idx].forward(next_states_tensor, goals_tensor)
                _, values = self.worker_nets[agent_idx].forward(states_tensor, goals_tensor)
                
                next_values = next_values.squeeze()
                values = values.squeeze()
                
                # Compute TD targets and advantages
                td_targets = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)
                advantages = td_targets - values
                
                # Normalize advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Get action masks for state-goal pairs if environment supports it
            action_masks = None
            if hasattr(self.env, "compute_valid_action_masks"):
                # Note: We're using the same action mask for all experiences
                # In a full implementation, you'd store the masks with the experiences
                action_masks = torch.tensor(self.get_valid_actions(agent_idx, agent_experiences[0].goal), 
                                          dtype=torch.bool).to(device)
            
            # Get current log probs, values, and entropy
            current_log_probs, current_values, entropy = self.worker_nets[agent_idx].evaluate_actions(
                states_tensor, goals_tensor, actions_tensor, action_masks
            )
            
            # Compute policy (actor) loss for own experiences
            policy_loss = -(current_log_probs * advantages.detach()).mean()
            
            # Compute value (critic) loss for own experiences
            value_loss = F.mse_loss(current_values.squeeze(), td_targets)
            
            # Initialize shared experience loss components
            shared_policy_loss = 0.0
            shared_value_loss = 0.0
            
            # Add shared experience from other agents
            shared_experiences_count = 0
            
            for other_idx in range(self.n_agents):
                if other_idx == agent_idx:
                    continue
                
                other_experiences = all_experiences[other_idx]
                
                for exp in other_experiences:
                    # Only share experiences with the same goal/zone
                    if not np.array_equal(exp.goal, agent_experiences[0].goal):
                        continue
                        
                    # Compute importance sampling ratio
                    with torch.no_grad():
                        other_state = torch.tensor(exp.state, dtype=torch.float32).unsqueeze(0).to(device)
                        other_goal = torch.tensor(exp.goal, dtype=torch.float32).unsqueeze(0).to(device)
                        other_action = torch.tensor([exp.action], dtype=torch.int64).unsqueeze(1).to(device)
                        
                        # Get probabilities from both policies
                        own_probs, _ = self.worker_nets[agent_idx].forward(other_state, other_goal)
                        own_prob = own_probs[0, exp.action].item()
                        
                        other_prob = np.exp(exp.log_prob)  # Convert log prob to prob
                        
                        # Compute ratio with clipping
                        importance_ratio = own_prob / (other_prob + 1e-10)
                        importance_ratio = np.clip(importance_ratio, 0, self.clip_importance_ratio)
                        importance_ratio_tensor = torch.tensor([importance_ratio], dtype=torch.float32).to(device)
                    
                    # Compute shared advantage
                    with torch.no_grad():
                        _, own_value = self.worker_nets[agent_idx].forward(other_state, other_goal)
                        next_own_value = 0.0
                        
                        if not exp.done:
                            next_state = torch.tensor(exp.next_state, dtype=torch.float32).unsqueeze(0).to(device)
                            _, next_own_value = self.worker_nets[agent_idx].forward(next_state, other_goal)
                            next_own_value = next_own_value.item()
                        
                        shared_adv = exp.reward + self.gamma * next_own_value * (1.0 - float(exp.done)) - own_value.item()
                    
                    # Get current policy's log prob for this experience
                    action_mask = None
                    if hasattr(self.env, "compute_valid_action_masks"):
                        action_mask = torch.tensor(self.get_valid_actions(agent_idx, exp.goal), 
                                                 dtype=torch.bool).unsqueeze(0).to(device)
                    
                    own_log_prob, own_value, _ = self.worker_nets[agent_idx].evaluate_actions(
                        other_state, other_goal, other_action, action_mask
                    )
                    
                    # Target value
                    target_value = torch.tensor([exp.reward + self.gamma * next_own_value * (1.0 - float(exp.done))], 
                                            dtype=torch.float32).to(device)
                    
                    # Add to shared losses
                    shared_policy_loss += importance_ratio_tensor * (-own_log_prob) * shared_adv
                    shared_value_loss += importance_ratio_tensor * 0.5 * (own_value.squeeze() - target_value) ** 2
                    
                    shared_experiences_count += 1
            
            # Average shared losses
            if shared_experiences_count > 0:
                shared_policy_loss = shared_policy_loss / shared_experiences_count
                shared_value_loss = shared_value_loss / shared_experiences_count
                
                # Apply lambda coefficient to shared losses
                shared_policy_loss *= self.lambda_coef
                shared_value_loss *= self.lambda_coef
            else:
                shared_policy_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                shared_value_loss = torch.tensor(0.0, dtype=torch.float32).to(device)
            
            # Combine own and shared losses
            total_policy_loss = policy_loss + shared_policy_loss.mean() - self.ent_coef * entropy.mean()
            total_value_loss = value_loss + shared_value_loss.mean()
            
            # Total loss
            total_loss = total_policy_loss + total_value_loss
            
            # Update worker network
            self.worker_optimizers[agent_idx].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.worker_nets[agent_idx].parameters(), 0.5)
            self.worker_optimizers[agent_idx].step()
            self.worker_schedulers[agent_idx].step()
            
            # Log metrics
            self.writer.add_scalar(f'worker_{agent_idx}/policy_loss', policy_loss.item(), self._train_step)
            self.writer.add_scalar(f'worker_{agent_idx}/value_loss', value_loss.item(), self._train_step)
            self.writer.add_scalar(f'worker_{agent_idx}/shared_policy_loss', 
                                shared_policy_loss.mean().item() if shared_experiences_count > 0 else 0, 
                                self._train_step)
            self.writer.add_scalar(f'worker_{agent_idx}/shared_value_loss', 
                                shared_value_loss.mean().item() if shared_experiences_count > 0 else 0, 
                                self._train_step)
            self.writer.add_scalar(f'worker_{agent_idx}/entropy', entropy.mean().item(), self._train_step)
    
    def train(self, n_episodes, max_steps=500, render=False):
        """
        Train the agents using HSEAC algorithm.
        """
        self._train_step = 0
        episode_rewards = []
        eval_returns = []
        eval_deliveries = []
        
        for episode in range(n_episodes):
            # Update deterministic action probability based on training progress
            if episode < self.det_action_anneal_episodes:
                progress = episode / self.det_action_anneal_episodes
                self.det_action_prob = self.det_action_start_prob + progress * (self.det_action_final_prob - self.det_action_start_prob)
            else:
                self.det_action_prob = self.det_action_final_prob
                
            # Log the current probability
            self.writer.add_scalar('training/det_action_prob', self.det_action_prob, episode)

            obs = self.env.reset()
            obs = self.preprocess_obs(obs)
            episode_reward = 0
            episode_shelf_deliveries = 0
            
            # Clear buffer for new episode
            self.buffer.clear()
            
            # Reset goals for new episode
            self.current_goals = [None] * self.n_agents
            self.goals_completed = [True] * self.n_agents
            self.goal_steps = [0] * self.n_agents
            
            # Update agent states tracking
            self._initialize_agent_states()
            
            # Combined observation for manager (all agents' observations)
            manager_obs = np.concatenate(obs)
            
            for step in range(max_steps):
                if render:
                    self.env.render()
                
                # Update previous agent states
                for i, agent in enumerate(self.env.agents):
                    self.prev_agent_states[i] = {
                        "carrying_shelf": agent.carrying_shelf is not None if hasattr(agent, "carrying_shelf") else False,
                        "busy": agent.busy if hasattr(agent, "busy") else False,
                        "x": agent.x,
                        "y": agent.y
                    }
                
                # Determine which agents need new goals
                needs_goal = [completed for completed in self.goals_completed]
                
                # If any agent needs a goal, get new goals from manager
                if any(needs_goal):
                    use_deterministic = random.random() < self.det_action_prob
                    goals, goal_log_probs, _ = self.manager_net.get_goals(manager_obs, deterministic=use_deterministic)
                    
                    # Assign goals only to agents that need them
                    for agent_idx in range(self.n_agents):
                        if needs_goal[agent_idx]:
                            # Validate goal index before assignment
                            goal_idx = goals[agent_idx]
                            if goal_idx is None or goal_idx >= self.n_zones:
                                if self.debug:
                                    print(f"[WARNING] Invalid goal index {goal_idx} for agent {agent_idx}, assigning None")
                                self.current_goals[agent_idx] = None
                                self.goals_completed[agent_idx] = True
                            else:
                                self.current_goals[agent_idx] = goal_idx
                                self.goals_completed[agent_idx] = False
                                self.goal_steps[agent_idx] = 0
                                
                                if self.debug:
                                    print(f"Agent {agent_idx} assigned to zone {goal_idx}")
                
                # Select actions for each agent based on their current goals
                actions = []
                log_probs = []
                one_hot_goals = []
                
                for agent_idx in range(self.n_agents):
                    # One-hot encode the goal for the worker network
                    if self.current_goals[agent_idx] is not None:
                        one_hot_goal = one_hot_encode(self.current_goals[agent_idx], self.n_zones)
                    else:
                        # If no goal assigned yet, use zeros
                        one_hot_goal = np.zeros(self.n_zones, dtype=np.float32)
                    
                    one_hot_goals.append(one_hot_goal)
                    
                    # Get valid actions within assigned zone
                    valid_actions = self.get_valid_actions(agent_idx, self.current_goals[agent_idx])
                    
                    # Use deterministic actions with increasing probability during training
                    use_deterministic = random.random() < self.det_action_prob
                    
                    # Select action within assigned zone
                    action, log_prob, _ = self.worker_nets[agent_idx].get_action(
                        obs[agent_idx], one_hot_goal, valid_actions, deterministic=use_deterministic
                    )
                    
                    actions.append(action)
                    log_probs.append(log_prob)
                
                # Execute actions in environment
                next_obs, rewards, terminated, truncated, info = self.env.step(tuple(actions))
                next_obs = self.preprocess_obs(next_obs)
                
                # Track shelf deliveries if available in info
                if 'shelf_deliveries' in info:
                    episode_shelf_deliveries += info['shelf_deliveries']
                
                # Check if any agent has completed its goal
                for agent_idx in range(self.n_agents):
                    # Increment step counter for this goal
                    self.goal_steps[agent_idx] += 1
                    
                    # Check if goal is completed
                    if not self.goals_completed[agent_idx] and self.is_goal_completed(agent_idx, next_obs, self.current_goals[agent_idx]):
                        self.goals_completed[agent_idx] = True
                        
                        if self.debug:
                            print(f"Agent {agent_idx} completed goal in zone {self.current_goals[agent_idx]}")
                
                # Combined observation for manager
                next_manager_obs = np.concatenate(next_obs)
                
                # Store experience
                goal_reward = sum(rewards)  # Manager gets sum of all worker rewards
                
                # Determine if episode is done (all agents are done)
                done = all(terminated) or all(truncated)
                
                # Only store manager experience if we assigned goals this step
                if any(needs_goal):
                    self.buffer.add_manager_experience(
                        manager_obs, [self.current_goals[i] for i in range(self.n_agents)], 
                        goal_reward, next_manager_obs, done, goal_log_probs
                    )
                
                # Store worker experiences
                self.buffer.add_worker_experience(
                    obs, one_hot_goals, actions, rewards, next_obs, done, log_probs
                )
                
                # Update observations
                obs = next_obs
                manager_obs = next_manager_obs
                
                # Accumulate episode reward
                episode_reward += sum(rewards)
                
                # Update networks if we have enough steps
                if step % self.update_interval == 0 and step > 0:
                    self.update_manager()
                    self.update_workers()
                    self._train_step += 1
                    
                    # Clear buffer after update
                    self.buffer.clear()
                
                # Check if episode is done
                if done or step == max_steps - 1:
                    break
            
            # Final update at end of episode if buffer is not empty
            if self.buffer.size > 0:
                self.update_manager()
                self.update_workers()
                self._train_step += 1
            
            # Store episode reward
            episode_rewards.append(episode_reward)
            
            # Log episode metrics
            self.writer.add_scalar('training/episode_reward', episode_reward, episode)
            self.writer.add_scalar('training/episode_length', step + 1, episode)
            self.writer.add_scalar('training/shelf_deliveries', episode_shelf_deliveries, episode)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/{n_episodes}, Average Reward (Last 10): {avg_reward:.2f}")
                
            # Evaluate every 100 episodes
            if (episode + 1) % 100 == 0:
                eval_return, eval_delivery = self.evaluate(5, render=False)
                eval_returns.append(eval_return)
                eval_deliveries.append(eval_delivery)
                
                # Log to tensorboard
                self.writer.add_scalar('evaluation/return', eval_return, episode)
                self.writer.add_scalar('evaluation/deliveries', eval_delivery, episode)
                
                print(f"Evaluation at episode {episode + 1}: Reward = {eval_return:.2f}, Deliveries = {eval_delivery:.2f}")
                
                # Save model checkpoint
                os.makedirs("checkpoints", exist_ok=True)
                self.save(f"checkpoints/hseac_tarware_episode_{episode+1}")
        
        # Close tensorboard writer
        self.writer.close()
        
        return episode_rewards, eval_returns, eval_deliveries
    
    def evaluate(self, n_episodes, render=False):
        """
        Evaluate the agent's performance.
        Returns both average return and average shelf deliveries.
        """
        total_returns = 0
        total_shelf_deliveries = 0
        
        for _ in range(n_episodes):
            obs = self.env.reset()
            obs = self.preprocess_obs(obs)
            episode_return = 0
            episode_shelf_deliveries = 0
            
            # Reset goals for new episode
            self.current_goals = [None] * self.n_agents
            self.goals_completed = [True] * self.n_agents
            self.goal_steps = [0] * self.n_agents
            
            # Update agent states tracking
            self._initialize_agent_states()
            
            # Combined observation for manager
            manager_obs = np.concatenate(obs)
            
            step = 0
            max_steps = 500  # Limit evaluation episodes
            
            while step < max_steps:
                if render:
                    self.env.render()
                
                # Update previous agent states
                for i, agent in enumerate(self.env.agents):
                    self.prev_agent_states[i] = {
                        "carrying_shelf": agent.carrying_shelf is not None if hasattr(agent, "carrying_shelf") else False,
                        "busy": agent.busy if hasattr(agent, "busy") else False,
                        "x": agent.x,
                        "y": agent.y
                    }
                
                # Determine which agents need new goals
                needs_goal = [completed for completed in self.goals_completed]
                
                # If any agent needs a goal, get new goals from manager
                if any(needs_goal):
                    goals, _, _ = self.manager_net.get_goals(manager_obs, deterministic=True)
                    
                    # Assign goals only to agents that need them
                    for agent_idx in range(self.n_agents):
                        if needs_goal[agent_idx]:
                            # Validate goal index before assignment
                            goal_idx = goals[agent_idx]
                            if goal_idx is None or goal_idx >= self.n_zones:
                                if self.debug:
                                    print(f"[WARNING] Invalid goal index {goal_idx} for agent {agent_idx}, assigning None")
                                self.current_goals[agent_idx] = None
                                self.goals_completed[agent_idx] = True
                            else:
                                self.current_goals[agent_idx] = goal_idx
                                self.goals_completed[agent_idx] = False
                                self.goal_steps[agent_idx] = 0
                
                # Select actions for each agent based on their current goals
                actions = []
                
                for agent_idx in range(self.n_agents):
                    # One-hot encode the goal for the worker network
                    if self.current_goals[agent_idx] is not None:
                        one_hot_goal = one_hot_encode(self.current_goals[agent_idx], self.n_zones)
                    else:
                        # If no goal assigned yet, use zeros
                        one_hot_goal = np.zeros(self.n_zones, dtype=np.float32)
                    
                    # Get valid actions within assigned zone
                    valid_actions = self.get_valid_actions(agent_idx, self.current_goals[agent_idx])
                    
                    # Select action deterministically
                    action, _, _ = self.worker_nets[agent_idx].get_action(
                        obs[agent_idx], one_hot_goal, valid_actions, deterministic=True
                    )
                    actions.append(action)
                
                # Execute actions
                next_obs, rewards, terminated, truncated, info = self.env.step(tuple(actions))
                next_obs = self.preprocess_obs(next_obs)
                
                # Track shelf deliveries if available
                if 'shelf_deliveries' in info:
                    episode_shelf_deliveries += info['shelf_deliveries']
                
                # Check if any agent has completed its goal
                for agent_idx in range(self.n_agents):
                    # Increment step counter for this goal
                    self.goal_steps[agent_idx] += 1
                    
                    # Check if goal is completed
                    if not self.goals_completed[agent_idx] and self.is_goal_completed(agent_idx, next_obs, self.current_goals[agent_idx]):
                        self.goals_completed[agent_idx] = True
                
                # Update observations
                obs = next_obs
                manager_obs = np.concatenate(next_obs)
                
                # Accumulate episode return
                episode_return += sum(rewards)
                
                # Check if episode is done or max steps reached
                if all(terminated) or all(truncated) or step == max_steps - 1:
                    if self.debug and (all(terminated) or all(truncated)):
                        print(f"Evaluation episode terminated/truncated at step {step}")
                    break
                
                step += 1
            
            total_returns += episode_return
            total_shelf_deliveries += episode_shelf_deliveries
            
            if self.debug:
                print(f"Evaluation episode: Return = {episode_return:.2f}, Deliveries = {episode_shelf_deliveries}")
        
        avg_return = total_returns / n_episodes
        avg_deliveries = total_shelf_deliveries / n_episodes
        
        if self.debug:
            print(f"Evaluation results: Avg Return = {avg_return:.2f}, Avg Deliveries = {avg_deliveries:.2f}")
        
        return avg_return, avg_deliveries
    
    def save(self, path):
        """
        Save the agent's parameters.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save manager network
        torch.save(
            self.manager_net.state_dict(), 
            f"{path}/manager.pt"
        )
        
        # Save worker networks
        for agent_idx in range(self.n_agents):
            torch.save(
                self.worker_nets[agent_idx].state_dict(), 
                f"{path}/worker_{agent_idx}.pt"
            )
    
    def load(self, path):
        """
        Load the agent's parameters.
        """
        try:
            # Load manager network
            self.manager_net.load_state_dict(
                torch.load(f"{path}/manager.pt", map_location=device)
            )
            
            # Load worker networks
            for agent_idx in range(self.n_agents):
                worker_path = f"{path}/worker_{agent_idx}.pt"
                if os.path.exists(worker_path):
                    self.worker_nets[agent_idx].load_state_dict(
                        torch.load(worker_path, map_location=device)
                    )
                else:
                    print(f"[WARNING] Worker network file not found: {worker_path}")
            
            print(f"Successfully loaded model from {path}")
            return True
        
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return False