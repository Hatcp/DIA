import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import time
import random
from collections import deque, namedtuple
import tarware  # Import tarware

# Named tuple for storing experience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob'])

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperienceBuffer:
    """
    Buffer to store and retrieve agent experiences.
    """
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.clear()
    
    def clear(self):
        """Reset the buffer by clearing all stored experiences."""
        self.experiences = [[] for _ in range(self.n_agents)]
        self.size = 0
    
    def add_experience(self, states, actions, rewards, next_states, done, log_probs):
        """Add transitions to the buffer."""
        for agent_idx in range(self.n_agents):
            experience = Experience(
                state=states[agent_idx],
                action=actions[agent_idx],
                reward=float(rewards[agent_idx]),
                next_state=next_states[agent_idx],
                done=done,
                log_prob=log_probs[agent_idx]
            )
            self.experiences[agent_idx].append(experience)
        self.size += 1
    
    def get_experiences(self, agent_idx):
        """Get all experiences for a specific agent."""
        return self.experiences[agent_idx]
    
    def get_all_experiences(self):
        """Get experiences for all agents."""
        return self.experiences

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

class ActorCritic(nn.Module):
    """
    Actor-Critic network for SEAC algorithm (with matched architecture to HSEAC worker)
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=[64, 64]):
        super(ActorCritic, self).__init__()
        
        # Actor network (policy)
        self.actor = create_mlp(obs_dim, action_dim, hidden_dims)
        
        # Critic network (value function)
        self.critic = create_mlp(obs_dim, 1, hidden_dims)
    
    def forward(self, x):
        """
        Forward pass through both networks.
        """
        # Get action logits and convert to probabilities
        logits = self.actor(x)
        policy = F.softmax(logits, dim=-1)
        
        # Get state value estimate
        value = self.critic(x)
        
        return policy, value
    
    def get_action(self, state, action_mask=None, deterministic=False):
        """
        Select an action based on the policy.
        
        Args:
            state: Observation numpy array
            action_mask: Boolean mask for valid actions (True = valid)
            deterministic: Whether to select actions deterministically
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            probs, _ = self.forward(state_tensor)
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
    
    def evaluate_actions(self, states, actions, action_masks=None):
        """
        Evaluate log probabilities, values and entropy for given states and actions.
        """
        probs, values = self.forward(states)
        
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

class EqualizedSEAC:
    """
    Shared Experience Actor-Critic implementation for TA-RWARE with parameters matched to HSEAC
    """
    def __init__(
        self, 
        env, 
        gamma=0.99, 
        lr=3e-4,
        ent_coef=0.01,
        lambda_coef=1.0,
        hidden_dims=[64, 64],  # Matched to HSEAC worker networks
        clip_importance_ratio=5.0,
        update_interval=100,
        # Add annealing parameters (matched to HSEAC)
        det_action_start_prob=0.0,    
        det_action_final_prob=0.8,    
        det_action_anneal_episodes=700,
        log_dir="runs/equalized_seac_tarware",
        debug=False
    ):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.ent_coef = ent_coef
        self.lambda_coef = lambda_coef
        self.clip_importance_ratio = clip_importance_ratio
        self.update_interval = update_interval
        self.debug = debug
        
        # Add deterministic action annealing parameters (matched to HSEAC)
        self.det_action_start_prob = det_action_start_prob
        self.det_action_final_prob = det_action_final_prob
        self.det_action_anneal_episodes = det_action_anneal_episodes
        self.det_action_prob = det_action_start_prob  # Current probability (will be updated)
        
        # Get agent counts
        self.n_agvs = self.env.num_agvs if hasattr(self.env, "num_agvs") else 0
        self.n_pickers = self.env.num_pickers if hasattr(self.env, "num_pickers") else 0
        self.n_agents = self.env.num_agents if hasattr(self.env, "num_agents") else len(self.env.agents)
        
        # Create zone mapping for valid action masking (to match HSEAC's environment understanding)
        if hasattr(self.env, "rack_groups"):
            self.n_zones = len(self.env.rack_groups)
        else:
            self.n_zones = max(10, self.n_agents * 2)  # Ensure enough zones (matched to HSEAC)
        
        # Setup tensorboard logging
        self.writer = SummaryWriter(log_dir)
        self._train_step = 0
        
        # Determine observation and action dimensions
        self.obs_dim = self._get_obs_dim()
        self.action_dim = self._get_action_dim()
        
        # Create actor-critic networks for each agent
        self.actor_critics = []
        self.optimizers = []
        self.schedulers = []  # Added schedulers to match HSEAC's learning rate scheduling
        
        for i in range(self.n_agents):
            net = ActorCritic(
                self.obs_dim, 
                self.action_dim, 
                hidden_dims
            ).to(device)
            
            self.actor_critics.append(net)
            
            # Optimizer for actor-critic networks (AdamW to match HSEAC)
            optimizer = optim.AdamW(
                net.parameters(), 
                lr=lr,
                weight_decay=0.01  # Matched to HSEAC's weight decay
            )
            self.optimizers.append(optimizer)
            
            # Scheduler for learning rate annealing (matched to HSEAC)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.det_action_anneal_episodes,
                eta_min=1e-6
            )
            self.schedulers.append(scheduler)
        
        # Create experience buffer
        self.buffer = ExperienceBuffer(self.n_agents)
        
        # For tracking previous states of agents
        self.prev_agent_states = [None] * self.n_agents
        
        # Initialize agent tracking info
        self._initialize_agent_states()
    
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
    
    def get_valid_actions(self, agent_idx):
        """
        Get mask of valid actions for an agent.
        """
        # Get valid action masks from the environment if available
        if hasattr(self.env, "compute_valid_action_masks"):
            valid_action_masks = self.env.compute_valid_action_masks()
            return valid_action_masks[agent_idx]
        
        # If environment doesn't provide masking, all actions are valid
        return np.ones(self.action_dim, dtype=bool)
    
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
    
    def update_agents(self):
        """
        Update all agents' networks using collected experiences with experience sharing.
        """
        # Get experiences for all agents
        all_experiences = self.buffer.get_all_experiences()
        
        # Update each agent
        for agent_idx in range(self.n_agents):
            # Get agent's own experiences
            agent_experiences = all_experiences[agent_idx]
            
            if len(agent_experiences) == 0:
                continue
            
            # Prepare data for network update
            states = np.array([e.state for e in agent_experiences], dtype=np.float32)
            actions = np.array([e.action for e in agent_experiences], dtype=np.int64)
            rewards = np.array([e.reward for e in agent_experiences], dtype=np.float32)
            next_states = np.array([e.next_state for e in agent_experiences], dtype=np.float32)
            dones = np.array([float(e.done) for e in agent_experiences], dtype=np.float32)
            log_probs = np.array([e.log_prob for e in agent_experiences], dtype=np.float32)
            
            # Convert to tensors
            states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
            log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32).to(device)
            
            # Compute TD errors and advantages
            with torch.no_grad():
                _, next_values = self.actor_critics[agent_idx].forward(next_states_tensor)
                _, values = self.actor_critics[agent_idx].forward(states_tensor)
                
                next_values = next_values.squeeze()
                values = values.squeeze()
                
                # Compute TD targets and advantages
                td_targets = rewards_tensor + self.gamma * next_values * (1 - dones_tensor)
                advantages = td_targets - values
                
                # Normalize advantages
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Get action masks for states if environment supports it
            action_masks = None
            if hasattr(self.env, "compute_valid_action_masks"):
                # Note: We're using the same action mask for all experiences
                # In a full implementation, you'd store the masks with the experiences
                action_masks = torch.tensor(self.get_valid_actions(agent_idx), 
                                          dtype=torch.bool).to(device)
            
            # Get current log probs, values, and entropy
            current_log_probs, current_values, entropy = self.actor_critics[agent_idx].evaluate_actions(
                states_tensor, actions_tensor, action_masks
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
                
                # Only share experiences between same types of agents
                if (agent_idx < self.n_agvs and other_idx >= self.n_agvs) or \
                   (agent_idx >= self.n_agvs and other_idx < self.n_agvs):
                    continue
                
                other_experiences = all_experiences[other_idx]
                
                for exp in other_experiences:
                    # Compute importance sampling ratio
                    with torch.no_grad():
                        other_state = torch.tensor(exp.state, dtype=torch.float32).unsqueeze(0).to(device)
                        other_action = torch.tensor([exp.action], dtype=torch.int64).unsqueeze(1).to(device)
                        
                        # Get probabilities from both policies
                        own_probs, _ = self.actor_critics[agent_idx].forward(other_state)
                        own_prob = own_probs[0, exp.action].item()
                        
                        other_prob = np.exp(exp.log_prob)  # Convert log prob to prob
                        
                        # Compute ratio with clipping
                        importance_ratio = own_prob / (other_prob + 1e-10)
                        importance_ratio = np.clip(importance_ratio, 0, self.clip_importance_ratio)
                        importance_ratio_tensor = torch.tensor([importance_ratio], dtype=torch.float32).to(device)
                    
                    # Compute shared advantage
                    with torch.no_grad():
                        _, own_value = self.actor_critics[agent_idx].forward(other_state)
                        next_own_value = 0.0
                        
                        if not exp.done:
                            next_state = torch.tensor(exp.next_state, dtype=torch.float32).unsqueeze(0).to(device)
                            _, next_own_value = self.actor_critics[agent_idx].forward(next_state)
                            next_own_value = next_own_value.item()
                        
                        shared_adv = exp.reward + self.gamma * next_own_value * (1.0 - float(exp.done)) - own_value.item()
                    
                    
                    # Get current policy's log prob for this experience
                    action_mask = None
                    if hasattr(self.env, "compute_valid_action_masks"):
                        action_mask = torch.tensor(self.get_valid_actions(agent_idx), 
                                                 dtype=torch.bool).unsqueeze(0).to(device)
                    
                    own_log_prob, own_value, _ = self.actor_critics[agent_idx].evaluate_actions(
                        other_state, other_action, action_mask
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
            
            # Update network
            self.optimizers[agent_idx].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critics[agent_idx].parameters(), 0.5)
            self.optimizers[agent_idx].step()
            
            # Step the scheduler (matched to HSEAC)
            self.schedulers[agent_idx].step()
            
            # Log metrics
            self.writer.add_scalar(f'agent_{agent_idx}/policy_loss', policy_loss.item(), self._train_step)
            self.writer.add_scalar(f'agent_{agent_idx}/value_loss', value_loss.item(), self._train_step)
            self.writer.add_scalar(f'agent_{agent_idx}/shared_policy_loss', 
                                shared_policy_loss.mean().item() if shared_experiences_count > 0 else 0, 
                                self._train_step)
            self.writer.add_scalar(f'agent_{agent_idx}/shared_value_loss', 
                                shared_value_loss.mean().item() if shared_experiences_count > 0 else 0, 
                                self._train_step)
            self.writer.add_scalar(f'agent_{agent_idx}/entropy', entropy.mean().item(), self._train_step)
    
    def train(self, n_episodes, max_steps=500, render=False):
        """
        Train the agents using equalized SEAC algorithm.
        """
        self._train_step = 0
        episode_rewards = []
        eval_returns = []
        eval_deliveries = []  # Added to track deliveries like HSEAC
        
        for episode in range(n_episodes):
            # Update deterministic action probability based on training progress (matched to HSEAC)
            if episode < self.det_action_anneal_episodes:
                progress = episode / self.det_action_anneal_episodes
                self.det_action_prob = self.det_action_start_prob + progress * (self.det_action_final_prob - self.det_action_start_prob)
            else:
                self.det_action_prob = self.det_action_final_prob
                
            # Log the current probability
            self.writer.add_scalar('training/det_action_prob', self.det_action_prob, episode)
            
            obs = self.env.reset()  # Modern Gymnasium API
            obs = self.preprocess_obs(obs)
            episode_reward = 0
            episode_shelf_deliveries = 0
            
            # Clear buffer for new episode
            self.buffer.clear()
            
            # Update agent states tracking
            self._initialize_agent_states()
            
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
                
                # Select actions for each agent
                actions = []
                log_probs = []
                
                for agent_idx in range(self.n_agents):
                    # Get valid actions
                    valid_actions = self.get_valid_actions(agent_idx)
                    
                    # Use deterministic actions with increasing probability during training
                    use_deterministic = random.random() < self.det_action_prob
                    
                    # Select action
                    action, log_prob, _ = self.actor_critics[agent_idx].get_action(
                        obs[agent_idx], valid_actions, deterministic=use_deterministic
                    )
                    
                    actions.append(action)
                    log_probs.append(log_prob)
                
                # Execute actions in environment
                next_obs, rewards, terminated, truncated, info = self.env.step(tuple(actions))
                next_obs = self.preprocess_obs(next_obs)
                
                # Track shelf deliveries if available in info
                if 'shelf_deliveries' in info:
                    episode_shelf_deliveries += info['shelf_deliveries']
                
                # Determine if episode is done (all agents are done)
                done = all(terminated) or all(truncated)
                
                # Store experience
                self.buffer.add_experience(
                    obs, actions, rewards, next_obs, done, log_probs
                )
                
                # Update observations
                obs = next_obs
                
                # Accumulate episode reward
                episode_reward += sum(rewards)
                
                # Update networks if we have enough steps
                if step % self.update_interval == 0 and step > 0:
                    self.update_agents()
                    self._train_step += 1
                    
                    # Clear buffer after update
                    self.buffer.clear()
                
                # Check if episode is done
                if done or step == max_steps - 1:
                    break
            
            # Final update at end of episode if buffer is not empty
            if self.buffer.size > 0:
                self.update_agents()
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
                self.save(f"checkpoints/equalized_seac_tarware_episode_{episode+1}")
        
        # Close tensorboard writer
        self.writer.close()
        
        return episode_rewards, eval_returns, eval_deliveries
    
    def evaluate(self, n_episodes, render=False):
        """
        Evaluate the agent's performance.
        Modified to also return shelf deliveries like HSEAC.
        """
        total_returns = 0
        total_shelf_deliveries = 0
        
        for _ in range(n_episodes):
            obs = self.env.reset()  # Modern Gymnasium API
            obs = self.preprocess_obs(obs)
            episode_return = 0
            episode_shelf_deliveries = 0
            
            # Update agent states tracking
            self._initialize_agent_states()
            
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
                
                # Select actions for each agent
                actions = []
                
                for agent_idx in range(self.n_agents):
                    # Get valid actions
                    valid_actions = self.get_valid_actions(agent_idx)
                    
                    # Select action deterministically
                    action, _, _ = self.actor_critics[agent_idx].get_action(
                        obs[agent_idx], valid_actions, deterministic=True
                    )
                    actions.append(action)
                
                # Execute actions
                next_obs, rewards, terminated, truncated, info = self.env.step(tuple(actions))
                next_obs = self.preprocess_obs(next_obs)
                
                # Track shelf deliveries if available
                if 'shelf_deliveries' in info:
                    episode_shelf_deliveries += info['shelf_deliveries']
                
                # Update observations
                obs = next_obs
                
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
        
        # Save actor-critic networks
        for agent_idx in range(self.n_agents):
            torch.save(
                self.actor_critics[agent_idx].state_dict(), 
                f"{path}/agent_{agent_idx}.pt"
            )
            
        # Save configuration
        config_path = os.path.join(path, "config.txt")
        with open(config_path, 'w') as f:
            f.write(f"n_agents: {self.n_agents}\n")
            f.write(f"n_agvs: {self.n_agvs}\n")
            f.write(f"n_pickers: {self.n_pickers}\n")
            f.write(f"obs_dim: {self.obs_dim}\n")
            f.write(f"action_dim: {self.action_dim}\n")
            f.write(f"gamma: {self.gamma}\n")
            f.write(f"lr: {self.lr}\n")
            f.write(f"ent_coef: {self.ent_coef}\n")
            f.write(f"lambda_coef: {self.lambda_coef}\n")
    
    def load(self, path):
        """
        Load the agent's parameters.
        """
        try:
            # Load actor-critic networks
            for agent_idx in range(self.n_agents):
                agent_path = f"{path}/agent_{agent_idx}.pt"
                if os.path.exists(agent_path):
                    self.actor_critics[agent_idx].load_state_dict(
                        torch.load(agent_path, map_location=device)
                    )
                else:
                    print(f"[WARNING] Agent network file not found: {agent_path}")
                    return False
            
            print(f"Successfully loaded model from {path}")
            return True
        
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return False