#!/usr/bin/env python3
"""
ppo_agent.py - FIXED VERSION
PPO Agent with flow_id tracking in buffer

KEY FIX: Store flow_id with each transition for proper reward mapping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
from torch.distributions import Categorical
import os

from agents.ppo_network import PPONetwork
from rl_spaces import FlowState, FlowAction
from config import PPOConfig


class PPOAgent:
    """PPO Agent with flow_id tracking"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # Create network
        self.network = PPONetwork(
            state_dim=15,
            action_dim=5,
            hidden_layers=config.hidden_layers,
            activation=config.activation,
            device=config.device
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.get_parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        # ✅ ADD HERE (after optimizer, before buffer initialization):
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,  # Decay every 100 updates
            gamma=0.95      # Multiply LR by 0.95
        )
        
        # ✅ FIX: Experience buffer with flow_id tracking
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.flow_ids = []  # ✅ NEW: Track which flow each transition belongs to
        
        # Statistics
        self.episode_count = 0
        self.update_count = 0
        self.total_steps = 0
        
        # Exploration
        self.exploration_rate = config.initial_exploration
        self.exploration_decay = (config.initial_exploration - config.final_exploration) / config.exploration_decay_steps
        
        print(f"✓ PPO Agent initialized")
        print(f"  Network parameters: {sum(p.numel() for p in self.network.get_parameters()):,}")
        print(f"  Device: {config.device}")

    
    def select_action(self, 
                    state: FlowState,
                    training: bool = True) -> Tuple[FlowAction, float, float]:
        """Select action for given flow state"""
        state_vector = state.to_vector()
        
        # Epsilon-greedy exploration
        if training and np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(0, 5)
            log_prob = np.log(1.0 / 5)
            value = 0.0
        else:
            # ✅ FIX: Use network.select_action() instead of network.forward()
            action_idx, log_prob, value = self.network.select_action(
                state_vector, 
                deterministic=not training
            )
            
            # Convert to FlowAction
            action_idx = min(action_idx, 4)
        
        # Decay exploration
        if training:
            self.exploration_rate = max(
                self.config.final_exploration,
                self.exploration_rate - self.exploration_decay
            )
        
        return FlowAction.from_index(action_idx), float(log_prob), float(value)
    
    def store_transition(self,
                        state: FlowState,
                        action: FlowAction,
                        reward: float,
                        next_state: FlowState,
                        done: bool,
                        log_prob: float = None,
                        value: float = None):
        """Store experience with flow_id tracking"""
        self.states.append(state.to_vector())
        self.actions.append(action.value)
        self.rewards.append(reward)
        self.next_states.append(next_state.to_vector())
        self.dones.append(done)
        self.flow_ids.append(state.flow_id)  # ✅ NEW: Store flow_id
        
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
        
        self.total_steps += 1
    
    def update_reward(self, flow_id: int, reward: float):
        """
        ✅ NEW METHOD: Update reward for specific flow's last transition
        """
        # Find the most recent transition for this flow_id
        for i in range(len(self.flow_ids) - 1, -1, -1):
            if self.flow_ids[i] == flow_id:
                self.rewards[i] = reward
                return True
        
        return False  # Flow not found in buffer
    
    def compute_gae(self, 
                   rewards: List[float],
                   values: List[float],
                   next_values: List[float],
                   dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE"""
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.config.gamma * next_values[t] - values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
            
            advantages.insert(0, gae)
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values, dtype=np.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO"""
        if len(self.states) < self.config.batch_size:
            print(f"⚠️  Not enough data for update ({len(self.states)} < {self.config.batch_size})")
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.config.device)
        actions = torch.LongTensor(self.actions).to(self.config.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.config.device)
        
        # Compute next state values
        with torch.no_grad():
            next_states = torch.FloatTensor(np.array(self.next_states)).to(self.config.device)
            _, next_values_tensor = self.network.network(next_states)
            next_values = next_values_tensor.squeeze(-1).cpu().numpy()
        
        # Compute advantages using GAE
        advantages, returns = self.compute_gae(
            self.rewards,
            self.values,
            next_values.tolist(),
            self.dones
        )
        
        advantages = torch.FloatTensor(advantages).to(self.config.device)
        returns = torch.FloatTensor(returns).to(self.config.device)
        
        # PPO update
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'kl_div': 0.0,
            'clip_fraction': 0.0
        }
        
        n_batches = 0
        
        for epoch in range(self.config.n_epochs):
            batch_size = min(self.config.minibatch_size, len(states))
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                log_probs, values, entropies = self.network.network.evaluate_actions(
                    batch_states, batch_actions
                )
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                entropy_loss = -entropies.mean()
                
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.get_parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
                
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropies.mean().item()
                stats['kl_div'] += kl_div
                stats['clip_fraction'] += clip_fraction
                n_batches += 1
        
        for key in stats:
            stats[key] /= n_batches

        # ✅ ADD HERE (after stats normalization, before update_count):
        self.lr_scheduler.step()  # Decay learning rate
        
        # Log current LR
        current_lr = self.optimizer.param_groups[0]['lr']

        # ✅ ADD: Adaptive LR based on KL
        if stats['kl_div'] > 0.05:
            # Reduce LR by 10% if KL too high
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.9
            print(f"⚠️  High KL detected, reducing LR to {param_group['lr']:.6f}")
        stats['learning_rate'] = current_lr
        
        self.update_count += 1
        self.clear_buffer()
        
        return stats
    
    def clear_buffer(self):
        """Clear experience buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.flow_ids.clear()  # ✅ NEW: Clear flow_ids too
    
    def save(self, filepath: str):
        """Save checkpoint"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        torch.save({
            'network': self.network.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),  # ✅ ADD THIS LINE
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
            'exploration_rate': self.exploration_rate,
            'state_mean': self.network.state_mean,
            'state_std': self.network.state_std,
            'state_count': self.network.state_count
        }, filepath)
        
        print(f"✓ Checkpoint saved to {filepath}")
    
    def load(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.network.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # ✅ ADD THIS LINE:
        if 'lr_scheduler' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.episode_count = checkpoint['episode_count']
        self.update_count = checkpoint['update_count']
        self.total_steps = checkpoint['total_steps']
        self.exploration_rate = checkpoint['exploration_rate']
        self.network.state_mean = checkpoint['state_mean']
        self.network.state_std = checkpoint['state_std']
        self.network.state_count = checkpoint['state_count']
        
        print(f"✓ Checkpoint loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get agent statistics"""
        return {
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
            'exploration_rate': self.exploration_rate,
            'buffer_size': len(self.states)
        }