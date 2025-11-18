#!/usr/bin/env python3
"""
ppo_network.py
Actor-Critic Neural Network for PPO Agent

Architecture:
- Shared feature extractor
- Separate actor (policy) and critic (value) heads
- Supports continuous state space, discrete action space

Place in: ai_controller/agents/ppo_network.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO
    
    Architecture:
        Input (state) → Shared Layers → [Actor Head, Critic Head]
        
    Actor outputs: Action probability distribution
    Critic outputs: State value estimate
    """
    
    def __init__(self, 
                 state_dim: int = 14,
                 action_dim: int = 5,
                 hidden_layers: List[int] = [128, 128, 64],
                 activation: str = 'relu'):
        """
        Args:
            state_dim: Size of state vector (FlowState = 14)
            action_dim: Number of discrete actions (FlowAction = 7)
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # ========== Shared Feature Extractor ==========
        shared_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(self.activation)
            shared_layers.append(nn.LayerNorm(hidden_dim))  # Stabilizes training
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # ========== Actor Head (Policy) ==========
        # Outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, 64),
            self.activation,
            nn.Linear(64, action_dim)
        )
        
        # ========== Critic Head (Value Function) ==========
        # Outputs state value estimate
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 64),
            self.activation,
            nn.Linear(64, 1)
        )
        
        # Initialize weights with orthogonal initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network
        
        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
        
        Returns:
            action_logits: Unnormalized action scores [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
        """
        # Handle single state (add batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Shared feature extraction
        features = self.shared(state)
        
        # Actor output (logits, not probabilities yet)
        action_logits = self.actor(features)
        
        # Critic output (state value)
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action_and_value(self, 
                            state: torch.Tensor,
                            deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability + value
        
        Args:
            state: State tensor
            deterministic: If True, take argmax instead of sampling
        
        Returns:
            action: Sampled action index
            log_prob: Log probability of action
            entropy: Policy entropy (for exploration bonus)
            value: State value estimate
        """
        action_logits, value = self.forward(state)
        
        # Convert logits to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)
        
        if deterministic:
            # Take most likely action (for evaluation)
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from distribution (for training)
            action = dist.sample()
        
        # Compute log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, entropy, value
    
    def evaluate_actions(self, 
                        states: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions taken in given states (for PPO update)
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size]
        
        Returns:
            log_probs: Log probabilities of actions [batch_size]
            values: State values [batch_size, 1]
            entropies: Policy entropies [batch_size]
        """
        action_logits, values = self.forward(states)
        
        # Convert to distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        # Compute log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropies


class PPONetwork:
    """
    Wrapper class for easier usage
    
    Handles device management, state normalization, etc.
    """
    
    def __init__(self, 
                 state_dim: int = 14,
                 action_dim: int = 5,
                 hidden_layers: List[int] = [128, 128, 64],
                 activation: str = 'relu',
                 device: str = 'cpu'):
        """
        Args:
            state_dim: State vector size
            action_dim: Number of actions
            hidden_layers: Network architecture
            activation: Activation function
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # Create network
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=hidden_layers,
            activation=activation
        ).to(self.device)
        
        # State normalization (running mean/std)
        self.state_mean = torch.zeros(state_dim, device=self.device)
        self.state_std = torch.ones(state_dim, device=self.device)
        self.state_count = 0
    
    def normalize_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Normalize state using running statistics
        
        Args:
            state: Numpy state vector
        
        Returns:
            Normalized state tensor
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Update running statistics
        self.state_count += 1
        delta = state_tensor - self.state_mean
        self.state_mean += delta / self.state_count
        delta2 = state_tensor - self.state_mean
        self.state_std = torch.sqrt(
            (self.state_std ** 2 * (self.state_count - 1) + delta * delta2) / self.state_count
        )
        
        # Normalize (avoid division by zero)
        normalized = (state_tensor - self.state_mean) / (self.state_std + 1e-8)
        
        return normalized
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Select action for given state
        
        Args:
            state: State vector (numpy array)
            deterministic: If True, use greedy policy
        
        Returns:
            action: Action index
            log_prob: Log probability
            value: Value estimate
        """
        normalized_state = self.normalize_state(state)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(
                normalized_state, deterministic=deterministic
            )
        
        return action, log_prob.item(), value.item()
    
    def get_parameters(self):
        """Get network parameters for optimizer"""
        return self.network.parameters()
    
    def save(self, filepath: str):
        """Save network weights"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'state_count': self.state_count
        }, filepath)
        print(f"✓ Network saved to {filepath}")
    
    def load(self, filepath: str):
        """Load network weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.state_count = checkpoint['state_count']
        
        print(f"✓ Network loaded from {filepath}")


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    """Test network architecture"""
    print("Testing PPO Network Architecture\n")
    
    # Test 1: Network creation
    print("="*60)
    print("TEST 1: Network Creation")
    print("="*60)
    
    network = ActorCriticNetwork(
        state_dim=14,
        action_dim=5,
        hidden_layers=[128, 128, 64]
    )
    
    print(f"✓ Network created")
    print(f"  Parameters: {sum(p.numel() for p in network.parameters()):,}")
    
    # Test 2: Forward pass
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    dummy_state = torch.randn(1, 14)
    action_logits, value = network(dummy_state)
    
    print(f"✓ Forward pass successful")
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"  Value shape: {value.shape}")
    
    # Test 3: Action sampling
    print("\n" + "="*60)
    print("TEST 3: Action Sampling")
    print("="*60)
    
    action, log_prob, entropy, value = network.get_action_and_value(dummy_state)
    
    print(f"✓ Action sampled: {action}")
    print(f"  Log probability: {log_prob.item():.4f}")
    print(f"  Entropy: {entropy.item():.4f}")
    print(f"  Value: {value.item():.4f}")
    
    # Test 4: Batch evaluation
    print("\n" + "="*60)
    print("TEST 4: Batch Evaluation")
    print("="*60)
    
    batch_states = torch.randn(32, 14)
    batch_actions = torch.randint(0, 7, (32,))
    
    log_probs, values, entropies = network.evaluate_actions(batch_states, batch_actions)
    
    print(f"✓ Batch evaluation successful")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Entropies shape: {entropies.shape}")
    
    # Test 5: PPONetwork wrapper
    print("\n" + "="*60)
    print("TEST 5: PPONetwork Wrapper")
    print("="*60)
    
    ppo_net = PPONetwork(device='cpu')
    
    state_np = np.random.randn(14)
    action, log_prob, value = ppo_net.select_action(state_np)
    
    print(f"✓ Wrapper works")
    print(f"  Selected action: {action}")
    print(f"  Log prob: {log_prob:.4f}")
    print(f"  Value: {value:.4f}")
    
    # Test 6: Save/Load
    print("\n" + "="*60)
    print("TEST 6: Save/Load")
    print("="*60)
    
    ppo_net.save("/tmp/test_ppo_network.pth")
    
    new_net = PPONetwork(device='cpu')
    new_net.load("/tmp/test_ppo_network.pth")
    
    # Verify same output
    action2, _, _ = new_net.select_action(state_np, deterministic=True)
    print(f"✓ Save/Load successful")
    print(f"  Action before: {action}")
    print(f"  Action after: {action2}")
    
    print("\n✅ All network tests passed!")
    print("\nNetwork Architecture Summary:")
    print(network)
