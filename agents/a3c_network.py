#!/usr/bin/env python3
"""
a3c_network.py
A3C Neural Network Architecture for Edge-Level Control

Implements:
- Shared actor-critic network with separate heads
- LSTM for temporal dependencies (edge state history)
- Efficient weight sharing across workers

Place in: ai_controller/agents/a3c_network.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class A3CNetwork(nn.Module):
    """
    A3C Actor-Critic Network
    
    Architecture:
    - Shared feature extraction layers
    - LSTM for temporal dependencies (optional)
    - Separate actor (policy) and critic (value) heads
    
    Input: Edge state vector (11 features)
    Output: Action probabilities + Value estimate
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_layers: List[int] = [128, 64],
                 use_lstm: bool = True,
                 lstm_hidden_size: int = 128,
                 device: str = 'cpu'):
        """
        Args:
            state_dim: Edge state vector dimension (11)
            action_dim: Number of edge actions (7)
            hidden_layers: Hidden layer sizes
            use_lstm: Use LSTM for temporal modeling
            lstm_hidden_size: LSTM hidden state size
            device: 'cpu' or 'cuda'
        """
        super(A3CNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device
        
        # Shared feature extraction layers
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # LSTM for temporal dependencies (optional)
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=prev_size,
                hidden_size=lstm_hidden_size,
                num_layers=1,
                batch_first=True
            )
            lstm_output_size = lstm_hidden_size
        else:
            lstm_output_size = prev_size
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # LSTM hidden states (for recurrent processing)
        self.lstm_hidden = None
        
        self.to(device)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights with Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
    
    def forward(self, state: torch.Tensor, 
                lstm_hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass
        
        Args:
            state: State tensor [batch_size, state_dim] or [batch_size, seq_len, state_dim]
            lstm_hidden: Previous LSTM hidden state (h, c)
        
        Returns:
            action_probs: Action probability distribution [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
            lstm_hidden: Updated LSTM hidden state
        """
        # Feature extraction
        features = self.feature_extractor(state)
        
        # LSTM processing (if enabled)
        if self.use_lstm:
            # If state is 2D, add sequence dimension
            if features.dim() == 2:
                features = features.unsqueeze(1)  # [batch, 1, features]
            
            # Initialize hidden state if None
            if lstm_hidden is None:
                batch_size = features.size(0)
                h0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(self.device)
                c0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(self.device)
                lstm_hidden = (h0, c0)
            
            lstm_out, lstm_hidden = self.lstm(features, lstm_hidden)
            lstm_out = lstm_out[:, -1, :]  # Take last timestep
        else:
            lstm_out = features
            lstm_hidden = None
        
        # Actor-Critic heads
        action_probs = self.actor(lstm_out)
        value = self.critic(lstm_out)
        
        return action_probs, value, lstm_hidden
    
    def reset_lstm_state(self):
        """Reset LSTM hidden state (call at episode start)"""
        self.lstm_hidden = None
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action given state
        
        Args:
            state: State tensor
            deterministic: If True, select highest probability action
        
        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
            value: State value estimate
        """
        action_probs, value, self.lstm_hidden = self.forward(state, self.lstm_hidden)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        
        log_prob = torch.log(action_probs.squeeze()[action] + 1e-8)
        
        return action.item(), log_prob, value.squeeze()


class A3CNetworkWrapper:
    """
    Wrapper for A3C network with utility functions
    Similar to PPONetwork wrapper
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_layers: List[int] = [128, 64],
                 use_lstm: bool = True,
                 device: str = 'cpu'):
        
        self.device = device
        self.network = A3CNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=hidden_layers,
            use_lstm=use_lstm,
            device=device
        )
        
        # Running statistics for state normalization
        self.state_mean = torch.zeros(state_dim).to(device)
        self.state_std = torch.ones(state_dim).to(device)
        self.state_count = 0
    
    def forward(self, state):
        """Forward pass with state normalization"""
        state_tensor = self._preprocess_state(state)
        return self.network.forward(state_tensor)
    
    def _preprocess_state(self, state):
        """Convert state to normalized tensor"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Normalize state
        state = (state - self.state_mean) / (self.state_std + 1e-8)
        
        return state
    
    def update_state_stats(self, state):
        """Update running statistics for state normalization"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        # Incremental mean/std update
        self.state_count += 1
        delta = state - self.state_mean
        self.state_mean += delta / self.state_count
        delta2 = state - self.state_mean
        self.state_std = torch.sqrt(
            ((self.state_count - 1) * self.state_std ** 2 + delta * delta2) / self.state_count
        )
    
    def get_parameters(self):
        """Get network parameters for optimizer"""
        return self.network.parameters()
    
    def reset_lstm(self):
        """Reset LSTM state"""
        self.network.reset_lstm_state()


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    """Test A3C network"""
    print("Testing A3C Network\n")
    
    # Test 1: Network creation
    print("=" * 60)
    print("TEST 1: Network Creation")
    print("=" * 60)
    
    network = A3CNetworkWrapper(
        state_dim=11,  # EdgeState vector size
        action_dim=7,   # Number of EdgeActions
        hidden_layers=[128, 64],
        use_lstm=True,
        device='cpu'
    )
    
    print(f"✓ A3C Network created")
    print(f"  Total parameters: {sum(p.numel() for p in network.network.parameters()):,}")
    
    # Test 2: Forward pass (single state)
    print("\n" + "=" * 60)
    print("TEST 2: Forward Pass (Single State)")
    print("=" * 60)
    
    import numpy as np
    test_state = np.random.randn(11).astype(np.float32)
    
    action_probs, value, lstm_hidden = network.forward(test_state)
    
    print(f"✓ Forward pass successful")
    print(f"  Action probs shape: {action_probs.shape}")
    print(f"  Action probs sum: {action_probs.sum().item():.4f} (should be 1.0)")
    print(f"  Value shape: {value.shape}")
    print(f"  Value estimate: {value.item():.4f}")
    print(f"  LSTM hidden: {type(lstm_hidden)}")
    
    # Test 3: Action selection
    print("\n" + "=" * 60)
    print("TEST 3: Action Selection")
    print("=" * 60)
    
    network.reset_lstm()
    
    action, log_prob, value = network.network.get_action(
        torch.FloatTensor(test_state).unsqueeze(0),
        deterministic=False
    )
    
    print(f"✓ Action selected: {action}")
    print(f"  Log probability: {log_prob.item():.4f}")
    print(f"  Value: {value.item():.4f}")
    
    # Test 4: Sequence processing (LSTM)
    print("\n" + "=" * 60)
    print("TEST 4: Sequence Processing (LSTM)")
    print("=" * 60)
    
    network.reset_lstm()
    sequence_length = 5
    
    for t in range(sequence_length):
        state_t = np.random.randn(11).astype(np.float32)
        action, log_prob, value = network.network.get_action(
            torch.FloatTensor(state_t).unsqueeze(0)
        )
        print(f"  t={t}: action={action}, value={value.item():.4f}")
    
    print(f"✓ Sequence processing successful")
    
    # Test 5: Batch processing
    print("\n" + "=" * 60)
    print("TEST 5: Batch Processing")
    print("=" * 60)
    
    batch_size = 8
    batch_states = torch.randn(batch_size, 11)
    
    network.reset_lstm()
    action_probs, values, _ = network.forward(batch_states)
    
    print(f"✓ Batch processed")
    print(f"  Batch size: {batch_size}")
    print(f"  Action probs shape: {action_probs.shape}")
    print(f"  Values shape: {values.shape}")
    print(f"  Action probs sum (per sample): {action_probs.sum(dim=1)}")
    
    # Test 6: State normalization
    print("\n" + "=" * 60)
    print("TEST 6: State Normalization")
    print("=" * 60)
    
    for _ in range(100):
        state = np.random.randn(11).astype(np.float32)
        network.update_state_stats(torch.FloatTensor(state))
    
    print(f"✓ State statistics updated")
    print(f"  State count: {network.state_count}")
    print(f"  State mean: {network.state_mean[:3]}...")
    print(f"  State std: {network.state_std[:3]}...")
    
    # Verify normalization
    normalized_state = network._preprocess_state(test_state)
    print(f"  Normalized state mean: {normalized_state.mean().item():.4f}")
    print(f"  Normalized state std: {normalized_state.std().item():.4f}")
    
    print("\n✅ All A3C network tests passed!")
    print("\nNetwork is ready for A3C agent implementation!")