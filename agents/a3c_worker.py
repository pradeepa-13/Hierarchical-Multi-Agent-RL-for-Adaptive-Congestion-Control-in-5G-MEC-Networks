#!/usr/bin/env python3
"""
a3c_worker.py
A3C Worker for Asynchronous Training

Implements:
- Individual worker process that collects experience
- Asynchronous gradient updates to global network
- N-step returns for advantage estimation
- Episode-based training with proper synchronization

Place in: ai_controller/agents/a3c_worker.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import threading
import time

from agents.a3c_network import A3CNetwork
from rl_spaces import EdgeState, EdgeAction


class A3CWorker:
    """
    A3C Worker Process
    
    Each worker:
    1. Maintains its own copy of the network
    2. Collects experience by interacting with environment
    3. Computes gradients
    4. Asynchronously updates global network
    """
    
    def __init__(self,
                 worker_id: int,
                 global_network: A3CNetwork,
                 optimizer: torch.optim.Optimizer,
                 config: 'A3CConfig',
                 env_getter: callable):
        """
        Args:
            worker_id: Unique worker identifier
            global_network: Shared global network
            optimizer: Shared optimizer
            config: A3C configuration
            env_getter: Function that returns a fresh environment instance
        """
        self.worker_id = worker_id
        self.global_network = global_network
        self.optimizer = optimizer
        self.config = config
        self.env_getter = env_getter
        
        # Local network (copy of global)
        self.local_network = A3CNetwork(
            state_dim=11,  # EdgeState
            action_dim=7,  # EdgeActions
            hidden_layers=config.hidden_layers,
            use_lstm=True,
            device=config.device
        )
        
        # Copy global weights
        self.local_network.load_state_dict(global_network.state_dict())
        
        # Statistics
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        
        # Thread lock for gradient updates
        self.lock = threading.Lock()
    
    def sync_with_global(self):
        """Synchronize local network with global network"""
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def compute_n_step_returns(self,
                               rewards: List[float],
                               values: List[float],
                               next_value: float,
                               dones: List[bool]) -> np.ndarray:
        """
        Compute n-step returns for advantage estimation
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value estimate for next state
            dones: List of done flags
        
        Returns:
            returns: N-step returns
        """
        returns = []
        R = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                R = 0  # Terminal state
            
            R = rewards[t] + self.config.gamma * R
            returns.insert(0, R)
        
        return np.array(returns, dtype=np.float32)
    
    def train_step(self, env) -> Dict[str, float]:
        """
        Execute one training step (n-step rollout + update)
        
        Args:
            env: Environment instance (simulated or real)
        
        Returns:
            stats: Training statistics
        """
        # Sync with global network
        self.sync_with_global()
        
        # Reset LSTM state
        self.local_network.reset_lstm_state()
        
        # Collect n-step experience
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        state = env.reset()  # Get initial state
        
        episode_reward = 0.0
        
        for step in range(self.config.n_steps):
            state_tensor = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.config.device)
            
            # Select action
            action_idx, log_prob, value = self.local_network.get_action(
                state_tensor,
                deterministic=False
            )
            
            # Store experience
            states.append(state.to_vector())
            actions.append(action_idx)
            log_probs.append(log_prob)
            values.append(value.item())
            
            # Take action in environment
            action = EdgeAction(action_idx)
            next_state, reward, done, _ = env.step(action)
            
            rewards.append(reward)
            dones.append(done)
            episode_reward += reward
            
            state = next_state
            self.step_count += 1
            
            if done:
                break
        
        # Compute next state value (for bootstrapping)
        with torch.no_grad():
            if done:
                next_value = 0.0
            else:
                next_state_tensor = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.config.device)
                _, next_value, _ = self.local_network.forward(next_state_tensor)
                next_value = next_value.item()
        
        # Compute n-step returns
        returns = self.compute_n_step_returns(rewards, values, next_value, dones)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.config.device)
        actions_tensor = torch.LongTensor(actions).to(self.config.device)
        returns_tensor = torch.FloatTensor(returns).to(self.config.device)
        values_tensor = torch.FloatTensor(values).to(self.config.device)
        log_probs_tensor = torch.stack(log_probs).to(self.config.device)
        
        # Compute advantages
        advantages = returns_tensor - values_tensor
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute losses
        self.local_network.reset_lstm_state()
        action_probs, critic_values, _ = self.local_network.forward(states_tensor)
        
        # Policy loss (with entropy regularization)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()
        
        policy_loss = -(new_log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(critic_values.squeeze(), returns_tensor)
        
        # Total loss
        loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss -
            self.config.entropy_coef * entropy
        )
        
        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.local_network.parameters(),
            self.config.max_grad_norm
        )
        
        # Update global network (with lock for thread safety)
        with self.lock:
            for global_param, local_param in zip(
                self.global_network.parameters(),
                self.local_network.parameters()
            ):
                if global_param.grad is not None:
                    global_param.grad += local_param.grad
                else:
                    global_param.grad = local_param.grad.clone()
            
            self.optimizer.step()
        
        # Statistics
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'episode_reward': episode_reward,
            'steps': len(states)
        }
        
        if done:
            self.episode_count += 1
            self.total_reward += episode_reward
        
        return stats
    
    def run(self, max_episodes: int = 1000):
        """
        Main worker loop
        
        Args:
            max_episodes: Maximum number of episodes to run
        """
        env = self.env_getter()  # Get environment instance
        
        print(f"Worker {self.worker_id} started")
        
        while self.episode_count < max_episodes:
            stats = self.train_step(env)
            
            if self.episode_count % 10 == 0 and stats['steps'] > 0:
                print(f"Worker {self.worker_id} | "
                      f"Episode {self.episode_count} | "
                      f"Reward: {stats['episode_reward']:.2f} | "
                      f"Policy Loss: {stats['policy_loss']:.4f} | "
                      f"Value Loss: {stats['value_loss']:.4f}")
        
        print(f"Worker {self.worker_id} finished")


# ============================================================================
# MOCK ENVIRONMENT FOR TESTING
# ============================================================================
class MockEdgeEnvironment:
    """
    Mock environment for testing A3C worker
    Simulates edge server management
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> EdgeState:
        """Reset environment to initial state"""
        self.state = EdgeState(
            location='test_edge',
            cpu_usage=0.5 + np.random.rand() * 0.2,
            memory_usage=0.5 + np.random.rand() * 0.2,
            network_usage=0.5 + np.random.rand() * 0.2,
            total_flows=10,
            embb_flows=4,
            urllc_flows=3,
            mmtc_flows=3,
            avg_delay_ms=25.0,
            sla_violations=0
        )
        self.step_count = 0
        return self.state
    
    def step(self, action: EdgeAction) -> Tuple[EdgeState, float, bool, Dict]:
        """
        Take action in environment
        
        Returns:
            next_state: New edge state
            reward: Immediate reward
            done: Episode terminated?
            info: Additional info
        """
        self.step_count += 1
        
        # Simulate action effects
        if action == EdgeAction.SCALE_UP_RESOURCES:
            self.state.cpu_usage = max(0.3, self.state.cpu_usage - 0.1)
            self.state.memory_usage = max(0.3, self.state.memory_usage - 0.1)
            reward = 0.5
        elif action == EdgeAction.SCALE_DOWN_RESOURCES:
            self.state.cpu_usage = min(0.9, self.state.cpu_usage + 0.1)
            self.state.memory_usage = min(0.9, self.state.memory_usage + 0.1)
            reward = 0.3
        else:
            reward = 0.1
        
        # Add small random fluctuation
        self.state.cpu_usage += np.random.randn() * 0.05
        self.state.memory_usage += np.random.randn() * 0.05
        self.state.network_usage += np.random.randn() * 0.05
        
        # Clip to valid range
        self.state.cpu_usage = np.clip(self.state.cpu_usage, 0.2, 1.0)
        self.state.memory_usage = np.clip(self.state.memory_usage, 0.2, 1.0)
        self.state.network_usage = np.clip(self.state.network_usage, 0.2, 1.0)
        
        # Compute reward based on resource efficiency
        if 0.6 <= self.state.cpu_usage <= 0.8:
            reward += 1.0
        else:
            reward -= 0.5
        
        done = (self.step_count >= 50)  # Episode length
        
        return self.state, reward, done, {}


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    """Test A3C worker"""
    from config import A3CConfig
    
    print("Testing A3C Worker\n")
    
    # Test 1: Worker creation
    print("=" * 60)
    print("TEST 1: Worker Creation")
    print("=" * 60)
    
    config = A3CConfig(
        hidden_layers=[64, 64],
        learning_rate=1e-3,
        n_steps=20,
        device='cpu'
    )
    
    global_network = A3CNetwork(
        state_dim=11,
        action_dim=7,
        hidden_layers=config.hidden_layers,
        device=config.device
    )
    
    optimizer = torch.optim.Adam(
        global_network.parameters(),
        lr=config.learning_rate
    )
    
    def env_getter():
        return MockEdgeEnvironment()
    
    worker = A3CWorker(
        worker_id=0,
        global_network=global_network,
        optimizer=optimizer,
        config=config,
        env_getter=env_getter
    )
    
    print(f"✓ Worker created (ID: {worker.worker_id})")
    
    # Test 2: Single training step
    print("\n" + "=" * 60)
    print("TEST 2: Single Training Step")
    print("=" * 60)
    
    env = MockEdgeEnvironment()
    stats = worker.train_step(env)
    
    print(f"✓ Training step completed")
    print(f"  Policy loss: {stats['policy_loss']:.4f}")
    print(f"  Value loss: {stats['value_loss']:.4f}")
    print(f"  Entropy: {stats['entropy']:.4f}")
    print(f"  Episode reward: {stats['episode_reward']:.2f}")
    print(f"  Steps: {stats['steps']}")
    
    # Test 3: Multiple episodes
    print("\n" + "=" * 60)
    print("TEST 3: Multiple Episodes")
    print("=" * 60)
    
    worker.run(max_episodes=5)
    
    print(f"✓ Training completed")
    print(f"  Total episodes: {worker.episode_count}")
    print(f"  Total steps: {worker.step_count}")
    print(f"  Average reward: {worker.total_reward / worker.episode_count:.2f}")
    
    print("\n✅ All A3C worker tests passed!")
    print("\nWorker is ready for multi-threaded A3C agent!")