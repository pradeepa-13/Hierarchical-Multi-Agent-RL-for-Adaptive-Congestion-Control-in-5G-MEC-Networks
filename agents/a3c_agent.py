#!/usr/bin/env python3
"""
a3c_agent.py
A3C (Asynchronous Advantage Actor-Critic) Agent

Implements:
- Asynchronous training with multiple workers
- Global network shared across workers
- Thread-safe gradient updates
- Edge-level resource management

Place in: ai_controller/agents/a3c_agent.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
import threading
import queue
import os

from agents.a3c_network import A3CNetwork, A3CNetworkWrapper
from agents.a3c_worker import A3CWorker
from rl_spaces import EdgeState, EdgeAction
from config import A3CConfig


class A3CAgent:
    """
    A3C Agent for Edge-Level Resource Management
    
    Manages:
    - Global network shared across workers
    - Multiple asynchronous workers
    - Gradient synchronization
    - Action selection for deployment
    
    Usage:
        agent = A3CAgent(config)
        
        # During episode (real-time decision)
        action = agent.select_action(edge_state)
        
        # Training happens asynchronously via workers
        agent.train(env_getter, max_episodes=1000)
    """
    
    def __init__(self, config: A3CConfig):
        """
        Args:
            config: A3CConfig with hyperparameters
        """
        self.config = config
        # Add logger
        import logging
        self.logger = logging.getLogger("A3CAgent")
        
        # Global network (shared across all workers)
        self.global_network = A3CNetwork(
            state_dim=11,  # EdgeState vector size
            action_dim=6,  # Number of EdgeActions
            hidden_layers=config.hidden_layers,
            use_lstm=True,
            device=config.device
        )
        
        # Shared optimizer (thread-safe)
        self.optimizer = optim.Adam(
            self.global_network.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Workers list
        self.workers: List[A3CWorker] = []
        self.worker_threads: List[threading.Thread] = []
        
        # Statistics
        self.episode_count = 0
        self.update_count = 0
        self.total_steps = 0
        
        # Exploration
        self.exploration_rate = config.initial_exploration
        self.exploration_decay = (config.initial_exploration - config.final_exploration) / config.exploration_decay_steps
        
        # Thread lock for statistics
        self.stats_lock = threading.Lock()
        
        # Training flag
        self.training = False
        
        print(f"✓ A3C Agent initialized")
        print(f"  Network parameters: {sum(p.numel() for p in self.global_network.parameters()):,}")
        print(f"  Device: {config.device}")
        print(f"  Workers: {config.num_workers}")

    def train_offline(self, transitions: List) -> Dict[str, float]:
        """
        Train A3C agent offline using collected transitions
        
        Args:
            transitions: List of Transition objects from state tracker
            
        Returns:
            Training statistics
        """

         # Debug: check what we're receiving
        self.logger.info(f"Received {len(transitions)} total transitions")
        if transitions:
            sample_trans = transitions[0]
            self.logger.info(f"Sample transition type: {type(sample_trans)}")
            self.logger.info(f"Sample transition attributes: {dir(sample_trans)}")
            if hasattr(sample_trans, 'agent_type'):
                self.logger.info(f"Agent type: {sample_trans.agent_type}")
        if not transitions:
            self.logger.info("No transitions available for A3C offline training")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'transitions_used': 0}
        
        # Filter only A3C transitions and ensure they have the right structure
        a3c_transitions = []
        for trans in transitions:
            if hasattr(trans, 'agent_type') and trans.agent_type == 'a3c':
                a3c_transitions.append(trans)
            elif not hasattr(trans, 'agent_type'):
                # Assume it's an A3C transition if agent_type is missing
                a3c_transitions.append(trans)
        
        if not a3c_transitions:
            self.logger.info("No valid A3C transitions found for training")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'transitions_used': 0}
        
        self.logger.info(f"Training A3C offline with {len(a3c_transitions)} transitions")
        
        # Convert transitions to training batches
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for trans in a3c_transitions:
            try:
                states.append(trans.state)
                actions.append(trans.action)
                rewards.append(trans.reward)
                next_states.append(trans.next_state)
                dones.append(trans.done)
            except AttributeError as e:
                self.logger.warning(f"Skipping invalid transition: {e}")
                continue
        
        if not states:
            self.logger.warning("No valid state data found in transitions")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'transitions_used': 0}
        
        try:
            # Convert to tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(self.config.device)
            actions_tensor = torch.LongTensor(actions).to(self.config.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.config.device)
            next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.config.device)
            dones_tensor = torch.BoolTensor(dones).to(self.config.device)
            
            # Training loop
            policy_losses = []
            value_losses = []
            entropies = []
            
            for epoch in range(3):  # 3 epochs over the data
                # ✅ FIX: Unpack ALL 3 values from network forward pass
                action_probs, values, lstm_hidden = self.global_network(states_tensor)
                _, next_values, _ = self.global_network(next_states_tensor)
                
                # Compute advantages
                advantages = rewards_tensor + self.config.gamma * next_values.squeeze() * (~dones_tensor).float() - values.squeeze()
                
                # Compute policy loss
                action_log_probs = torch.log(action_probs + 1e-10)
                selected_action_log_probs = action_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
                policy_loss = -(selected_action_log_probs * advantages.detach()).mean()
                
                # Compute value loss
                value_loss = advantages.pow(2).mean()
                
                # Compute entropy
                entropy = -(action_log_probs * action_probs).sum(dim=1).mean()
                
                # Total loss
                total_loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.global_network.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
            
            # Update statistics
            self.update_count += 1
            
            return {
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'entropy': np.mean(entropies),
                'transitions_used': len(states)
            }
            
        except Exception as e:
            self.logger.error(f"Error during A3C offline training: {e}")
            import traceback
            traceback.print_exc()
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'transitions_used': 0}

    def enable_offline_mode(self):
        """Enable offline training mode"""
        self.training = True
        # Use lower exploration for offline training
        self.exploration_rate = max(self.config.final_exploration, self.exploration_rate * 0.5)

    def disable_offline_mode(self):
        """Disable offline training mode"""
        self.training = False
    
    def select_action(self, 
                     state: EdgeState,
                     training: bool = True) -> Tuple[EdgeAction, float, float]:
        """
        Select action for given edge state (for real-time deployment)
        
        Args:
            state: EdgeState object
            training: If False, use deterministic policy
        
        Returns:
            action: EdgeAction enum
            log_prob: Log probability of action
            value: Value estimate
        """
        state_vector = state.to_vector()
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.config.device)
        
        # Epsilon-greedy exploration (during training)
        if training and np.random.rand() < self.exploration_rate:
            # Random action
            action_idx = np.random.randint(0, 7)
            log_prob = np.log(1.0 / 7)  # Uniform distribution
            value = 0.0
        else:
            # Policy action
            with torch.no_grad():
                action_idx, log_prob, value = self.global_network.get_action(
                    state_tensor,
                    deterministic=(not training)
                )
        action_idx = max(0, min(5, action_idx))
        # Decay exploration
        if training:
            self.exploration_rate = max(
                self.config.final_exploration,
                self.exploration_rate - self.exploration_decay
            )
        
        return EdgeAction(action_idx), float(log_prob), float(value)
    
    def create_workers(self, env_getter: callable) -> List[A3CWorker]:
        """
        Create worker processes
        
        Args:
            env_getter: Function that returns environment instance
        
        Returns:
            List of A3CWorker objects
        """
        workers = []
        
        for worker_id in range(self.config.num_workers):
            worker = A3CWorker(
                worker_id=worker_id,
                global_network=self.global_network,
                optimizer=self.optimizer,
                config=self.config,
                env_getter=env_getter
            )
            workers.append(worker)
        
        return workers
    
    def train(self, env_getter: callable, max_episodes: int = 1000):
        """
        Start asynchronous training with multiple workers
        
        Args:
            env_getter: Function that returns fresh environment instance
            max_episodes: Maximum episodes per worker
        """
        print("\n" + "=" * 60)
        print("STARTING A3C ASYNCHRONOUS TRAINING")
        print("=" * 60)
        print(f"  Workers: {self.config.num_workers}")
        print(f"  Max Episodes per Worker: {max_episodes}")
        print(f"  N-steps: {self.config.n_steps}")
        print("=" * 60 + "\n")
        
        self.training = True
        
        # Create workers
        self.workers = self.create_workers(env_getter)
        
        # Start worker threads
        for worker in self.workers:
            thread = threading.Thread(
                target=worker.run,
                args=(max_episodes,),
                daemon=True
            )
            self.worker_threads.append(thread)
            thread.start()
        
        # Wait for all workers to complete
        for thread in self.worker_threads:
            thread.join()
        
        self.training = False
        
        print("\n" + "=" * 60)
        print("A3C TRAINING COMPLETED")
        print("=" * 60)
        
        # Aggregate statistics from workers
        total_episodes = sum(w.episode_count for w in self.workers)
        total_steps = sum(w.step_count for w in self.workers)
        avg_reward = sum(w.total_reward for w in self.workers) / total_episodes if total_episodes > 0 else 0
        
        print(f"  Total Episodes: {total_episodes}")
        print(f"  Total Steps: {total_steps}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print("=" * 60 + "\n")
    
    def save(self, filepath: str):
        """Save agent checkpoint"""
        # CRITICAL: Create directory structure
        import os
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        torch.save({
            'global_network': self.global_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
            'exploration_rate': self.exploration_rate,
            'config': self.config
        }, filepath)
        
        print(f"✓ A3C Agent checkpoint saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.config.device, weights_only=False)
        
        self.global_network.load_state_dict(checkpoint['global_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_count = checkpoint['episode_count']
        self.update_count = checkpoint['update_count']
        self.total_steps = checkpoint['total_steps']
        self.exploration_rate = checkpoint['exploration_rate']
        
        print(f"✓ A3C Agent checkpoint loaded from {filepath}")
        print(f"  Episodes: {self.episode_count}, Steps: {self.total_steps}")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get agent statistics"""
        with self.stats_lock:
            return {
                'episode_count': self.episode_count,
                'update_count': self.update_count,
                'total_steps': self.total_steps,
                'exploration_rate': self.exploration_rate,
                'num_workers': len(self.workers),
                'training': self.training
            }


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    """Test A3C agent"""
    from config import A3CConfig
    from agents.a3c_worker import MockEdgeEnvironment
    
    print("Testing A3C Agent\n")
    
    # Test 1: Agent creation
    print("=" * 60)
    print("TEST 1: Agent Creation")
    print("=" * 60)
    
    config = A3CConfig(
        hidden_layers=[64, 64],
        learning_rate=1e-3,
        n_steps=20,
        num_workers=4,
        device='cpu'
    )
    
    agent = A3CAgent(config)
    print(f"✓ Agent created")
    
    # Test 2: Action selection
    print("\n" + "=" * 60)
    print("TEST 2: Action Selection")
    print("=" * 60)
    
    test_state = EdgeState(
        location='india',
        cpu_usage=0.68,
        memory_usage=0.72,
        network_usage=0.65,
        total_flows=15,
        embb_flows=5,
        urllc_flows=6,
        mmtc_flows=4,
        avg_delay_ms=22.0,
        sla_violations=1
    )
    
    action, log_prob, value = agent.select_action(test_state, training=True)
    
    print(f"✓ Action selected: {action.name}")
    print(f"  Log prob: {log_prob:.4f}")
    print(f"  Value: {value:.4f}")
    
    # Test 3: Worker creation
    print("\n" + "=" * 60)
    print("TEST 3: Worker Creation")
    print("=" * 60)
    
    def env_getter():
        return MockEdgeEnvironment()
    
    workers = agent.create_workers(env_getter)
    print(f"✓ Created {len(workers)} workers")
    
    # Test 4: Short training run
    print("\n" + "=" * 60)
    print("TEST 4: Asynchronous Training (5 episodes)")
    print("=" * 60)
    
    agent.train(env_getter, max_episodes=5)
    
    stats = agent.get_statistics()
    print(f"✓ Training completed")
    print(f"  Statistics: {stats}")
    
    # Test 5: Save/Load
    print("\n" + "=" * 60)
    print("TEST 5: Save/Load")
    print("=" * 60)
    
    agent.save("/tmp/test_a3c_agent.pth")
    
    new_agent = A3CAgent(config)
    new_agent.load("/tmp/test_a3c_agent.pth")
    
    print(f"✓ Save/Load successful")
    
    print("\n✅ All A3C agent tests passed!")
    print("\nAgent is ready for edge-level control!")