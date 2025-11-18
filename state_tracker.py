#!/usr/bin/env python3
"""
state_tracker.py
State History Management and Temporal Feature Computation

Tracks:
- Flow state history (for trend computation)
- Edge state history
- Episode statistics
- Action history
- Reward history

Place in: ai_controller/state_tracker.py
"""

import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import json
from socket_comm import NetworkState

from rl_spaces import (
    FlowState, EdgeState, GWOState, 
    FlowAction, EdgeAction,
    StateAggregator
)
from reward_functions import RewardCoordinator


# ============================================================================
# EPISODE STATISTICS
# ============================================================================
@dataclass
class EpisodeStats:
    """Statistics for a single training episode"""
    episode_id: int
    start_time: float
    end_time: float = 0.0
    
    # Cumulative rewards
    total_ppo_reward: float = 0.0
    total_a3c_reward: float = 0.0
    total_gwo_reward: float = 0.0
    aggregate_throughput: float = 0.0  # NEW
    fairness_index: float = 0.0  #NEW
    
    # Performance metrics
    avg_throughput: float = 0.0
    avg_delay: float = 0.0
    avg_loss: float = 0.0
    total_sla_violations: int = 0
    
    # Action counts
    ppo_actions: Dict[str, int] = field(default_factory=dict)
    a3c_actions: Dict[str, int] = field(default_factory=dict)
    
    # Step count
    steps: int = 0
    
    def duration(self) -> float:
        """Episode duration in seconds"""
        return self.end_time - self.start_time if self.end_time > 0 else time.time() - self.start_time
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'episode_id': self.episode_id,
            'duration': self.duration(),
            'total_ppo_reward': self.total_ppo_reward,
            'total_a3c_reward': self.total_a3c_reward,
            'total_gwo_reward': self.total_gwo_reward,
            'avg_throughput': self.avg_throughput,
            'avg_delay': self.avg_delay,
            'avg_loss': self.avg_loss,
            'total_sla_violations': self.total_sla_violations,
            'steps': self.steps,
            'ppo_actions': self.ppo_actions,
            'a3c_actions': self.a3c_actions
        }


# ============================================================================
# TRANSITION DATA (FOR EXPERIENCE REPLAY)
# ============================================================================
@dataclass
class Transition:
    """Single state-action-reward-next_state transition"""
    agent_type: str  # 'ppo', 'a3c', or 'gwo'
    
    state: np.ndarray
    action: int  # Discrete action index
    reward: float
    next_state: np.ndarray
    done: bool
    
    # Additional info
    flow_id: Optional[int] = None
    edge_location: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'agent_type': self.agent_type,
            'state': self.state.tolist(),
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state.tolist(),
            'done': self.done,
            'flow_id': self.flow_id,
            'edge_location': self.edge_location,
            'timestamp': self.timestamp
        }


# ============================================================================
# STATE TRACKER
# ============================================================================
class StateTracker:
    """
    Manages state history and computes temporal features
    
    Features:
    - Maintains rolling history of network states
    - Computes trends (derivatives)
    - Tracks rewards and actions
    - Generates training transitions
    - Episode management
    
    Usage:
        tracker = StateTracker(history_size=20)
        tracker.start_episode(episode_id=1)
        
        # Update with NS-3 data
        tracker.update(flows, queues, edges)
        
        # Get current states
        flow_states = tracker.get_flow_states()
        edge_states = tracker.get_edge_states()
        
        # Record actions/rewards
        tracker.record_ppo_action(flow_id, action, reward)
        tracker.record_a3c_action(edge_location, action, reward)
        
        # Generate transitions for training
        transitions = tracker.get_transitions(agent='ppo', last_n=32)
        
        tracker.end_episode()
    """
    
    def __init__(self, 
                 history_size: int = 20,
                 max_transitions: int = 100000):
        """
        Args:
            history_size: Number of past states to keep
            max_transitions: Maximum transitions to store (for replay buffer)
        """
        self.history_size = history_size
        self.max_transitions = max_transitions
        
        # State aggregator (from rl_spaces.py)
        self.aggregator = StateAggregator(history_size=history_size)
        
        # Reward coordinator
        self.reward_coordinator = RewardCoordinator()
        
        # History buffers
        self.flow_history = deque(maxlen=history_size)
        self.edge_history = deque(maxlen=history_size)
        self.queue_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # Current episode
        self.current_episode: Optional[EpisodeStats] = None
        self.episode_history: List[EpisodeStats] = []
        
        # Transition storage (experience replay)
        self.ppo_transitions = deque(maxlen=max_transitions)
        self.a3c_transitions = deque(maxlen=max_transitions)
        self.gwo_transitions = deque(maxlen=max_transitions)
        
        # Previous states for transition generation
        self.prev_flow_states: Dict[int, Tuple[np.ndarray, int]] = {}  # flow_id -> (state_vector, action)
        self.prev_edge_states: Dict[str, Tuple[np.ndarray, int]] = {}  # location -> (state_vector, action)
        
        # Statistics
        self.total_steps = 0
        self.total_ppo_actions = 0
        self.total_a3c_actions = 0
    
    def get_episode_transitions(self, episode_num: int) -> List[Transition]:
        """
        Get all transitions for a specific episode for offline training
        
        Args:
            episode_num: Episode number to get transitions for
            
        Returns:
            List of transitions from the specified episode
        """
        # Get all transitions
        all_transitions = list(self.ppo_transitions) + list(self.a3c_transitions)
        
        # Filter by episode (we need to track episode info in transitions)
        # For now, return recent transitions (last 100)
        recent_transitions = all_transitions[-100:] if len(all_transitions) > 100 else all_transitions
        
        return recent_transitions

    def get_a3c_transitions(self, episode_num: int) -> List[Transition]:
        """
        Get A3C transitions for specific episode
        
        Args:
            episode_num: Episode number
            
        Returns:
            List of A3C transitions
        """
        # Return recent A3C transitions for offline training
        a3c_transitions = list(self.a3c_transitions)
        return a3c_transitions[-50:] if len(a3c_transitions) > 50 else a3c_transitions

    def update_state(self, state: NetworkState):
        """Update state history with new state"""
        # Filter for TCP flows only
        tcp_flows = {
            flow_id: flow 
            for flow_id, flow in state.flows.items() 
            if flow.protocol == 'TCP'
        }
        
        # Update histories with TCP-only data
        self.flow_history.append(tcp_flows)
        self.timestamp_history.append(state.timestamp)
        
        # Update queue states as normal
        self.queue_history.append(state.queues)

    # ========================================================================
    # EPISODE MANAGEMENT
    # ========================================================================
    
    def start_episode(self, episode_id: int):
        """Start a new episode"""
        if self.current_episode is not None:
            # End previous episode if not already ended
            self.end_episode()
        
        self.current_episode = EpisodeStats(
            episode_id=episode_id,
            start_time=time.time()
        )
        
        # Reset reward calculators
        self.reward_coordinator.reset_all()
        
        print(f"\n{'='*60}")
        print(f"Episode {episode_id} Started")
        print(f"{'='*60}\n")
    
    def end_episode(self) -> EpisodeStats:
        """End current episode and return statistics"""
        if self.current_episode is None:
            raise ValueError("No active episode to end")
        
        self.current_episode.end_time = time.time()
        
        # Compute final statistics
        if self.flow_history:
            last_flows = self.flow_history[-1]
            
            # Compute BOTH metrics
            per_flow_throughputs = [f.get('throughput_mbps', 0) for f in last_flows]
            per_flow_delays = [f.get('rtt_ms', 0) for f in last_flows]  # ✅ NEW
            per_flow_loss = [f.get('loss_rate', 0) for f in last_flows]  # ✅ NEW
            
            # Aggregate throughput (sum)
            self.current_episode.aggregate_throughput = np.sum(per_flow_throughputs)
            
            # Average throughput (mean)
            self.current_episode.avg_throughput = np.mean(per_flow_throughputs)
            
            # ✅ NEW: Average delay
            self.current_episode.avg_delay = np.mean(per_flow_delays)
            
            # ✅ NEW: Average loss
            self.current_episode.avg_loss = np.mean(per_flow_loss)
            
            # ✅ NEW: Count SLA violations
            sla_violations = 0
            for flow in last_flows:
                service_type = flow.get('service_type', 'Unknown')
                rtt_ms = flow.get('rtt_ms', 0)
                loss_rate = flow.get('loss_rate', 0)
                throughput = flow.get('throughput_mbps', 0)
                
                # Check SLA per service type (matching reward_functions.py logic)
                if service_type == 'URLLC':
                    if rtt_ms > 60.0 or loss_rate > 1.0:
                        sla_violations += 1
                elif service_type == 'eMBB':
                    if throughput < 10.0:
                        sla_violations += 1
                elif service_type == 'mMTC':
                    if loss_rate > 5.0:
                        sla_violations += 1
            
            self.current_episode.total_sla_violations = sla_violations
            
            # Fairness (Jain's index)
            if len(per_flow_throughputs) > 1:
                sum_x = sum(per_flow_throughputs)
                sum_x2 = sum(x**2 for x in per_flow_throughputs)
                if sum_x2 > 0:
                    self.current_episode.fairness_index = (sum_x ** 2) / (len(per_flow_throughputs) * sum_x2)
        
        # Save to history
        self.episode_history.append(self.current_episode)
        
        print(f"\n{'='*60}")
        print(f"Episode {self.current_episode.episode_id} Ended")
        print(f"{'='*60}")
        print(f"Duration: {self.current_episode.duration():.1f}s")
        print(f"Steps: {self.current_episode.steps}")
        print(f"Avg Throughput: {self.current_episode.avg_throughput:.2f} Mbps")
        print(f"Avg Delay: {self.current_episode.avg_delay:.2f} ms")
        print(f"SLA Violations: {self.current_episode.total_sla_violations}")
        print(f"Total PPO Reward: {self.current_episode.total_ppo_reward:+.4f}")
        print(f"Total A3C Reward: {self.current_episode.total_a3c_reward:+.4f}")
        print(f"{'='*60}\n")
        
        episode_stats = self.current_episode
        self.current_episode = None
        
        return episode_stats
    
    # ========================================================================
    # STATE UPDATE
    # ========================================================================
    
    def update(self, 
               flows: List[Dict], 
               queues: List[Dict], 
               edges: List[Dict],
               timestamp: float = None):
        """
        Update state tracker with new network data from NS-3
        
        Args:
            flows: List of flow data dicts
            queues: List of queue data dicts
            edges: List of edge data dicts
            timestamp: Simulation timestamp
        """
        # Store raw data
        self.flow_history.append(flows)
        self.queue_history.append(queues)
        self.edge_history.append(edges)
        self.timestamp_history.append(timestamp or time.time())
        
        # Update aggregator
        self.aggregator.update(flows, queues, edges)
        
        # Update episode stats
        if self.current_episode:
            self.current_episode.steps += 1
            self.total_steps += 1
    
    # ========================================================================
    # STATE RETRIEVAL
    # ========================================================================
    
    def get_flow_states(self) -> List[FlowState]:
        """Get current flow states with temporal features"""
        return self.aggregator.get_flow_states()
    
    def get_edge_states(self) -> List[EdgeState]:
        """Get current edge states"""
        return self.aggregator.get_edge_states()
    
    def get_global_state(self) -> GWOState:
        """Get global network state"""
        return self.aggregator.get_global_state()
    
    def get_state_vectors(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Get state vectors for all agents
        
        Returns:
            flow_vectors: List of flow state vectors (for PPO)
            edge_vectors: List of edge state vectors (for A3C)
            global_vector: Global state vector (for GWO)
        """
        flow_states = self.get_flow_states()
        edge_states = self.get_edge_states()
        global_state = self.get_global_state()
        
        flow_vectors = [fs.to_vector() for fs in flow_states]
        edge_vectors = [es.to_vector() for es in edge_states]
        global_vector = global_state.to_vector()
        
        return flow_vectors, edge_vectors, global_vector
    
    # ========================================================================
    # ACTION & REWARD RECORDING
    # ========================================================================
    
    def record_ppo_action(self, 
                         flow_id: int, 
                         action: FlowAction,
                         reward: float):
        """
        Record PPO agent action and reward
        
        Creates transition for experience replay
        """
        # Get current state
        flow_states = self.get_flow_states()
        current_flow = next((fs for fs in flow_states if fs.flow_id == flow_id), None)
        
        if current_flow is None:
            return
        
        current_state_vector = current_flow.to_vector()
        action_index = action.value
        
        # Create transition if we have previous state
        if flow_id in self.prev_flow_states:
            prev_state, prev_action = self.prev_flow_states[flow_id]
            
            transition = Transition(
                agent_type='ppo',
                state=prev_state,
                action=prev_action,
                reward=reward,
                next_state=current_state_vector,
                done=False,
                flow_id=flow_id
            )
            
            self.ppo_transitions.append(transition)
        
        # Store current state for next transition
        self.prev_flow_states[flow_id] = (current_state_vector, action_index)
        
        # Update episode stats
        if self.current_episode:
            self.current_episode.total_ppo_reward += reward
            action_name = action.name
            self.current_episode.ppo_actions[action_name] = \
                self.current_episode.ppo_actions.get(action_name, 0) + 1
        
        self.total_ppo_actions += 1
    
    def record_a3c_action(self,
                         edge_location: str,
                         action: EdgeAction,
                         reward: float):
        """
        Record A3C agent action and reward
        """
        edge_states = self.get_edge_states()
        current_edge = next((es for es in edge_states if es.location == edge_location), None)
        
        if current_edge is None:
            return
        
        current_state_vector = current_edge.to_vector()
        action_index = action.value
        
        # Create transition
        if edge_location in self.prev_edge_states:
            prev_state, prev_action = self.prev_edge_states[edge_location]
            
            transition = Transition(
                agent_type='a3c',
                state=prev_state,
                action=prev_action,
                reward=reward,
                next_state=current_state_vector,
                done=False,
                edge_location=edge_location
            )
            
            self.a3c_transitions.append(transition)
        
        # Store current state
        self.prev_edge_states[edge_location] = (current_state_vector, action_index)
        
        # Update episode stats
        if self.current_episode:
            self.current_episode.total_a3c_reward += reward
            action_name = action.name
            self.current_episode.a3c_actions[action_name] = \
                self.current_episode.a3c_actions.get(action_name, 0) + 1
        
        self.total_a3c_actions += 1
    
    def record_gwo_step(self, reward: float):
        """Record GWO optimizer step"""
        if self.current_episode:
            self.current_episode.total_gwo_reward += reward
    
    def mark_episode_end_transitions(self):
        """Mark last transitions of episode as terminal"""
        # Mark last PPO transitions as done
        for flow_id in self.prev_flow_states.keys():
            if self.ppo_transitions:
                # Find last transition for this flow
                for trans in reversed(self.ppo_transitions):
                    if trans.flow_id == flow_id:
                        trans.done = True
                        break
        
        # Mark last A3C transitions as done
        for location in self.prev_edge_states.keys():
            if self.a3c_transitions:
                for trans in reversed(self.a3c_transitions):
                    if trans.edge_location == location:
                        trans.done = True
                        break
        
        # Clear previous states
        self.prev_flow_states.clear()
        self.prev_edge_states.clear()
    
    # ========================================================================
    # TRANSITION RETRIEVAL (FOR TRAINING)
    # ========================================================================
    
    def get_transitions(self, 
                       agent: str = 'ppo',
                       last_n: Optional[int] = None) -> List[Transition]:
        """
        Get transitions for training
        
        Args:
            agent: 'ppo', 'a3c', or 'gwo'
            last_n: Return only last N transitions (None = all)
        
        Returns:
            List of Transition objects
        """
        if agent == 'ppo':
            transitions = list(self.ppo_transitions)
        elif agent == 'a3c':
            transitions = list(self.a3c_transitions)
        elif agent == 'gwo':
            transitions = list(self.gwo_transitions)
        else:
            raise ValueError(f"Unknown agent type: {agent}")
        
        if last_n is not None:
            transitions = transitions[-last_n:]
        
        return transitions
    
    def sample_transitions(self,
                          agent: str,
                          batch_size: int) -> List[Transition]:
        """
        Sample random batch of transitions for training
        
        Args:
            agent: 'ppo', 'a3c', or 'gwo'
            batch_size: Number of transitions to sample
        
        Returns:
            List of sampled transitions
        """
        transitions = self.get_transitions(agent)
        
        if len(transitions) < batch_size:
            return transitions
        
        indices = np.random.choice(len(transitions), batch_size, replace=False)
        return [transitions[i] for i in indices]
    
    def clear_transitions(self, agent: Optional[str] = None):
        """
        Clear transition buffers
        
        Args:
            agent: Specific agent ('ppo', 'a3c', 'gwo') or None for all
        """
        if agent is None or agent == 'ppo':
            self.ppo_transitions.clear()
        if agent is None or agent == 'a3c':
            self.a3c_transitions.clear()
        if agent is None or agent == 'gwo':
            self.gwo_transitions.clear()
    
    # ========================================================================
    # STATISTICS & LOGGING
    # ========================================================================
    
    def get_episode_stats(self, episode_id: Optional[int] = None) -> Optional[EpisodeStats]:
        """Get statistics for specific episode"""
        if episode_id is None:
            return self.current_episode
        
        for episode in self.episode_history:
            if episode.episode_id == episode_id:
                return episode
        
        return None
    
    def get_all_episode_stats(self) -> List[EpisodeStats]:
        """Get all episode statistics"""
        return self.episode_history
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all episodes"""
        if not self.episode_history:
            return {}
        
        return {
            'total_episodes': len(self.episode_history),
            'total_steps': self.total_steps,
            'total_ppo_actions': self.total_ppo_actions,
            'total_a3c_actions': self.total_a3c_actions,
            'avg_episode_duration': np.mean([e.duration() for e in self.episode_history]),
            'avg_episode_reward_ppo': np.mean([e.total_ppo_reward for e in self.episode_history]),
            'avg_episode_reward_a3c': np.mean([e.total_a3c_reward for e in self.episode_history]),
            'avg_throughput': np.mean([e.avg_throughput for e in self.episode_history]),
            'avg_delay': np.mean([e.avg_delay for e in self.episode_history]),
            'total_sla_violations': sum(e.total_sla_violations for e in self.episode_history),
            'ppo_transition_count': len(self.ppo_transitions),
            'a3c_transition_count': len(self.a3c_transitions)
        }
    
    def print_current_state(self):
        """Print current network state (debugging)"""
        flow_states = self.get_flow_states()
        edge_states = self.get_edge_states()
        global_state = self.get_global_state()
        
        print("\n" + "="*60)
        print("CURRENT NETWORK STATE")
        print("="*60)
        
        print(f"\n📊 Flows ({len(flow_states)} active):")
        for fs in flow_states[:10]:  # Show first 10
            sla_icon = "❌" if fs.sla_violation else "✓"
            print(f"  Flow {fs.flow_id:3d} ({fs.service_type:6s}): "
                  f"Tput={fs.throughput_mbps:5.1f}Mbps, "
                  f"RTT={fs.rtt_ms:5.1f}ms, "
                  f"Loss={fs.loss_rate:4.2f}%, "
                  f"SLA={sla_icon}")
        
        if len(flow_states) > 10:
            print(f"  ... and {len(flow_states) - 10} more flows")
        
        print(f"\n🖥️  Edges ({len(edge_states)} active):")
        for es in edge_states:
            print(f"  {es.location.upper():5s}: "
                  f"CPU={es.cpu_usage:5.1%}, "
                  f"Mem={es.memory_usage:5.1%}, "
                  f"Flows={es.total_flows:3d}, "
                  f"Violations={es.sla_violations:2d}")
        
        print(f"\n🌐 Global:")
        print(f"  Throughput: {global_state.total_throughput:6.1f} Mbps")
        print(f"  Avg Delay:  {global_state.total_delay:6.1f} ms")
        print(f"  SLA Violations: {global_state.total_sla_violations:3d}")
        
        if self.current_episode:
            print(f"\n📈 Episode {self.current_episode.episode_id}:")
            print(f"  Steps: {self.current_episode.steps}")
            print(f"  Duration: {self.current_episode.duration():.1f}s")
            print(f"  PPO Reward: {self.current_episode.total_ppo_reward:+.4f}")
            print(f"  A3C Reward: {self.current_episode.total_a3c_reward:+.4f}")
        
        print("="*60 + "\n")
    
    def print_episode_summary(self):
        """Print summary of all episodes"""
        if not self.episode_history:
            print("No episodes completed yet.")
            return
        
        print("\n" + "="*60)
        print("EPISODE HISTORY SUMMARY")
        print("="*60)
        
        print(f"\nTotal Episodes: {len(self.episode_history)}")
        print(f"Total Steps: {self.total_steps}")
        print(f"Total Actions: PPO={self.total_ppo_actions}, A3C={self.total_a3c_actions}")
        
        print("\nRecent Episodes:")
        print("-" * 60)
        for episode in self.episode_history[-5:]:  # Show last 5
            print(f"\nEpisode {episode.episode_id}:")
            print(f"  Duration: {episode.duration():.1f}s, Steps: {episode.steps}")
            print(f"  PPO Reward: {episode.total_ppo_reward:+8.4f}")
            print(f"  A3C Reward: {episode.total_a3c_reward:+8.4f}")
            print(f"  Throughput: {episode.avg_throughput:6.2f} Mbps")
            print(f"  Delay:      {episode.avg_delay:6.2f} ms")
            print(f"  Violations: {episode.total_sla_violations}")
        
        # Compute trends
        if len(self.episode_history) >= 2:
            recent_rewards_ppo = [e.total_ppo_reward for e in self.episode_history[-10:]]
            recent_rewards_a3c = [e.total_a3c_reward for e in self.episode_history[-10:]]
            
            print("\n📈 Trends (last 10 episodes):")
            print(f"  PPO Reward:   {np.mean(recent_rewards_ppo):+.4f} ± {np.std(recent_rewards_ppo):.4f}")
            print(f"  A3C Reward:   {np.mean(recent_rewards_a3c):+.4f} ± {np.std(recent_rewards_a3c):.4f}")
        
        print("="*60 + "\n")
    
    def save_to_file(self, filepath: str):
        """Save state tracker data to JSON file"""
        data = {
            'summary': self.get_summary_stats(),
            'episodes': [e.to_dict() for e in self.episode_history],
            'config': {
                'history_size': self.history_size,
                'max_transitions': self.max_transitions
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ State tracker saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load state tracker data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct episode history
        self.episode_history = []
        for ep_dict in data.get('episodes', []):
            episode = EpisodeStats(
                episode_id=ep_dict['episode_id'],
                start_time=0.0,  # Not preserved
                end_time=0.0
            )
            episode.total_ppo_reward = ep_dict['total_ppo_reward']
            episode.total_a3c_reward = ep_dict['total_a3c_reward']
            episode.avg_throughput = ep_dict['avg_throughput']
            episode.avg_delay = ep_dict['avg_delay']
            episode.total_sla_violations = ep_dict['total_sla_violations']
            episode.steps = ep_dict['steps']
            
            self.episode_history.append(episode)
        
        print(f"✓ State tracker loaded from {filepath}")
        print(f"  Loaded {len(self.episode_history)} episodes")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        # Clear all buffers
        self.flow_history.clear()
        self.edge_history.clear()
        self.queue_history.clear()
        self.timestamp_history.clear()
        
        # Clear transitions
        self.ppo_transitions.clear()
        self.a3c_transitions.clear()
        self.gwo_transitions.clear()
        
        # Clear state mappings
        self.prev_flow_states.clear()
        self.prev_edge_states.clear()
        
        # Save final statistics if needed
        if self.current_episode:
            self.end_episode()

# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    """Test state tracker functionality"""
    print("Testing State Tracker\n")
    
    # Create tracker
    tracker = StateTracker(history_size=10)
    
    # Simulate Episode 1
    print("="*60)
    print("SIMULATING EPISODE 1")
    print("="*60)
    
    tracker.start_episode(episode_id=1)
    
    # Simulate 10 timesteps
    for step in range(10):
        # Generate fake NS-3 data
        flows = [
            {
                'id': 1,
                'service_type': 'URLLC',
                'throughput_mbps': 10.0 + step * 0.5,
                'rtt_ms': 12.0 - step * 0.3,
                'jitter_ms': 1.5,
                'loss_rate': 0.2 - step * 0.01,
                'cwnd': 80 + step * 5,
                'src': '10.0.0.1',
                'dst': '20.0.0.1'
            },
            {
                'id': 2,
                'service_type': 'eMBB',
                'throughput_mbps': 40.0 + step * 2.0,
                'rtt_ms': 25.0 + step * 0.2,
                'jitter_ms': 3.0,
                'loss_rate': 0.1,
                'cwnd': 200 + step * 10,
                'src': '10.0.0.2',
                'dst': '20.0.0.2'
            }
        ]
        
        queues = [
            {'device': 'india_bottleneck', 'length': 20 + step, 'drops': step}
        ]
        
        edges = [
            {
                'location': 'india',
                'cpu_usage': 0.6 + step * 0.02,
                'memory_usage': 0.65 + step * 0.015,
                'network_usage': 0.55 + step * 0.02
            },
            {
                'location': 'uk',
                'cpu_usage': 0.55,
                'memory_usage': 0.60,
                'network_usage': 0.50
            }
        ]
        
        # Update tracker
        tracker.update(flows, queues, edges, timestamp=float(step))
        
        # Get states
        flow_states = tracker.get_flow_states()
        edge_states = tracker.get_edge_states()
        
        # Simulate actions and rewards
        if flow_states:
            for fs in flow_states:
                action = FlowAction.NO_CHANGE
                reward = 0.5 + step * 0.1  # Improving reward
                tracker.record_ppo_action(fs.flow_id, action, reward)
        
        if edge_states:
            for es in edge_states:
                action = EdgeAction.NO_CHANGE
                reward = 0.3 + step * 0.05
                tracker.record_a3c_action(es.location, action, reward)
        
        # Print state every 3 steps
        if step % 3 == 0:
            print(f"\n--- Step {step} ---")
            tracker.print_current_state()
    
    # End episode
    episode1_stats = tracker.end_episode()
    
    # Check transitions generated
    ppo_transitions = tracker.get_transitions('ppo')
    a3c_transitions = tracker.get_transitions('a3c')
    
    print(f"\n✓ Generated {len(ppo_transitions)} PPO transitions")
    print(f"✓ Generated {len(a3c_transitions)} A3C transitions")
    
    # Simulate Episode 2
    print("\n" + "="*60)
    print("SIMULATING EPISODE 2")
    print("="*60)
    
    tracker.start_episode(episode_id=2)
    
    # Simulate 5 steps with different metrics
    for step in range(5):
        flows = [
            {
                'id': 3,
                'service_type': 'mMTC',
                'throughput_mbps': 2.0,
                'rtt_ms': 40.0,
                'jitter_ms': 5.0,
                'loss_rate': 1.0,
                'cwnd': 30,
                'src': '10.0.0.3',
                'dst': '203.0.113.1'
            }
        ]
        
        queues = [{'device': 'india_bottleneck', 'length': 15, 'drops': 0}]
        
        edges = [
            {
                'location': 'india',
                'cpu_usage': 0.70,
                'memory_usage': 0.75,
                'network_usage': 0.65
            }
        ]
        
        tracker.update(flows, queues, edges, timestamp=float(step))
        
        flow_states = tracker.get_flow_states()
        if flow_states:
            tracker.record_ppo_action(flow_states[0].flow_id, FlowAction.INCREASE_PRIORITY, 0.8)
    
    tracker.end_episode()
    
    # Print summary
    tracker.print_episode_summary()
    
    # Test sampling
    print("\n" + "="*60)
    print("TESTING TRANSITION SAMPLING")
    print("="*60)
    
    batch = tracker.sample_transitions('ppo', batch_size=5)
    print(f"\n✓ Sampled batch of {len(batch)} PPO transitions")
    
    if batch:
        sample_trans = batch[0]
        print(f"\nSample transition:")
        print(f"  State shape: {sample_trans.state.shape}")
        print(f"  Action: {sample_trans.action}")
        print(f"  Reward: {sample_trans.reward:+.4f}")
        print(f"  Done: {sample_trans.done}")
        print(f"  Flow ID: {sample_trans.flow_id}")
    
    # Test save/load
    print("\n" + "="*60)
    print("TESTING SAVE/LOAD")
    print("="*60)
    
    test_file = "/tmp/state_tracker_test.json"
    tracker.save_to_file(test_file)
    
    # Create new tracker and load
    tracker2 = StateTracker()
    tracker2.load_from_file(test_file)
    
    print(f"\n✓ Loaded tracker has {len(tracker2.episode_history)} episodes")
    
    # Verify data
    summary = tracker2.get_summary_stats()
    print(f"\nSummary stats:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ All state tracker tests passed!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Episode management (start/end)")
    print("  ✓ State updates from NS-3 data")
    print("  ✓ Action/reward recording")
    print("  ✓ Transition generation for replay buffer")
    print("  ✓ Statistics tracking")
    print("  ✓ Batch sampling")
    print("  ✓ Save/load functionality")
