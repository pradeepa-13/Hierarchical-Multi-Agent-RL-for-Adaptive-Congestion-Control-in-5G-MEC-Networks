#!/usr/bin/env python3
"""
reward_functions.py
Reward Function Definitions for 5G MEC Multi-Agent RL System

Implements:
1. PPO Flow-level rewards (balanced throughput + QoS)
2. A3C Edge-level rewards (resource efficiency + SLA compliance)
3. Global reward for GWO (system-wide optimization)
4. Penalty shaping for constraint violations

Place in: ai_controller/reward_functions.py
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from config import RewardConfig
from rl_spaces import FlowState, EdgeState, GWOState, ServiceType



# ============================================================================
# PPO FLOW-LEVEL REWARD FUNCTION
# ============================================================================
class PPOFlowReward:
    """
    Reward calculator for PPO agent (per-flow control
    Balances:
    - Throughput maximization
    - Delay minimization (especially URLLC)
    - Loss rate minimization
    - Fairness across flows
    
    Returns reward in range [-10, +10] approximately
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.config.validate()
        
        # Track previous state for delta rewards
        self.prev_states: Dict[int, FlowState] = {}
    
    def compute_reward(self, 
                       current_state: FlowState,
                       all_flows: List[FlowState]) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for a single flow's action
        
        Returns:
            reward: Scalar reward value
            components: Dict of reward breakdown for logging
        """
        components = {}
        
        # ====== COMPONENT 1: Throughput Reward ======
        throughput_reward = self._compute_throughput_reward(current_state)
        components['throughput'] = throughput_reward
        
        # ====== COMPONENT 2: Delay Reward ======
        delay_reward = self._compute_delay_reward(current_state)
        components['delay'] = delay_reward
        
        # ====== COMPONENT 3: Loss Reward ======
        loss_reward = self._compute_loss_reward(current_state)
        components['loss'] = loss_reward
        
        # ====== COMPONENT 4: Fairness Reward ======
        fairness_reward = self._compute_fairness_reward(current_state, all_flows)
        components['fairness'] = fairness_reward
        
        # ====== COMPONENT 5: SLA Violation Penalties ======
        violation_penalty = self._compute_violation_penalty(current_state)
        components['violation_penalty'] = violation_penalty
        
        # # ====== COMPONENT 6: Improvement Bonus ======
        # improvement_bonus = self._compute_improvement_bonus(current_state)
        # components['improvement'] = improvement_bonus
        
        # Weighted sum
        reward = (
            self.config.ppo_throughput_weight * throughput_reward +
            self.config.ppo_delay_weight * delay_reward +
            self.config.ppo_loss_weight * loss_reward +
            self.config.ppo_fairness_weight * fairness_reward +
            violation_penalty 
            # improvement_bonus
        )
        
        self.prev_states[current_state.flow_id] = current_state
        components['total'] = reward
        
        return reward, components
    
    def _compute_throughput_reward(self, state: FlowState) -> float:
        """
        Throughput reward with service-aware targets
        
        Returns: [0, 10]
        """
        throughput = state.throughput_mbps
        
        if state.service_type == ServiceType.URLLC.value:
            # URLLC: Low throughput expected (5-15 Mbps target)
            target = 10.0
            optimal = 15.0
            
        elif state.service_type == ServiceType.EMBB.value:
            # eMBB: High throughput critical (20-60 Mbps target)
            target = 20.0
            optimal = 60.0
            
        elif state.service_type == ServiceType.MMTC.value:
            # mMTC: Very low throughput (0.5-2 Mbps target)
            target = 1.0
            optimal = 2.0
            
        else:
            # Background: Moderate throughput (10-30 Mbps)
            target = 10.0
            optimal = 30.0
        
        # âœ… Normalized sigmoid-like reward
        if throughput < target:
            # Below target: negative reward
            reward = -1.0 + (throughput / target)
        else:
            # Above target: positive reward (saturates at +1)
            excess = (throughput - target) / (optimal - target)
            reward = min(excess, 1.0)
        
        return np.clip(reward, -1.0, 1.0)
    
    def _compute_delay_reward(self, state: FlowState) -> float:
        """
        OPTIMIZED: Normalized delay reward [-1, +1]
        
        Returns:
            +1.0: Near-zero delay
             0.0: At target delay
            -1.0: Severely violating delay
        """
        rtt_ms = state.rtt_ms
        
        # Service-specific delay targets
        if state.service_type == ServiceType.URLLC.value:
            target = self.config.urllc_delay_target_ms  # 50ms
            critical = target * 2.0  # 100ms (severe violation)
            
        elif state.service_type == ServiceType.EMBB.value:
            target = self.config.embb_delay_target_ms  # 100ms
            critical = target * 1.5  # 150ms
            
        else:
            target = self.config.mmtc_delay_target_ms  # 150ms
            critical = target * 1.5  # 225ms
        
        # âœ… Normalized exponential decay
        if rtt_ms <= target:
            # Below target: positive reward
            reward = 1.0 - (rtt_ms / target) * 0.5  # [0.5, 1.0]
        else:
            # Above target: negative reward (exponential penalty)
            violation_ratio = (rtt_ms - target) / (critical - target)
            reward = -min(violation_ratio, 1.0)
        
        return np.clip(reward, -1.0, 1.0)
    
    def _compute_loss_reward(self, state: FlowState) -> float:
        """
        Packet loss reward
        
        Returns: [-1, 1]
        """
        loss_rate = state.loss_rate  # Percentage
        
        # Service-specific loss tolerance
        if state.service_type == ServiceType.URLLC.value:
            threshold = 0.1   # 0.1% (very strict)
            critical = 1.0    # 1% (severe)
        elif state.service_type == ServiceType.EMBB.value:
            threshold = 0.5   # 0.5%
            critical = 2.0    # 2%
        else:
            threshold = 2.0   # 2%
            critical = 5.0    # 5%
        
        # âœ… Normalized reward
        if loss_rate <= threshold:
            # Below threshold: positive reward
            reward = 1.0 - (loss_rate / threshold) * 0.5
        else:
            # Above threshold: negative reward
            violation_ratio = (loss_rate - threshold) / (critical - threshold)
            reward = -min(violation_ratio, 1.0)
        
        return np.clip(reward, -1.0, 1.0)
    
    def _compute_fairness_reward(self, 
                                  state: FlowState,
                                  all_flows: List[FlowState]) -> float:
        """
        Fairness reward using Jain's Fairness Index
        
        Prevents one flow from starving others
        Returns: [0, 1]
        """
        if len(all_flows) <= 1:
            return 0.5  # Neutral
        
        same_type_flows = [
            f for f in all_flows 
            if f.service_type == state.service_type
        ]
        
        if len(same_type_flows) <= 1:
            return 0.5
        
        throughputs = [f.throughput_mbps for f in same_type_flows]
        
        # Jain's Fairness Index: (sum x)^2 / (n * sum x^2)
        sum_x = sum(throughputs)
        sum_x2 = sum(x**2 for x in throughputs)
        
        if sum_x2 == 0:
            return 0.5
        
        fairness_index = (sum_x ** 2) / (len(throughputs) * sum_x2)
        
        # âœ… Scale to [0, 1] (fairness_index already in [0, 1])
        return fairness_index
    
    def _compute_violation_penalty(self, state: FlowState) -> float:
        """
        Heavy penalties for SLA violations
        
        Returns: [-20, 0]
        """
        if not state.sla_violation:
            return 0.0
        
        penalty = 0.0
        
        if state.service_type == ServiceType.URLLC.value:
            # URLLC violation is CRITICAL
            if state.rtt_ms > self.config.urllc_delay_target_ms:
                violation_severity = (state.rtt_ms / self.config.urllc_delay_target_ms) - 1.0
                penalty -= min(violation_severity, 1.0)  # Cap at -1.0
            
            if state.loss_rate > 1.0:
                penalty -= 0.5
        
        elif state.service_type == ServiceType.EMBB.value:
            # eMBB throughput violation
            if state.throughput_mbps < 10.0:
                penalty -= 0.5
        
        # General severe loss penalty
        if state.loss_rate > 2.0:
            penalty -= 0.3
        
        return max(penalty, -1.0)  # Clamp to [-1, 0]
    
    
    def reset(self):
        """Clear history (call at episode end)"""
        self.prev_states.clear()

# ============================================================================
# A3C EDGE-LEVEL REWARD FUNCTION
# ============================================================================
class A3CEdgeReward:
    """
    Reward calculator for A3C agent (edge server management)
    
    Balances:
    - SLA compliance across all flows
    - Resource utilization efficiency
    - Load balancing between edges
    
    Returns reward in range [-10, +10] approximately
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.config.validate()
        
        self.prev_edge_states: Dict[str, EdgeState] = {}
    
    def compute_reward(self,
                       current_state: EdgeState,
                       all_edges: List[EdgeState]) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for edge server management
        
        Returns:
            reward: Scalar reward value
            components: Dict of reward breakdown
        """
        components = {}
        
        # ====== COMPONENT 1: SLA Compliance Reward ======
        sla_reward = self._compute_sla_compliance_reward(current_state)
        components['sla_compliance'] = sla_reward
        
        # ====== COMPONENT 2: Resource Efficiency Reward ======
        resource_reward = self._compute_resource_efficiency_reward(current_state)
        components['resource_efficiency'] = resource_reward
        
        # ====== COMPONENT 3: Load Balance Reward ======
        balance_reward = self._compute_load_balance_reward(current_state, all_edges)
        components['load_balance'] = balance_reward
        
        # ====== COMPONENT 4: Violation Penalties ======
        violation_penalty = self._compute_edge_violation_penalty(current_state)
        components['violation_penalty'] = violation_penalty
        
        # Weighted sum
        reward = (
            self.config.a3c_sla_compliance_weight * sla_reward +
            self.config.a3c_resource_efficiency_weight * resource_reward +
            self.config.a3c_load_balance_weight * balance_reward +
            violation_penalty
        )
        
        reward *= 2.0  # Brings range to approximately [-4, +4]
        
        self.prev_edge_states[current_state.location] = current_state
        components['total'] = reward
        
        return reward, components
    
    def _compute_sla_compliance_reward(self, state: EdgeState) -> float:
        """
        Reward based on SLA violation rate
        
        Returns: [-10, 10]
        """
        total_flows = max(state.total_flows, 1)
        violation_rate = state.sla_violations / total_flows
        
        if violation_rate == 0.0:
            return +2.0  # âœ… Perfect compliance
        elif violation_rate < 0.05:
            return +1.5 - (violation_rate * 30.0)  # [+1.5, +0.0]
        elif violation_rate < 0.1:
            return -1.0 * violation_rate * 10.0  # [0.0, -1.0]
        else:
            return -2.0  # Critical violation
    
    def _compute_resource_efficiency_reward(self, state: EdgeState) -> float:
        """
        Reward efficient resource usage (not too high, not too low)
        
        Optimal CPU/memory usage: 60-80%
        Returns: [0, 10]
        """
        cpu_usage = state.cpu_usage
        mem_usage = state.memory_usage
        
        optimal_usage = 0.7
        tolerance = 0.1
        
        # âœ… Gaussian-like reward centered at optimal
        cpu_deviation = abs(cpu_usage - optimal_usage)
        mem_deviation = abs(mem_usage - optimal_usage)
        
        cpu_score = 2.0 * np.exp(-(cpu_deviation ** 2) / (2 * tolerance ** 2))
        mem_score = 2.0 * np.exp(-(mem_deviation ** 2) / (2 * tolerance ** 2))
        
        return (cpu_score + mem_score) / 2.0  # Average: [-2, +2]
    
    def _compute_load_balance_reward(self,
                                     state: EdgeState,
                                     all_edges: List[EdgeState]) -> float:
        """
        Reward balanced load distribution across edges
        
        Returns: [0, 10]
        """
        if len(all_edges) <= 1:
            return 1.0  # Neutral
        
        cpu_usages = [e.cpu_usage for e in all_edges]
        mem_usages = [e.memory_usage for e in all_edges]
        
        # Coefficient of variation (lower is better)
        cpu_cv = np.std(cpu_usages) / (np.mean(cpu_usages) + 1e-6)
        mem_cv = np.std(mem_usages) / (np.mean(mem_usages) + 1e-6)
        
        avg_cv = (cpu_cv + mem_cv) / 2.0
        
        # âœ… Exponential reward (decays with imbalance)
        return 2.0 * np.exp(-3.0 * avg_cv)
    
    def _compute_edge_violation_penalty(self, state: EdgeState) -> float:
        """
        Penalties for edge-level issues
        
        Returns: [-15, 0]
        """
        penalty = 0.0
        
        # Overutilization penalty
        if state.cpu_usage > 0.9:
            penalty -= 1.0 * (state.cpu_usage - 0.9) * 10.0
        
        if state.memory_usage > 0.9:
            penalty -= 1.0 * (state.memory_usage - 0.9) * 10.0
        
        # Network saturation
        if state.network_usage > 0.85:
            penalty -= 0.5 * (state.network_usage - 0.85) * 6.0
        
        # High delay penalty
        if state.avg_delay_ms > 85.0:
            penalty -= 0.5 * (state.avg_delay_ms / 85.0 - 1.0)
        
        return max(penalty, -2.0)
    
    
    def reset(self):
        """Clear history"""
        self.prev_edge_states.clear()


# ============================================================================
# GWO GLOBAL REWARD FUNCTION
# ============================================================================
class GWOGlobalReward:
    """
    Global reward for Grey Wolf Optimizer
    
    Optimizes entire network performance
    Returns: [-10, 10]
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.config.validate()
        
        self.prev_global_state = None
    
    def compute_reward(self, state: GWOState) -> Tuple[float, Dict[str, float]]:
        """
        Compute global system reward
        
        Returns:
            reward: Scalar reward
            components: Breakdown dict
        """
        components = {}
        
        # ====== COMPONENT 1: Total Throughput ======
        throughput_reward = self._compute_global_throughput_reward(state)
        components['throughput'] = throughput_reward
        
        # ====== COMPONENT 2: Average Delay ======
        delay_reward = self._compute_global_delay_reward(state)
        components['delay'] = delay_reward
        
        # ====== COMPONENT 3: SLA Compliance ======
        sla_reward = self._compute_global_sla_reward(state)
        components['sla'] = sla_reward
        
        # ====== COMPONENT 4: Resource Utilization ======
        resource_reward = self._compute_global_resource_reward(state)
        components['resource'] = resource_reward
        
        # ====== COMPONENT 5: System Stability ======
        stability_bonus = self._compute_stability_bonus(state)
        components['stability'] = stability_bonus
        
        # Weighted sum
        reward = (
            self.config.gwo_throughput_weight * throughput_reward +
            self.config.gwo_delay_weight * delay_reward +
            self.config.gwo_sla_weight * sla_reward +
            self.config.gwo_resource_weight * resource_reward +
            stability_bonus
        )
        
        self.prev_global_state = state
        components['total'] = reward
        
        return reward, components
    
    def _compute_global_throughput_reward(self, state: GWOState) -> float:
        """Global throughput maximization"""
        # Target: 400 Mbps (realistic for this topology)
        target = 400.0
        throughput = state.total_throughput
        
        normalized = min(throughput / target, 1.5)
        reward = 10.0 * normalized
        
        return reward
    
    def _compute_global_delay_reward(self, state: GWOState) -> float:
        """Global delay minimization"""
        delay = state.total_delay
        
        if delay < 30.0:
            return 10.0 * np.exp(-delay / 20.0)
        else:
            return -5.0 * (delay / 30.0 - 1.0)
    
    def _compute_global_sla_reward(self, state: GWOState) -> float:
        """Global SLA compliance"""
        total_violations = state.total_sla_violations
        
        if total_violations == 0:
            return 10.0
        elif total_violations <= 3:
            return 7.0 - total_violations
        else:
            return -5.0 * np.log1p(total_violations)
    
    def _compute_global_resource_reward(self, state: GWOState) -> float:
        """Global resource efficiency"""
        avg_cpu = (state.india_cpu + state.uk_cpu) / 2.0
        avg_mem = (state.india_memory + state.uk_memory) / 2.0
        
        # Target: 70% average
        optimal = 0.7
        
        cpu_score = 10.0 * np.exp(-((avg_cpu - optimal) ** 2) / 0.1)
        mem_score = 10.0 * np.exp(-((avg_mem - optimal) ** 2) / 0.1)
        
        return (cpu_score + mem_score) / 2.0
    
    def _compute_stability_bonus(self, state: GWOState) -> float:
        """Bonus for stable performance over time"""
        if self.prev_global_state is None:
            return 0.0
        
        bonus = 0.0
        
        # Reward consistent low violations
        if state.total_sla_violations == 0 and self.prev_global_state.total_sla_violations == 0:
            bonus += 2.0
        
        # Reward stable throughput (no wild swings)
        throughput_delta = abs(state.total_throughput - self.prev_global_state.total_throughput)
        if throughput_delta < 50.0:
            bonus += 1.0
        
        return bonus
    
    def reset(self):
        """Clear history"""
        self.prev_global_state = None


# ============================================================================
# REWARD COORDINATOR
# ============================================================================
class RewardCoordinator:
    """
    Coordinates all reward calculators
    
    Usage:
        coordinator = RewardCoordinator()
        
        # PPO agent rewards
        ppo_rewards = coordinator.compute_ppo_rewards(flow_states)
        
        # A3C agent rewards
        a3c_rewards = coordinator.compute_a3c_rewards(edge_states)
        
        # GWO global reward
        gwo_reward = coordinator.compute_gwo_reward(global_state)
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        
        self.ppo_calculator = PPOFlowReward(self.config)
        self.a3c_calculator = A3CEdgeReward(self.config)
        self.gwo_calculator = GWOGlobalReward(self.config)
    
    def compute_ppo_rewards(self, 
                           flow_states: List[FlowState]) -> List[Tuple[float, Dict]]:
        """
        Compute rewards for all flows (PPO agent)
        
        Returns: List of (reward, components) tuples
        """
        rewards = []
        for flow_state in flow_states:
            reward, components = self.ppo_calculator.compute_reward(
                flow_state, flow_states
            )
            rewards.append((reward, components))
        
        return rewards
    
    def compute_a3c_rewards(self,
                           edge_states: List[EdgeState]) -> List[Tuple[float, Dict]]:
        """
        Compute rewards for all edges (A3C agent)
        
        Returns: List of (reward, components) tuples
        """
        rewards = []
        for edge_state in edge_states:
            reward, components = self.a3c_calculator.compute_reward(
                edge_state, edge_states
            )
            rewards.append((reward, components))
        
        return rewards
    
    def compute_gwo_reward(self, global_state: GWOState) -> Tuple[float, Dict]:
        """
        Compute global reward (GWO optimizer)
        
        Returns: (reward, components) tuple
        """
        return self.gwo_calculator.compute_reward(global_state)
    
    def reset_all(self):
        """Reset all calculators (call at episode end)"""
        self.ppo_calculator.reset()
        self.a3c_calculator.reset()
        self.gwo_calculator.reset()
    
    def print_reward_summary(self,
                            ppo_rewards: List[Tuple[float, Dict]],
                            a3c_rewards: List[Tuple[float, Dict]],
                            gwo_reward: Tuple[float, Dict]):
        """Pretty-print reward breakdown"""
        print("\n" + "="*60)
        print("REWARD SUMMARY")
        print("="*60)
        
        # PPO summary
        if ppo_rewards:
            avg_ppo = np.mean([r[0] for r in ppo_rewards])
            print(f"\n📊 PPO Flow Rewards ({len(ppo_rewards)} flows):")
            print(f"  Average: {avg_ppo:+.4f}")
            print(f"  Range: [{min(r[0] for r in ppo_rewards):+.4f}, "
                  f"{max(r[0] for r in ppo_rewards):+.4f}]")
            
            # Show component breakdown of first flow
            if ppo_rewards:
                sample_components = ppo_rewards[0][1]
                print(f"  Sample breakdown:")
                for key, value in sample_components.items():
                    if key != 'total':
                        print(f"    {key}: {value:+.4f}")
        
        # A3C summary
        if a3c_rewards:
            avg_a3c = np.mean([r[0] for r in a3c_rewards])
            print(f"\n🖥️  A3C Edge Rewards ({len(a3c_rewards)} edges):")
            print(f"  Average: {avg_a3c:+.4f}")
            for i, (reward, components) in enumerate(a3c_rewards):
                print(f"  Edge {i}: {reward:+.4f}")
        
        # GWO summary
        gwo_value, gwo_components = gwo_reward
        print(f"\n🌐 GWO Global Reward: {gwo_value:+.4f}")
        print(f"  Components:")
        for key, value in gwo_components.items():
            if key != 'total':
                print(f"    {key}: {value:+.4f}")
        
        print("="*60 + "\n")


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    """Test reward functions"""
    from rl_spaces import FlowState, EdgeState, GWOState
    
    print("Testing Reward Functions\n")
    
    config = RewardConfig()
    coordinator = RewardCoordinator(config)
    
    # Test PPO rewards
    print("=" * 60)
    print("Testing PPO Flow Rewards")
    print("=" * 60)
    
    test_flows = [
        FlowState(
            flow_id=1, service_type='URLLC',
            throughput_mbps=15.0, rtt_ms=45.0, jitter_ms=5.0,
            loss_rate=0.05, cwnd=120, sla_violation=False,
            throughput_trend=0, rtt_trend=0,
            queue_length=15, queue_drops=0
        ),
        FlowState(
            flow_id=2, service_type='eMBB',
            throughput_mbps=45.0, rtt_ms=25.0, jitter_ms=3.0,
            loss_rate=0.2, cwnd=200, sla_violation=False,
            throughput_trend=0, rtt_trend=0,
            queue_length=25, queue_drops=2
        ),
        FlowState(
            flow_id=3, service_type='URLLC',
            throughput_mbps=8.0, rtt_ms=55.0, jitter_ms=8.0,
            loss_rate=0.15, cwnd=80, sla_violation=True,  # Violating URLLC delay
            throughput_trend=0, rtt_trend=0,
            queue_length=35, queue_drops=5
        )
    ]
    
    ppo_rewards = coordinator.compute_ppo_rewards(test_flows)
    
    print("\nPPO Reward Results:")
    for i, (reward, components) in enumerate(ppo_rewards):
        flow = test_flows[i]
        print(f"\n  Flow {flow.flow_id} ({flow.service_type}):")
        print(f"    Total Reward: {reward:+.4f}")
        print(f"    Components:")
        for key, value in components.items():
            if key != 'total':
                print(f"      {key}: {value:+.4f}")
    
    # Test A3C rewards
    print("\n" + "=" * 60)
    print("Testing A3C Edge Rewards")
    print("=" * 60)
    
    test_edges = [
        EdgeState(
            location='india',
            cpu_usage=0.68, memory_usage=0.72, network_usage=0.65,
            total_flows=15, embb_flows=5, urllc_flows=6, mmtc_flows=4,
            avg_delay_ms=22.0, sla_violations=1
        ),
        EdgeState(
            location='uk',
            cpu_usage=0.55, memory_usage=0.60, network_usage=0.48,
            total_flows=12, embb_flows=4, urllc_flows=5, mmtc_flows=3,
            avg_delay_ms=18.5, sla_violations=0
        )
    ]
    
    a3c_rewards = coordinator.compute_a3c_rewards(test_edges)
    
    print("\nA3C Reward Results:")
    for i, (reward, components) in enumerate(a3c_rewards):
        edge = test_edges[i]
        print(f"\n  Edge {edge.location}:")
        print(f"    Total Reward: {reward:+.4f}")
        print(f"    Components:")
        for key, value in components.items():
            if key != 'total':
                print(f"      {key}: {value:+.4f}")
    
    # Test GWO reward
    print("\n" + "=" * 60)
    print("Testing GWO Global Reward")
    print("=" * 60)
    
    test_gwo = GWOState(
        total_throughput=425.0,
        total_delay=23.5,
        total_loss=0.3,
        india_cpu=0.68,
        india_memory=0.72,
        uk_cpu=0.55,
        uk_memory=0.60,
        backbone_bandwidth_usage=0.62,
        total_sla_violations=2,
        urllc_violations=1,
        embb_violations=1,
        india_to_uk_flows=15,
        uk_to_india_flows=12,
        to_cloud_flows=8
    )
    
    gwo_reward, gwo_components = coordinator.compute_gwo_reward(test_gwo)
    
    print(f"\nGWO Reward Result:")
    print(f"  Total Reward: {gwo_reward:+.4f}")
    print(f"  Components:")
    for key, value in gwo_components.items():
        if key != 'total':
            print(f"    {key}: {value:+.4f}")
    
    # Test comprehensive summary
    print("\n" + "=" * 60)
    print("Comprehensive Reward Summary")
    print("=" * 60)
    
    coordinator.print_reward_summary(ppo_rewards, a3c_rewards, (gwo_reward, gwo_components))
    
    # Test reward evolution over time
    print("\n" + "=" * 60)
    print("Testing Reward Evolution (Improvement Bonus)")
    print("=" * 60)
    
    # Simulate improvement over 3 timesteps
    urllc_flow_t0 = FlowState(
        flow_id=100, service_type='URLLC',
        throughput_mbps=5.0, rtt_ms=15.0, jitter_ms=2.5,
        loss_rate=0.5, cwnd=50, sla_violation=True,
        throughput_trend=0, rtt_trend=0,
        queue_length=40, queue_drops=8
    )
    
    urllc_flow_t1 = FlowState(
        flow_id=100, service_type='URLLC',
        throughput_mbps=8.0, rtt_ms=11.0, jitter_ms=1.8,
        loss_rate=0.2, cwnd=80, sla_violation=True,  # Still violating
        throughput_trend=0, rtt_trend=0,
        queue_length=25, queue_drops=3
    )
    
    urllc_flow_t2 = FlowState(
        flow_id=100, service_type='URLLC',
        throughput_mbps=12.0, rtt_ms=8.5, jitter_ms=1.2,
        loss_rate=0.05, cwnd=120, sla_violation=False,  # Fixed!
        throughput_trend=0, rtt_trend=0,
        queue_length=15, queue_drops=0
    )
    
    print("\nFlow 100 Evolution:")
    
    # t=0
    reward_t0, comp_t0 = coordinator.ppo_calculator.compute_reward(urllc_flow_t0, [urllc_flow_t0])
    print(f"\n  t=0 (Initial - Poor Performance):")
    print(f"    RTT: {urllc_flow_t0.rtt_ms:.1f}ms, Loss: {urllc_flow_t0.loss_rate:.2f}%")
    print(f"    Reward: {reward_t0:+.4f}")
    print(f"    Improvement Bonus: {comp_t0.get('improvement', 0.0):+.4f}")
    
    # t=1
    reward_t1, comp_t1 = coordinator.ppo_calculator.compute_reward(urllc_flow_t1, [urllc_flow_t1])
    print(f"\n  t=1 (Improving):")
    print(f"    RTT: {urllc_flow_t1.rtt_ms:.1f}ms, Loss: {urllc_flow_t1.loss_rate:.2f}%")
    print(f"    Reward: {reward_t1:+.4f} (Δ = {reward_t1 - reward_t0:+.4f})")
    print(f"    Improvement Bonus: {comp_t1.get('improvement', 0.0):+.4f}")
    
    # t=2
    reward_t2, comp_t2 = coordinator.ppo_calculator.compute_reward(urllc_flow_t2, [urllc_flow_t2])
    print(f"\n  t=2 (SLA Compliant!):")
    print(f"    RTT: {urllc_flow_t2.rtt_ms:.1f}ms, Loss: {urllc_flow_t2.loss_rate:.2f}%")
    print(f"    Reward: {reward_t2:+.4f} (Δ = {reward_t2 - reward_t1:+.4f})")
    print(f"    Improvement Bonus: {comp_t2.get('improvement', 0.0):+.4f}")
    print(f"\n  Total Improvement: {reward_t2 - reward_t0:+.4f}")
    
    # Test edge overload scenario
    print("\n" + "=" * 60)
    print("Testing Edge Overload Scenario")
    print("=" * 60)
    
    overloaded_edge = EdgeState(
        location='india',
        cpu_usage=0.95, memory_usage=0.92, network_usage=0.88,
        total_flows=35, embb_flows=15, urllc_flows=12, mmtc_flows=8,
        avg_delay_ms=65.0, sla_violations=8
    )
    
    normal_edge = EdgeState(
        location='uk',
        cpu_usage=0.45, memory_usage=0.50, network_usage=0.42,
        total_flows=8, embb_flows=3, urllc_flows=3, mmtc_flows=2,
        avg_delay_ms=15.0, sla_violations=0
    )
    
    overload_rewards = coordinator.compute_a3c_rewards([overloaded_edge, normal_edge])
    
    print("\nOverload Scenario Results:")
    print(f"\n  Overloaded Edge (India):")
    print(f"    CPU: {overloaded_edge.cpu_usage:.1%}, Flows: {overloaded_edge.total_flows}")
    print(f"    Reward: {overload_rewards[0][0]:+.4f}")
    print(f"    Violation Penalty: {overload_rewards[0][1]['violation_penalty']:+.4f}")
    
    print(f"\n  Normal Edge (UK):")
    print(f"    CPU: {normal_edge.cpu_usage:.1%}, Flows: {normal_edge.total_flows}")
    print(f"    Reward: {overload_rewards[1][0]:+.4f}")
    print(f"    Load Balance Score: {overload_rewards[1][1]['load_balance']:+.4f}")
    
    print("\n✅ All reward function tests completed!")
    print("\nKey Observations:")
    print("  • URLLC violations receive heavy penalties")
    print("  • Improvement bonuses encourage positive trends")
    print("  • Resource efficiency rewards balanced utilization")
    print("  • Load balancing prevents single-edge overload")
    print("  • SLA compliance is prioritized across all agents")
