#!/usr/bin/env python3
"""
test_all_agents.py
Comprehensive testing script for all RL agents

Tests:
- PPO agent (flow-level)
- A3C agent (edge-level)
- GWO optimizer (global)
- Integration with main.py

Place in: ai_controller/test_all_agents.py

Usage:
    python test_all_agents.py
"""

import sys
import numpy as np
from config import MasterConfig, get_config_preset, validate_config 

print("\n" + "=" * 70)
print("    COMPREHENSIVE AGENT TESTING")
print("=" * 70 + "\n")

# ============================================================================
# TEST 1: PPO AGENT
# ============================================================================
print("TEST 1: PPO Agent (Flow-Level Control)")
print("-" * 70)

try:
    from agents.ppo_agent import PPOAgent
    from rl_spaces import FlowState, FlowAction
    
    config = get_config_preset('dev')
    ppo_agent = PPOAgent(config.ppo)
    
    # Create test flow state
    test_flow = FlowState(
        flow_id=1,
        service_type='URLLC',
        protocol='TCP',
        src_port=5001,
        dst_port=9000,
        throughput_mbps=15.0,
        rtt_ms=45.0,
        jitter_ms=5.0,
        loss_rate=0.05,
        cwnd=120,
        sla_violation=False,
        throughput_trend=0.0,
        rtt_trend=0.0,
        queue_length=15,
        queue_drops=0
    )
    
    # Test action selection
    action, log_prob, value = ppo_agent.select_action(test_flow, training=True)
    
    print(f" PPO Agent operational")
    print(f"  Action: {action.name}")
    print(f"  Log probability: {log_prob:.4f}")
    print(f"  Value estimate: {value:.4f}")
    
    # Test storing transitions
    for _ in range(10):
        ppo_agent.store_transition(test_flow, action, 0.5, test_flow, False, log_prob, value)
    
    print(f" Stored {len(ppo_agent.states)} transitions")
    
    print(" PPO Agent: PASSED\n")
    
except Exception as e:
    print(f"❌ PPO Agent: FAILED")
    print(f"   Error: {e}\n")
    sys.exit(1)

# ============================================================================
# TEST 2: A3C AGENT
# ============================================================================
print("TEST 2: A3C Agent (Edge-Level Control)")
print("-" * 70)

try:
    from agents.a3c_agent import A3CAgent
    from rl_spaces import EdgeState, EdgeAction
    
    a3c_agent = A3CAgent(config.a3c)
    
    # Create test edge state
    test_edge = EdgeState(
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
    
    # Test action selection
    action, log_prob, value = a3c_agent.select_action(test_edge, training=True)
    
    print(f" A3C Agent operational")
    print(f"  Action: {action.name}")
    print(f"  Log probability: {log_prob:.4f}")
    print(f"  Value estimate: {value:.4f}")
    
    print("A3C Agent: PASSED\n")
    
except Exception as e:
    print(f"❌ A3C Agent: FAILED")
    print(f"   Error: {e}\n")
    sys.exit(1)

# ============================================================================
# TEST 3: GREY WOLF OPTIMIZER
# ============================================================================
print("TEST 3: Grey Wolf Optimizer (Global Optimization)")
print("-" * 70)

try:
    from agents.gwo_optimizer import GreyWolfOptimizer
    
    gwo = GreyWolfOptimizer(config.gwo)
    
    # Define simple fitness function
    def test_fitness(allocation):
        # Penalize deviation from 0.7 (70% utilization)
        target = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
        return np.sum((allocation - target) ** 2)
    
    # Run optimization
    best_allocation, best_fitness = gwo.optimize(
        test_fitness,
        verbose=False
    )
    
    print(f" GWO Optimizer operational")
    print(f"  Best allocation: {best_allocation}")
    print(f"  Best fitness: {best_fitness:.6f}")
    print(f"  Iterations: {gwo.config.max_iterations}")
    
    # Test hybrid optimization
    rl_solution = np.array([0.6, 0.65, 0.55, 0.6, 0.7])
    hybrid_allocation, hybrid_fitness = gwo.hybrid_optimize(
        test_fitness,
        rl_solution,
        verbose=False
    )
    
    print(f" Hybrid RL-GWO optimization works")
    print(f"  Hybrid fitness: {hybrid_fitness:.6f}")
    
    print("GWO Optimizer: PASSED\n")
    
except Exception as e:
    print(f"❌ GWO Optimizer: FAILED")
    print(f"   Error: {e}\n")
    sys.exit(1)

# ============================================================================
# TEST 4: REWARD FUNCTIONS
# ============================================================================
print("TEST 4: Reward Functions")
print("-" * 70)

try:
    from reward_functions import RewardCoordinator
    
    reward_coord = RewardCoordinator(config.reward)
    
    # Test PPO rewards
    flow_states = [test_flow]
    ppo_rewards = reward_coord.compute_ppo_rewards(flow_states)
    
    print(f" PPO rewards computed")
    print(f"  Reward: {ppo_rewards[0][0]:.4f}")
    
    # Test A3C rewards
    edge_states = [test_edge]
    a3c_rewards = reward_coord.compute_a3c_rewards(edge_states)
    
    print(f" A3C rewards computed")
    print(f"  Reward: {a3c_rewards[0][0]:.4f}")
    
    # Test GWO reward
    from rl_spaces import GWOState
    test_gwo_state = GWOState(
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
    
    gwo_reward, gwo_components = reward_coord.compute_gwo_reward(test_gwo_state)
    
    print(f" GWO reward computed")
    print(f"  Reward: {gwo_reward:.4f}")
    
    print(" Reward Functions: PASSED\n")
    
except Exception as e:
    print(f"❌ Reward Functions: FAILED")
    print(f"   Error: {e}\n")
    sys.exit(1)

# ============================================================================
# TEST 5: STATE TRACKER
# ============================================================================
print("TEST 5: State Tracker")
print("-" * 70)

try:
    from state_tracker import StateTracker
    
    tracker = StateTracker(history_size=10)
    tracker.start_episode(episode_id=1)
    
    # Simulate some states
    for _ in range(5):
        flows = [{
            'id': 1,
            'service_type': 'URLLC',
            'protocol': 'TCP',
            'src_port': 5001,
            'dst_port': 9000,
            'throughput_mbps': 15.0,
            'rtt_ms': 45.0,
            'jitter_ms': 5.0,
            'loss_rate': 0.05,
            'cwnd': 120,
            'src': '10.0.0.1',
            'dst': '20.0.0.1'
        }]
        
        queues = []
        edges = [{
            'location': 'india',
            'cpu_usage': 0.65,
            'memory_usage': 0.70,
            'network_usage': 0.60
        }]
        
        tracker.update(flows, queues, edges, timestamp=float(_))
    
    # Get states
    flow_states = tracker.get_flow_states()
    edge_states = tracker.get_edge_states()
    global_state = tracker.get_global_state()
    
    print(f" State tracker operational")
    print(f"  Flow states: {len(flow_states)}")
    print(f"  Edge states: {len(edge_states)}")
    print(f"  Global state: throughput={global_state.total_throughput:.1f} Mbps")
    
    tracker.end_episode()
    
    print(" State Tracker: PASSED\n")
    
except Exception as e:
    print(f"❌ State Tracker: FAILED")
    print(f"   Error: {e}\n")
    sys.exit(1)

# ============================================================================
# TEST 6: CONFIGURATION SYSTEM
# ============================================================================
print("TEST 6: Configuration System")
print("-" * 70)

try:
    # Test presets
    presets = ['dev', 'ppo_only', 'a3c_only', 'hybrid']
    
    for preset in presets:
        test_config = get_config_preset(preset)
        warnings = validate_config(test_config)
        
        if warnings:
            print(f"  {preset}: {len(warnings)} warnings")
        else:
            print(f"  {preset}: valid")
    
    # Test save/load
    test_config = get_config_preset('dev')
    test_config.save('/tmp/test_config.json')
    
    loaded_config = MasterConfig.load('/tmp/test_config.json')
    
    print(f" Config save/load works")
    
    print("Configuration System: PASSED\n")
    
except Exception as e:
    print(f"❌ Configuration System: FAILED")
    print(f"   Error: {e}\n")
    sys.exit(1)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 70)
print("    ALL TESTS PASSED! ")
print("=" * 70)

print("\n System Status:")
print("  PPO Agent (Flow-level)")
print("  A3C Agent (Edge-level)")
print("  GWO Optimizer (Global)")
print("  Reward Functions")
print("  State Tracker")
print("  Configuration System")

print("\nReady for NS-3 Integration!")
print("\nNext Steps:")
print("  1. Start NS-3 simulation:")
print("     cd ~/Desktop/ns-allinone-3.45/ns-3.45")
print("     ./ns3 run 'scratch/mec_full_simulation --enableRL=true'")
print()
print("  2. In another terminal, start AI controller:")
print("     cd ~/Desktop/ns-allinone-3.45/ai_controller")
print("     python main.py --preset hybrid")
print()
print("=" * 70 + "\n")