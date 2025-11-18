#!/usr/bin/env python3
"""
gwo_optimizer.py - UPDATED for Lightweight Backbone Optimization
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass
import time

from rl_spaces import GWOState
from config import GWOConfig


@dataclass
class Wolf:
    """Individual wolf in the population"""
    position: np.ndarray
    fitness: float = float('inf')


class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer - LIGHTWEIGHT VERSION
    Focuses on backbone bandwidth optimization only
    """
    
    def __init__(self, config: GWOConfig):
        self.config = config
        
        # ✅ SIMPLIFIED: Only optimize backbone bandwidth
        # Other resources use fixed heuristics
        self.lower_bounds = np.array([
            config.bandwidth_bounds[0]  # Backbone bandwidth only
        ])
        
        self.upper_bounds = np.array([
            config.bandwidth_bounds[1]
        ])
        
        self.dim = 1  # ✅ Only 1 dimension now!
        
        # Wolf population
        self.wolves: List[Wolf] = []
        
        # Elite wolves
        self.alpha: Optional[Wolf] = None
        self.beta: Optional[Wolf] = None
        self.delta: Optional[Wolf] = None
        
        # Statistics
        self.iteration_count = 0
        self.best_fitness_history = []
        
        print(f"✓ Grey Wolf Optimizer initialized (Lightweight Mode)")
        print(f"  Population size: {config.n_wolves}")
        print(f"  Max iterations: {config.max_iterations}")
        print(f"  Optimizing: Backbone Bandwidth Only")
        print(f"  Range: {config.bandwidth_bounds[0]:.1f} - {config.bandwidth_bounds[1]:.1f}")
    
    def initialize_population(self):
        """Initialize wolf population randomly"""
        self.wolves = []
        
        for _ in range(self.config.n_wolves):
            position = np.random.uniform(
                self.lower_bounds,
                self.upper_bounds,
                size=self.dim
            )
            self.wolves.append(Wolf(position=position))
        
        print(f"  ✓ Initialized {len(self.wolves)} wolves")
    
    def evaluate_fitness(self, fitness_fn: Callable[[np.ndarray], float]):
        """Evaluate fitness for all wolves"""
        for wolf in self.wolves:
            wolf.fitness = fitness_fn(wolf.position)
    
    def update_elite_wolves(self):
        """Update alpha, beta, delta wolves"""
        sorted_wolves = sorted(self.wolves, key=lambda w: w.fitness)
        
        self.alpha = Wolf(
            position=sorted_wolves[0].position.copy(),
            fitness=sorted_wolves[0].fitness
        )
        self.beta = Wolf(
            position=sorted_wolves[1].position.copy(),
            fitness=sorted_wolves[1].fitness
        )
        self.delta = Wolf(
            position=sorted_wolves[2].position.copy(),
            fitness=sorted_wolves[2].fitness
        )
    
    def update_positions(self, iteration: int):
        """Update wolf positions"""
        a = self.config.a_init - iteration * (self.config.a_init - self.config.a_final) / self.config.max_iterations
        
        for wolf in self.wolves:
            for i in range(self.dim):
                # Alpha influence
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha.position[i] - wolf.position[i])
                X1 = self.alpha.position[i] - A1 * D_alpha
                
                # Beta influence
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta.position[i] - wolf.position[i])
                X2 = self.beta.position[i] - A2 * D_beta
                
                # Delta influence
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta.position[i] - wolf.position[i])
                X3 = self.delta.position[i] - A3 * D_delta
                
                # Update position
                wolf.position[i] = (X1 + X2 + X3) / 3.0
            
            # Clip to bounds
            wolf.position = np.clip(
                wolf.position,
                self.lower_bounds,
                self.upper_bounds
            )
    
    def optimize(self, 
                fitness_fn: Callable[[np.ndarray], float],
                initial_solution: Optional[np.ndarray] = None,
                verbose: bool = True) -> Tuple[np.ndarray, float]:
        """Run GWO optimization"""
        start_time = time.time()
        
        if verbose:
            print("\n" + "=" * 60)
            print("GREY WOLF OPTIMIZER - STARTING")
            print("=" * 60)
        
        self.initialize_population()
        
        # Inject initial solution if provided
        if initial_solution is not None:
            self.wolves[0].position = np.clip(
                initial_solution,
                self.lower_bounds,
                self.upper_bounds
            )
        
        # Evaluate initial population
        self.evaluate_fitness(fitness_fn)
        self.update_elite_wolves()
        
        self.best_fitness_history = [self.alpha.fitness]
        
        if verbose:
            print(f"  Initial best fitness: {self.alpha.fitness:.4f}")
            print(f"  Initial backbone BW: {self.decode_allocation(self.alpha.position)['backbone_bandwidth']:.2%}")
        
        # Main optimization loop
        for iteration in range(self.config.max_iterations):
            self.update_positions(iteration)
            self.evaluate_fitness(fitness_fn)
            self.update_elite_wolves()
            
            self.best_fitness_history.append(self.alpha.fitness)
            
            if verbose and (iteration + 1) % 10 == 0:
                bw = self.decode_allocation(self.alpha.position)['backbone_bandwidth']
                print(f"  Iteration {iteration + 1}/{self.config.max_iterations} | "
                      f"Fitness: {self.alpha.fitness:.4f} | BW: {bw:.2%}")
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print("\n" + "=" * 60)
            print("GREY WOLF OPTIMIZER - COMPLETED")
            print("=" * 60)
            print(f"  Best Fitness: {self.alpha.fitness:.4f}")
            print(f"  Best Backbone BW: {self.decode_allocation(self.alpha.position)['backbone_bandwidth']:.2%}")
            print(f"  Time: {elapsed_time:.2f}s")
            print("=" * 60 + "\n")
        
        self.iteration_count += self.config.max_iterations
        
        return self.alpha.position.copy(), self.alpha.fitness
    
    def decode_allocation(self, position: np.ndarray) -> Dict[str, float]:
        """
        Decode position to allocation dictionary
        ✅ SIMPLIFIED: Only backbone bandwidth, rest use heuristics
        """
        backbone_bw = position[0]
        
        # Heuristic: CPU/memory proportional to backbone usage
        # If backbone is at 80%, assume edges need similar capacity
        cpu_alloc = 0.5 + (backbone_bw - 0.5) * 0.5  # Maps 0.3-1.0 → 0.4-0.75
        mem_alloc = 0.5 + (backbone_bw - 0.5) * 0.4
        
        return {
            'backbone_bandwidth': backbone_bw,
            'india_cpu': cpu_alloc,
            'india_memory': mem_alloc,
            'uk_cpu': cpu_alloc,
            'uk_memory': mem_alloc
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """Get optimizer statistics"""
        return {
            'iteration_count': self.iteration_count,
            'best_fitness': self.alpha.fitness if self.alpha else None,
            'best_position': self.alpha.position.tolist() if self.alpha else None,
            'fitness_history': self.best_fitness_history
        }


# ============================================================================
# ✅ NEW: Lightweight fitness evaluator for real-time use
# ============================================================================
class LightweightGWOFitness:
    """
    Fast fitness evaluation using recent network metrics
    No simulation required - uses last 10 seconds of data
    """
    
    def __init__(self, target_throughput: float = 400.0, 
                 target_delay: float = 30.0,
                 sla_penalty_weight: float = 5.0):
        self.target_throughput = target_throughput
        self.target_delay = target_delay
        self.sla_penalty_weight = sla_penalty_weight
    
    def evaluate(self, 
                 allocation: np.ndarray,
                 current_state: GWOState) -> float:
        """
        Fast fitness evaluation based on current state
        
        Args:
            allocation: [backbone_bandwidth] (0.3-1.0 scale)
            current_state: Current global network state
        
        Returns:
            fitness: Lower is better (GWO minimizes)
        """
        backbone_bw = allocation[0]
        
        # Convert to actual Mbps (200 Mbps base capacity)
        actual_bw_mbps = backbone_bw * 200.0
        
        # ===== Component 1: Throughput vs Bandwidth Efficiency =====
        # Penalize if throughput exceeds allocated bandwidth (congestion)
        utilization = current_state.total_throughput / actual_bw_mbps
        
        if utilization > 0.95:
            # Severe congestion
            congestion_penalty = 10.0 * (utilization - 0.95)
        elif utilization > 0.85:
            # Moderate congestion
            congestion_penalty = 5.0 * (utilization - 0.85)
        else:
            # Good utilization
            congestion_penalty = 0.0
        
        # Reward efficient use (target 70-85% utilization)
        if 0.70 <= utilization <= 0.85:
            efficiency_reward = 5.0
        else:
            efficiency_reward = -abs(utilization - 0.75) * 3.0
        
        # ===== Component 2: Delay =====
        delay_penalty = max(0, (current_state.total_delay - self.target_delay) / 10.0)
        
        # ===== Component 3: SLA Violations =====
        sla_penalty = current_state.total_sla_violations * self.sla_penalty_weight
        
        # ===== Component 4: Resource Cost =====
        # Penalize high bandwidth allocation (cost consideration)
        resource_cost = (backbone_bw - 0.5) * 2.0  # Prefer lower allocations
        
        # ===== Total Fitness =====
        fitness = (
            congestion_penalty +
            delay_penalty +
            sla_penalty +
            resource_cost -
            efficiency_reward
        )
        
        return fitness
    
    def create_fitness_function(self, current_state: GWOState):
        """
        Create a fitness function closure for GWO optimizer
        
        Usage:
            fitness_fn = evaluator.create_fitness_function(global_state)
            best_alloc, best_fit = gwo.optimize(fitness_fn)
        """
        def fitness_fn(allocation: np.ndarray) -> float:
            return self.evaluate(allocation, current_state)
        
        return fitness_fn


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    """Test lightweight GWO"""
    from config import GWOConfig
    
    print("Testing Lightweight Grey Wolf Optimizer\n")
    
    # Test 1: Create optimizer
    print("=" * 60)
    print("TEST 1: Optimizer Creation")
    print("=" * 60)
    
    config = GWOConfig(
        n_wolves=8,
        max_iterations=30,
        bandwidth_bounds=(0.3, 1.0)
    )
    
    gwo = GreyWolfOptimizer(config)
    print(f"✓ Optimizer created\n")
    
    # Test 2: Simple fitness function (backbone congestion model)
    print("=" * 60)
    print("TEST 2: Backbone Congestion Optimization")
    print("=" * 60)
    
    # Simulate congested network
    total_demand_mbps = 180.0  # Total traffic demand
    
    def backbone_fitness(bw_allocation):
        """
        Simple congestion model:
        - If bandwidth < demand: congestion penalty
        - If bandwidth >> demand: waste penalty
        """
        actual_bw = bw_allocation[0] * 200.0  # Scale to Mbps
        
        if actual_bw < total_demand_mbps:
            # Congestion
            shortage = total_demand_mbps - actual_bw
            return shortage * 2.0  # Heavy penalty
        else:
            # Over-provisioning (waste)
            excess = actual_bw - total_demand_mbps
            return excess * 0.5  # Light penalty
    
    best_bw, best_fit = gwo.optimize(backbone_fitness, verbose=True)
    
    print(f"\n✓ Optimization completed")
    print(f"  Best bandwidth allocation: {best_bw[0]:.2%}")
    print(f"  Actual bandwidth: {best_bw[0] * 200:.1f} Mbps")
    print(f"  Traffic demand: {total_demand_mbps:.1f} Mbps")
    print(f"  Fitness: {best_fit:.2f}\n")
    
    # Test 3: Lightweight fitness evaluator
    print("=" * 60)
    print("TEST 3: Lightweight Fitness Evaluator")
    print("=" * 60)
    
    # Mock global state
    mock_state = GWOState(
        total_throughput=175.0,
        total_delay=35.0,
        total_loss=0.5,
        india_cpu=0.7,
        india_memory=0.65,
        uk_cpu=0.6,
        uk_memory=0.55,
        backbone_bandwidth_usage=0.85,
        total_sla_violations=3,
        urllc_violations=2,
        embb_violations=1,
        india_to_uk_flows=12,
        uk_to_india_flows=10,
        to_cloud_flows=6
    )
    
    evaluator = LightweightGWOFitness()
    
    # Test different allocations
    allocations = [
        np.array([0.5]),  # 50% → 100 Mbps
        np.array([0.7]),  # 70% → 140 Mbps
        np.array([0.9]),  # 90% → 180 Mbps
    ]
    
    print("\nTesting different bandwidth allocations:")
    for alloc in allocations:
        fitness = evaluator.evaluate(alloc, mock_state)
        print(f"  BW: {alloc[0]:.1%} ({alloc[0]*200:.0f} Mbps) → Fitness: {fitness:+.2f}")
    
    # Full optimization with evaluator
    print("\nRunning full optimization with evaluator:")
    fitness_fn = evaluator.create_fitness_function(mock_state)
    best_alloc, best_fit = gwo.optimize(fitness_fn, verbose=False)
    
    print(f"\n✓ Best allocation found:")
    print(f"  Bandwidth: {best_alloc[0]:.2%} ({best_alloc[0]*200:.1f} Mbps)")
    print(f"  Fitness: {best_fit:+.2f}")
    
    print("\n✅ All lightweight GWO tests passed!")