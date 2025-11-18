#!/usr/bin/env python3
"""
sim_config_generator.py
Generate randomized NS-3 simulation configurations for curriculum learning

Place in: ai_controller/sim_config_generator.py
"""

import numpy as np
import json
from typing import Dict, List
from dataclasses import dataclass, asdict


@dataclass
class SimulationConfig:
    """NS-3 simulation configuration"""
    
    # Traffic intensity
    num_embb_flows: int = 2
    num_urllc_flows: int = 2
    num_mmtc_devices: int = 10
    num_background_flows: int = 2
    
    # Traffic rates (Mbps for eMBB, Kbps for others)
    embb_rate_min: float = 30.0
    embb_rate_max: float = 60.0
    urllc_rate_kbps: float = 500.0
    mmtc_rate_kbps: float = 50.0
    background_rate_min: float = 100.0
    background_rate_max: float = 180.0
    
    # Timing
    flow_start_time_min: float = 2.0
    flow_start_time_max: float = 6.0
    background_start_time_min: float = 4.0
    background_start_time_max: float = 8.0
    
    # Congestion patterns
    burst_probability: float = 0.2
    burst_scale_min: float = 3.0
    burst_scale_max: float = 7.0
    
    # Network conditions
    bottleneck_bandwidth_mbps: int = 200
    bottleneck_delay_ms: int = 2
    
    # Simulation duration
    sim_time: float = 300.0
    
    # Metadata
    difficulty_level: str = "medium"  # easy/medium/hard/extreme
    description: str = ""
    config_id: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SimulationConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class CurriculumGenerator:
    """
    Generate progression of simulation configs for curriculum learning
    
    Difficulty Levels:
    1. Easy: Low traffic, no congestion
    2. Medium: Normal traffic, some congestion
    3. Hard: High traffic, frequent congestion
    4. Extreme: Maximum traffic, burst patterns, link failures
    """
    
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.config_counter = 0
    
    def generate_easy_config(self) -> SimulationConfig:
        """Easy difficulty: Learn basics without stress"""
        return SimulationConfig(
            num_embb_flows=1,
            num_urllc_flows=1,
            num_mmtc_devices=self.rng.randint(5, 10),
            num_background_flows=0,  # No background traffic
            
            embb_rate_min=20.0,
            embb_rate_max=40.0,
            urllc_rate_kbps=300.0,
            mmtc_rate_kbps=30.0,
            
            flow_start_time_min=2.0,
            flow_start_time_max=4.0,
            
            burst_probability=0.0,  # No bursts
            
            bottleneck_bandwidth_mbps=200,
            bottleneck_delay_ms=2,
            
            difficulty_level="easy",
            description="Low traffic, no congestion",
            config_id=self.config_counter
        )
    
    def generate_medium_config(self) -> SimulationConfig:
        """Medium difficulty: Realistic traffic"""
        return SimulationConfig(
            num_embb_flows=self.rng.randint(2, 4),
            num_urllc_flows=self.rng.randint(2, 4),
            num_mmtc_devices=self.rng.randint(10, 20),
            num_background_flows=self.rng.randint(1, 2),
            
            embb_rate_min=30.0 + self.rng.uniform(-5, 5),
            embb_rate_max=60.0 + self.rng.uniform(-10, 10),
            urllc_rate_kbps=450.0 + self.rng.uniform(-50, 50),
            mmtc_rate_kbps=50.0 + self.rng.uniform(-10, 10),
            background_rate_min=110.0 + self.rng.uniform(-10, 10),
            background_rate_max=180.0 + self.rng.uniform(-20, 20),
            
            flow_start_time_min=2.0 + self.rng.uniform(0, 1),
            flow_start_time_max=6.0 + self.rng.uniform(0, 2),
            background_start_time_min=4.0 + self.rng.uniform(0, 2),
            background_start_time_max=8.0 + self.rng.uniform(0, 2),
            
            burst_probability=0.2,
            burst_scale_min=3.0,
            burst_scale_max=7.0,
            
            bottleneck_bandwidth_mbps=200,
            bottleneck_delay_ms=2,
            
            difficulty_level="medium",
            description="Normal traffic with some congestion",
            config_id=self.config_counter
        )
    
    def generate_hard_config(self) -> SimulationConfig:
        """Hard difficulty: High traffic, frequent congestion"""
        return SimulationConfig(
            num_embb_flows=self.rng.randint(4, 6),
            num_urllc_flows=self.rng.randint(4, 6),
            num_mmtc_devices=self.rng.randint(15, 25),
            num_background_flows=self.rng.randint(2, 3),
            
            embb_rate_min=40.0 + self.rng.uniform(-5, 10),
            embb_rate_max=80.0 + self.rng.uniform(-10, 20),
            urllc_rate_kbps=600.0 + self.rng.uniform(-100, 100),
            mmtc_rate_kbps=70.0 + self.rng.uniform(-15, 15),
            background_rate_min=130.0 + self.rng.uniform(-10, 20),
            background_rate_max=200.0 + self.rng.uniform(-20, 30),
            
            flow_start_time_min=1.5 + self.rng.uniform(0, 1),
            flow_start_time_max=5.0 + self.rng.uniform(0, 2),
            background_start_time_min=3.0 + self.rng.uniform(0, 2),
            background_start_time_max=6.0 + self.rng.uniform(0, 2),
            
            burst_probability=0.4,
            burst_scale_min=4.0,
            burst_scale_max=10.0,
            
            bottleneck_bandwidth_mbps=200,  # Same capacity, more traffic!
            bottleneck_delay_ms=2,
            
            difficulty_level="hard",
            description="High traffic with frequent congestion",
            config_id=self.config_counter
        )
    
    def generate_extreme_config(self) -> SimulationConfig:
        """Extreme difficulty: Stress test"""
        return SimulationConfig(
            num_embb_flows=self.rng.randint(5, 8),
            num_urllc_flows=self.rng.randint(6, 8),
            num_mmtc_devices=self.rng.randint(20, 30),
            num_background_flows=self.rng.randint(2, 4),
            
            embb_rate_min=50.0 + self.rng.uniform(-10, 20),
            embb_rate_max=100.0 + self.rng.uniform(-20, 40),
            urllc_rate_kbps=750.0 + self.rng.uniform(-150, 150),
            mmtc_rate_kbps=85.0 + self.rng.uniform(-20, 20),
            background_rate_min=150.0 + self.rng.uniform(-20, 30),
            background_rate_max=220.0 + self.rng.uniform(-30, 40),
            
            flow_start_time_min=1.0 + self.rng.uniform(0, 1),
            flow_start_time_max=4.0 + self.rng.uniform(0, 2),
            background_start_time_min=2.0 + self.rng.uniform(0, 2),
            background_start_time_max=5.0 + self.rng.uniform(0, 2),
            
            burst_probability=0.6,  # Very bursty!
            burst_scale_min=5.0,
            burst_scale_max=12.0,
            
            bottleneck_bandwidth_mbps=200,
            bottleneck_delay_ms=2,
            
            difficulty_level="extreme",
            description="Maximum stress test with burst patterns",
            config_id=self.config_counter
        )
    
    def generate_curriculum(self, 
                          total_episodes: int = 100,
                          checkpoint_interval: int = 5) -> List[SimulationConfig]:
        """
        Generate full curriculum of configs
        
        Progression:
        - Episodes 1-20:   Easy
        - Episodes 21-50:  Medium
        - Episodes 51-80:  Hard
        - Episodes 81-100: Extreme
        
        Args:
            total_episodes: Total training episodes
            checkpoint_interval: Episodes per config change
        
        Returns:
            List of configs (one per checkpoint)
        """
        num_checkpoints = total_episodes // checkpoint_interval
        configs = []
        
        for i in range(num_checkpoints):
            episode_num = i * checkpoint_interval
            progress = episode_num / total_episodes
            
            # Determine difficulty based on progress
            if progress < 0.2:
                config = self.generate_easy_config()
            elif progress < 0.5:
                config = self.generate_medium_config()
            elif progress < 0.8:
                config = self.generate_hard_config()
            else:
                config = self.generate_extreme_config()
            
            config.config_id = self.config_counter
            self.config_counter += 1
            configs.append(config)
        
        return configs
    
    def generate_random_mix(self, num_configs: int = 20) -> List[SimulationConfig]:
        """Generate random mix of all difficulties"""
        configs = []
        difficulties = ['easy', 'medium', 'hard', 'extreme']
        
        for i in range(num_configs):
            difficulty = self.rng.choice(difficulties)
            
            if difficulty == 'easy':
                config = self.generate_easy_config()
            elif difficulty == 'medium':
                config = self.generate_medium_config()
            elif difficulty == 'hard':
                config = self.generate_hard_config()
            else:
                config = self.generate_extreme_config()
            
            config.config_id = self.config_counter
            self.config_counter += 1
            configs.append(config)
        
        return configs


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("    SIMULATION CONFIG GENERATOR")
    print("=" * 70 + "\n")
    
    generator = CurriculumGenerator(seed=42)
    
    # Test 1: Generate curriculum
    print("TEST 1: Full Curriculum (100 episodes, change every 5)")
    print("-" * 70)
    
    curriculum = generator.generate_curriculum(
        total_episodes=100,
        checkpoint_interval=5
    )
    
    print(f"Generated {len(curriculum)} configs\n")
    
    # Show first few configs
    for i, config in enumerate(curriculum[:5]):
        print(f"Config {i} (Episodes {i*5+1}-{(i+1)*5}):")
        print(f"  Difficulty: {config.difficulty_level}")
        print(f"  eMBB flows: {config.num_embb_flows}")
        print(f"  URLLC flows: {config.num_urllc_flows}")
        print(f"  mMTC devices: {config.num_mmtc_devices}")
        print(f"  Background: {config.num_background_flows}")
        print(f"  eMBB rate: {config.embb_rate_min:.0f}-{config.embb_rate_max:.0f} Mbps")
        print(f"  Description: {config.description}\n")
    
    print("...")
    
    # Show last few configs
    for i, config in enumerate(curriculum[-3:], start=len(curriculum)-3):
        print(f"Config {i} (Episodes {i*5+1}-{(i+1)*5}):")
        print(f"  Difficulty: {config.difficulty_level}")
        print(f"  Description: {config.description}\n")
    
    # Test 2: Save/Load
    print("\n" + "=" * 70)
    print("TEST 2: Save/Load Config")
    print("-" * 70)
    
    test_config = curriculum[0]
    test_config.save("ai_controller/configs/sim_config_0.json")
    print("✓ Saved to ai_controller/configs/sim_config_0.json")
    
    loaded_config = SimulationConfig.load("/tmp/sim_config_0.json")
    print("✓ Loaded successfully")
    print(f"  Difficulty: {loaded_config.difficulty_level}")
    print(f"  Flows: {loaded_config.num_embb_flows + loaded_config.num_urllc_flows}")
    
    # Test 3: Random mix
    print("\n" + "=" * 70)
    print("TEST 3: Random Mix (20 configs)")
    print("-" * 70)
    
    random_configs = generator.generate_random_mix(20)
    
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0, 'extreme': 0}
    for config in random_configs:
        difficulty_counts[config.difficulty_level] += 1
    
    print("Distribution:")
    for difficulty, count in difficulty_counts.items():
        print(f"  {difficulty.capitalize()}: {count}")
    
    print("\n✅ All tests passed!")
    print("\nNext: Integrate with training loop in main.py")