#!/usr/bin/env python3
"""
config.py
Configuration and Hyperparameters for Multi-Agent 5G MEC RL System

Centralized configuration for:
- PPO agent hyperparameters
- A3C agent hyperparameters
- GWO optimizer settings
- Training configuration
- NS-3 simulation settings
- Logging and checkpointing

Place in: ai_controller/config.py
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json
import os


# ============================================================================
# PPO AGENT CONFIGURATION
# ============================================================================
@dataclass
class PPOConfig:
    """PPO (Proximal Policy Optimization) agent configuration"""
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128, 64])
    activation: str = "relu"  # relu, tanh, elu
    
    # PPO-specific hyperparameters
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping parameter
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    value_loss_coef: float = 0.5  # Value function loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training parameters
    n_epochs: int = 5  # Epochs per update
    batch_size: int = 64
    minibatch_size: int = 32
    buffer_size: int = 2048  # Steps before update
    
    # Exploration
    initial_exploration: float = 1.0
    final_exploration: float = 0.1
    exploration_decay_steps: int = 10000
    
    # Target network update
    target_update_interval: int = 1000
    
    # Device
    device: str = "cpu"  # "cpu" or "cuda"
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# A3C AGENT CONFIGURATION
# ============================================================================
@dataclass
class A3CConfig:
    """A3C (Asynchronous Advantage Actor-Critic) agent configuration"""
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    
    # A3C-specific hyperparameters
    learning_rate: float = 5e-5
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 40.0
    
    # Training parameters
    n_steps: int = 20  # Steps before update
    num_workers: int = 4  # Number of parallel workers
    
    # Exploration
    initial_exploration: float = 1.0
    final_exploration: float = 0.05
    exploration_decay_steps: int = 50000
    
    # Device
    device: str = "cpu"
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# GREY WOLF OPTIMIZER CONFIGURATION
# ============================================================================
@dataclass
class GWOConfig:
    """Grey Wolf Optimizer configuration - LIGHTWEIGHT MODE"""
    
    # Population parameters (smaller for faster optimization)
    n_wolves: int = 6  # ✅ Reduced from 10
    max_iterations: int = 20  # ✅ Reduced from 50
    
    # Search space bounds (backbone only now)
    cpu_bounds: tuple = (0.4, 0.9)  # Not used in lightweight mode
    memory_bounds: tuple = (0.4, 0.9)  # Not used in lightweight mode
    bandwidth_bounds: tuple = (0.3, 1.0)  # ✅ PRIMARY: Backbone bandwidth ratio
    
    # Convergence parameters
    a_init: float = 2.0
    a_final: float = 0.0
    
    # Hybrid RL-GWO parameters (not used in lightweight)
    rl_weight: float = 0.7
    gwo_weight: float = 0.3
    
    # ✅ UPDATED: More aggressive triggers
    trigger_on_sla_violations: int = 2  # Reduced from 3
    trigger_on_reward_plateau: int = 5
    
    def to_dict(self) -> Dict:
        return asdict(self)

# ============================================================================
# REWARD CONFIGURATION (imported from reward_functions.py)
# ============================================================================
@dataclass
class RewardConfig:
    """Reward function weights (same as in reward_functions.py)"""
    
    # PPO Flow-level weights
    ppo_throughput_weight: float = 0.3
    ppo_delay_weight: float = 0.35
    ppo_loss_weight: float = 0.2
    ppo_fairness_weight: float = 0.15
    
    # A3C Edge-level weights
    a3c_sla_compliance_weight: float = 0.4
    a3c_resource_efficiency_weight: float = 0.3
    a3c_load_balance_weight: float = 0.3
    
    # GWO Global weights
    gwo_throughput_weight: float = 0.25
    gwo_delay_weight: float = 0.25
    gwo_sla_weight: float = 0.3
    gwo_resource_weight: float = 0.2
    
    # Penalties
    urllc_delay_violation_penalty: float = -2.0
    embb_throughput_violation_penalty: float = -1.0
    packet_loss_penalty_multiplier: float = -0.5
    
    # ADD NEW FIELD:
    urllc_delay_target_ms: float = 70.0  # New configurable target
    embb_delay_target_ms: float = 100.0
    mmtc_delay_target_ms: float = 150.0

    # Bonuses
    perfect_sla_bonus: float = 2.0
    efficiency_bonus: float = 2.0
    
    # Scaling
    reward_scale: float = 1.0
    
    def validate(self):
        """Ensure weights are properly normalized"""
        ppo_sum = (self.ppo_throughput_weight + self.ppo_delay_weight + 
                   self.ppo_loss_weight + self.ppo_fairness_weight)
        
        a3c_sum = (self.a3c_sla_compliance_weight + 
                   self.a3c_resource_efficiency_weight + 
                   self.a3c_load_balance_weight)
        
        if abs(ppo_sum - 1.0) > 0.01:
            print(f"⚠️  Warning: PPO reward weights sum to {ppo_sum:.3f} (expected 1.0)")
        
        if abs(a3c_sum - 1.0) > 0.01:
            print(f"⚠️  Warning: A3C reward weights sum to {a3c_sum:.3f} (expected 1.0)")

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
@dataclass
class TrainingConfig:
    """Overall training configuration"""
    
    # Training phases
    phase: str = "ppo_only", "a3c_only", "combined", "gwo_hybrid"
    
    # Episode settings
    max_episodes: int = 1000
    max_steps_per_episode: int = 180 #  ⬅️ KEY CHANGE: Always 180 steps per episode
    episode_timeout: float = 600.0  # Seconds (slightly longer than sim time)
    a3c_offline_training: bool = True  # Enable A3C offline mode

    # NS-3 simulation settings
    ns3_sim_time: float = 300.0  # Simulation duration in seconds
    ns3_metrics_interval: float = 2.0  # Metric collection interval
    
    # Learning schedule
    warmup_episodes: int = 10  # Random actions for initial exploration
    checkpoint_interval: int = 50  # Save model every N episodes
    evaluation_interval: int = 10  # Evaluate without exploration every N episodes
    
    # Early stopping
    early_stopping_patience: int = 100  # Stop if no improvement for N episodes
    target_reward_ppo: float = 5.0  # Stop if this reward achieved
    target_reward_a3c: float = 4.0
    
    # Replay buffer
    replay_buffer_size: int = 100000
    min_replay_size: int = 1000  # Start training after this many transitions
    
    # Logging
    log_interval: int = 1  # Log every N episodes
    tensorboard_log: bool = True
    verbose: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # TCP algorithm switching (for dynamic comparison)
    tcp_algorithms: List[str] = field(default_factory=lambda: [
        "TcpBbr", "TcpCubic", "TcpNewReno", "TcpVegas"
    ])
    enable_tcp_switching: bool = True  # Allow RL to switch TCP algo
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# NS-3 SOCKET CONFIGURATION
# ============================================================================
@dataclass
class SocketConfig:
    """Socket communication configuration (same as socket_comm.py)"""
    
    host: str = "127.0.0.1"
    port: int = 5000
    buffer_size: int = 16384
    timeout: float = 1.0
    max_queue_size: int = 1000
    verbose: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# MASTER CONFIGURATION
# ============================================================================
@dataclass
class MasterConfig:
    """Master configuration combining all sub-configs"""
    
    ppo: PPOConfig = field(default_factory=PPOConfig)
    a3c: A3CConfig = field(default_factory=A3CConfig)
    gwo: GWOConfig = field(default_factory=GWOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    socket: SocketConfig = field(default_factory=SocketConfig)
    
    # Metadata
    version: str = "1.0.0"
    experiment_name: str = "5g_mec_rl"
    description: str = "Multi-agent RL for 5G MEC congestion control"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'ppo': self.ppo.to_dict(),
            'a3c': self.a3c.to_dict(),
            'gwo': self.gwo.to_dict(),
            'reward': self.reward.to_dict(),
            'training': self.training.to_dict(),
            'socket': self.socket.to_dict(),
            'version': self.version,
            'experiment_name': self.experiment_name,
            'description': self.description
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"✓ Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MasterConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls(
            ppo=PPOConfig(**data.get('ppo', {})),
            a3c=A3CConfig(**data.get('a3c', {})),
            gwo=GWOConfig(**data.get('gwo', {})),
            reward=RewardConfig(**data.get('reward', {})),
            training=TrainingConfig(**data.get('training', {})),
            socket=SocketConfig(**data.get('socket', {})),
            version=data.get('version', '1.0.0'),
            experiment_name=data.get('experiment_name', '5g_mec_rl'),
            description=data.get('description', '')
        )
        
        print(f"✓ Configuration loaded from {filepath}")
        return config
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print(f"CONFIGURATION: {self.experiment_name}")
        print("="*60)
        
        print(f"\nVersion: {self.version}")
        print(f"Description: {self.description}")
        print(f"Training Phase: {self.training.phase}")
        
        print("\n📊 PPO Agent:")
        print(f"  Architecture: {self.ppo.hidden_layers}")
        print(f"  Learning Rate: {self.ppo.learning_rate}")
        print(f"  Batch Size: {self.ppo.batch_size}")
        print(f"  Clip Epsilon: {self.ppo.clip_epsilon}")
        
        print("\n🖥️  A3C Agent:")
        print(f"  Architecture: {self.a3c.hidden_layers}")
        print(f"  Learning Rate: {self.a3c.learning_rate}")
        print(f"  Workers: {self.a3c.num_workers}")
        
        print("\n🐺 Grey Wolf Optimizer:")
        print(f"  Population: {self.gwo.n_wolves} wolves")
        print(f"  Max Iterations: {self.gwo.max_iterations}")
        print(f"  RL Weight: {self.gwo.rl_weight}, GWO Weight: {self.gwo.gwo_weight}")
        
        print("\n🎯 Reward Weights:")
        print(f"  PPO - Throughput: {self.reward.ppo_throughput_weight}")
        print(f"  PPO - Delay: {self.reward.ppo_delay_weight}")
        print(f"  A3C - SLA Compliance: {self.reward.a3c_sla_compliance_weight}")
        
        print("\n🏋️  Training:")
        print(f"  Max Episodes: {self.training.max_episodes}")
        print(f"  Steps/Episode: {self.training.max_steps_per_episode}")
        print(f"  NS-3 Sim Time: {self.training.ns3_sim_time}s")
        print(f"  Checkpoint Interval: {self.training.checkpoint_interval}")
        
        print("\n🔌 Socket:")
        print(f"  Host: {self.socket.host}:{self.socket.port}")
        print(f"  Buffer Size: {self.socket.buffer_size}")
        
        print("="*60 + "\n")


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================
def get_config_preset(preset_name: str) -> MasterConfig:
    """
    Get predefined configuration presets
    
    Presets:
    - "dev": Fast training for development/debugging
    - "quick": Quick training run (~1 hour)
    - "standard": Standard training run (~6 hours)
    - "full": Full training run (~24 hours)
    - "ppo_only": Train only PPO agent
    - "a3c_only": Train only A3C agent
    - "hybrid": Combined PPO + A3C + GWO
    """
    
    if preset_name == "dev":
        # Fast development preset - FIXED 30 steps per RL episode
        config = MasterConfig(
            experiment_name="dev_test"
        )
        config.training.max_episodes = 10
        config.training.max_steps_per_episode = 60  # ⬅️ FIXED: Always 60 steps
        config.training.ns3_sim_time = 300.0  # Simulation can be any length
        config.training.ns3_metrics_interval = 2.0
        config.training.episode_timeout = 120.0
        config.ppo.buffer_size = 256
        config.ppo.batch_size = 32
        config.training.checkpoint_interval = 5
        
    elif preset_name == "quick":
        # Quick training (~1 hour)
        config = MasterConfig(
            experiment_name="quick_train"
        )
        config.training.max_episodes = 100
        config.training.ns3_sim_time = 180.0
        config.training.checkpoint_interval = 20
        
    elif preset_name == "standard":
        # Standard training (default - ~6 hours)
        config = MasterConfig(
            experiment_name="standard_train"
        )
        config.training.max_episodes = 500
        config.training.ns3_sim_time = 300.0
        
    elif preset_name == "full":
        # Full training (~24 hours)
        config = MasterConfig(
            experiment_name="full_train"
        )
        config.training.max_episodes = 2000
        config.training.ns3_sim_time = 300.0
        config.training.early_stopping_patience = 200
        
    elif preset_name == "ppo_only":
        # PPO-only training
        config = MasterConfig(
            experiment_name="ppo_flow_control"
        )
        config.training.phase = "ppo_only"
        config.training.max_episodes = 500
        config.ppo.learning_rate = 3e-4
        config.ppo.buffer_size = 4096
        
    elif preset_name == "a3c_only":
    # A3C-only training with offline mode
        config = MasterConfig(
            experiment_name="a3c_edge_management"
        )
        config.training.phase = "a3c_only"
        config.training.max_episodes = 500
        config.training.max_steps_per_episode = 180  # ⬅️ 180 steps per episode
        config.training.a3c_offline_training = True  # ⬅️ Enable offline training
        config.a3c.num_workers = 4  # For offline mode, use 4 workers
        config.a3c.learning_rate = 1e-3
        
    elif preset_name == "hybrid":
        # Combined training with GWO
        config = MasterConfig(
            experiment_name="hybrid_rl_gwo"
        )
        config.training.phase = "gwo_hybrid"
        config.training.max_episodes = 1000
        config.training.max_steps_per_episode = 180  # ⬅ 180 steps per episode
        config.gwo.rl_weight = 0.7
        config.gwo.gwo_weight = 0.3
        
    else:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available: dev, quick, standard, full, ppo_only, a3c_only, hybrid")
    
    return config


# ============================================================================
# HYPERPARAMETER TUNING UTILITIES
# ============================================================================
class HyperparameterTuner:
    """
    Helper class for hyperparameter search
    
    Usage:
        tuner = HyperparameterTuner()
        best_config = tuner.random_search(n_trials=10)
    """
    
    def __init__(self, base_config: Optional[MasterConfig] = None):
        self.base_config = base_config or MasterConfig()
    
    def random_search_space(self) -> Dict:
        """Define random search space for key hyperparameters"""
        import numpy as np
        
        return {
            # PPO hyperparameters
            'ppo.learning_rate': np.random.choice([1e-4, 3e-4, 1e-3]),
            'ppo.clip_epsilon': np.random.choice([0.1, 0.2, 0.3]),
            'ppo.gamma': np.random.choice([0.95, 0.99, 0.995]),
            'ppo.entropy_coef': np.random.choice([0.0, 0.01, 0.05]),
            
            # A3C hyperparameters
            'a3c.learning_rate': np.random.choice([5e-4, 1e-3, 3e-3]),
            'a3c.num_workers': np.random.choice([2, 4, 8]),
            
            # Reward weights
            'reward.ppo_delay_weight': np.random.uniform(0.3, 0.5),
            'reward.a3c_sla_compliance_weight': np.random.uniform(0.35, 0.50)
        }
    
    def apply_hyperparameters(self, config: MasterConfig, params: Dict) -> MasterConfig:
        """Apply hyperparameters to config"""
        for key, value in params.items():
            parts = key.split('.')
            obj = config
            
            # Navigate to the right sub-config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            
            # Set the parameter
            setattr(obj, parts[-1], value)
        
        return config


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================
def validate_config(config: MasterConfig) -> List[str]:
    """
    Validate configuration for common issues
    
    Returns:
        List of warning/error messages (empty if valid)
    """
    warnings = []
    
    # Check reward weights sum to 1.0
    ppo_sum = (config.reward.ppo_throughput_weight + 
               config.reward.ppo_delay_weight + 
               config.reward.ppo_loss_weight + 
               config.reward.ppo_fairness_weight)
    
    if abs(ppo_sum - 1.0) > 0.01:
        warnings.append(f"PPO reward weights sum to {ppo_sum:.3f} (should be 1.0)")
    
    a3c_sum = (config.reward.a3c_sla_compliance_weight + 
               config.reward.a3c_resource_efficiency_weight + 
               config.reward.a3c_load_balance_weight)
    
    if abs(a3c_sum - 1.0) > 0.01:
        warnings.append(f"A3C reward weights sum to {a3c_sum:.3f} (should be 1.0)")
    
    # Check learning rates are reasonable
    if config.ppo.learning_rate > 0.01:
        warnings.append(f"PPO learning rate {config.ppo.learning_rate} is very high (typical: 1e-4 to 1e-3)")
    
    if config.a3c.learning_rate > 0.01:
        warnings.append(f"A3C learning rate {config.a3c.learning_rate} is very high")
    
    # Check buffer sizes
    if config.ppo.buffer_size < config.ppo.batch_size:
        warnings.append("PPO buffer_size should be >= batch_size")
    
    # Check episode settings
    expected_steps = config.training.ns3_sim_time / config.training.ns3_metrics_interval
    if config.training.max_steps_per_episode < expected_steps:
        warnings.append(f"max_steps_per_episode ({config.training.max_steps_per_episode}) "
                       f"< expected steps ({expected_steps:.0f})")
    
    # Check GWO weights
    gwo_weight_sum = config.gwo.rl_weight + config.gwo.gwo_weight
    if abs(gwo_weight_sum - 1.0) > 0.01:
        warnings.append(f"GWO RL+GWO weights sum to {gwo_weight_sum:.3f} (should be 1.0)")
    
    return warnings


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================
if __name__ == "__main__":
    """Test configuration system"""
    print("Testing Configuration System\n")
    
    # Test 1: Create default config
    print("="*60)
    print("TEST 1: Default Configuration")
    print("="*60)
    
    default_config = MasterConfig()
    default_config.print_summary()
    
    # Validate
    warnings = validate_config(default_config)
    if warnings:
        print("⚠️  Validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ Configuration is valid")
    
    # Test 2: Save and load
    print("\n" + "="*60)
    print("TEST 2: Save and Load")
    print("="*60)
    
    test_file = "/tmp/test_config.json"
    default_config.save(test_file)
    
    loaded_config = MasterConfig.load(test_file)
    print("✓ Configuration loaded successfully")
    
    # Verify values match
    assert loaded_config.ppo.learning_rate == default_config.ppo.learning_rate
    assert loaded_config.training.max_episodes == default_config.training.max_episodes
    print("✓ Values match original config")
    
    # Test 3: Presets
    print("\n" + "="*60)
    print("TEST 3: Configuration Presets")
    print("="*60)
    
    presets = ["dev", "quick", "standard", "ppo_only", "a3c_only", "hybrid"]
    
    for preset_name in presets:
        print(f"\n{preset_name.upper()} preset:")
        preset_config = get_config_preset(preset_name)
        print(f"  Experiment: {preset_config.experiment_name}")
        print(f"  Phase: {preset_config.training.phase}")
        print(f"  Max Episodes: {preset_config.training.max_episodes}")
        print(f"  Sim Time: {preset_config.training.ns3_sim_time}s")
    
    # Test 4: Hyperparameter tuning
    print("\n" + "="*60)
    print("TEST 4: Hyperparameter Random Search")
    print("="*60)
    
    tuner = HyperparameterTuner()
    
    print("\nGenerating 3 random hyperparameter configurations:")
    for i in range(3):
        params = tuner.random_search_space()
        print(f"\n  Trial {i+1}:")
        for key, value in params.items():
            print(f"    {key}: {value}")
    
    # Test 5: Custom configuration
    print("\n" + "="*60)
    print("TEST 5: Custom Configuration")
    print("="*60)
    
    custom_config = MasterConfig(
        experiment_name="custom_experiment"
    )
    
    # Modify PPO settings
    custom_config.ppo.learning_rate = 5e-4
    custom_config.ppo.hidden_layers = [256, 256, 128]
    
    # Modify training settings
    custom_config.training.max_episodes = 750
    custom_config.training.phase = "combined"
    
    # Modify reward weights
    custom_config.reward.ppo_delay_weight = 0.45
    custom_config.reward.ppo_throughput_weight = 0.25
    custom_config.reward.ppo_loss_weight = 0.20
    custom_config.reward.ppo_fairness_weight = 0.10
    
    print("\nCustom configuration created:")
    custom_config.print_summary()
    
    # Validate custom config
    warnings = validate_config(custom_config)
    if warnings:
        print("\n⚠️  Validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✅ Custom configuration is valid")
    
    # Test 6: Export all presets
    print("\n" + "="*60)
    print("TEST 6: Export All Presets")
    print("="*60)
    
    output_dir = "/tmp/config_presets"
    os.makedirs(output_dir, exist_ok=True)
    
    for preset_name in presets:
        config = get_config_preset(preset_name)
        filepath = os.path.join(output_dir, f"{preset_name}_config.json")
        config.save(filepath)
    
    print(f"\n✓ All presets exported to {output_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\n✅ All configuration tests passed!")
    print("\nKey Features:")
    print("  ✓ Modular dataclass-based configuration")
    print("  ✓ JSON save/load functionality")
    print("  ✓ Pre-defined presets for different scenarios")
    print("  ✓ Hyperparameter tuning support")
    print("  ✓ Configuration validation")
    print("  ✓ Easy customization and extension")
    
    print("\nUsage Examples:")
    print("  # Use preset")
    print("  config = get_config_preset('ppo_only')")
    print()
    print("  # Customize")
    print("  config.ppo.learning_rate = 1e-3")
    print("  config.training.max_episodes = 1000")
    print()
    print("  # Save")
    print("  config.save('my_config.json')")
    print()
    print("  # Load")
    print("  config = MasterConfig.load('my_config.json')")
    print()
    print("  # Validate")
    print("  warnings = validate_config(config)")
    
    print("\n" + "="*60 + "\n")
