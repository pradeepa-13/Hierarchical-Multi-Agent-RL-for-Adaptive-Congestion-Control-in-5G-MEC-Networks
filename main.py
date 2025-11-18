#!/usr/bin/env python3
"""
main.py - FIXED VERSION
Critical Fix: A3C checkpoints now save BEFORE config changes

Changes:
1. Save checkpoint immediately after episode completes (not just at intervals)
2. Proper episode counter management
3. Clear logging for checkpoint saves
"""

import argparse
import sys
import time
import signal
from typing import Optional, List, Dict
from dataclasses import asdict
import traceback
import logging
import numpy as np
from datetime import datetime  # ✅ NEW: For timestamps

# Import our modules
from config import MasterConfig, get_config_preset, validate_config
from socket_comm import SocketCommunicator, SocketConfig, NetworkState, RLAction
from state_tracker import StateTracker
from reward_functions import RewardCoordinator
from rl_spaces import FlowState, EdgeState, GWOState, FlowAction, EdgeAction
from agents.ppo_agent import PPOAgent
from agents.a3c_agent import A3CAgent
from agents.gwo_optimizer import GreyWolfOptimizer
from sim_config_generator import CurriculumGenerator, SimulationConfig
from agents.gwo_optimizer import LightweightGWOFitness

class MEC_RL_Controller:
    """Main controller - FIXED VERSION with proper A3C checkpointing"""
    
    def __init__(self, config: MasterConfig, resume_ppo: str = None, resume_a3c: str = None):
        from action_verifier import ActionVerifier
        self.action_verifier = ActionVerifier(verification_window=2)

        # Logger setup
        self.logger = logging.getLogger("MEC_RL_Controller")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
                                        "%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.config = config

        # ✅ NEW: Session tracking for checkpoint management
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_resumed_session = (resume_ppo is not None or resume_a3c is not None)
        self.resumed_from_episode = 0  # Track where we resumed from
        
        print("\n" + "="*60)
        print("Initializing 5G MEC RL Controller")
        print("="*60)
        
        # ✅ NEW: Print session info
        if self.is_resumed_session:
            print(f"\n🔄 RESUMING TRAINING SESSION")
            print(f"   Session ID: {self.session_id}")
            print(f"   Checkpoints will be saved with format:")
            print(f"     episode_{{N}}_{self.session_id}_{{agent}}.pth")
        else:
            print(f"\n🆕 NEW TRAINING SESSION")
            print(f"   Session ID: {self.session_id}")
        
        # Socket communicator
        print("\n1. Setting up socket communication...")
        socket_config = SocketConfig(
            host=config.socket.host,
            port=config.socket.port,
            buffer_size=config.socket.buffer_size,
            timeout=config.socket.timeout,
            verbose=config.socket.verbose
        )
        self.comm = SocketCommunicator(socket_config)
        
        # State tracker
        print("2. Initializing state tracker...")
        self.tracker = StateTracker(
            history_size=20,
            max_transitions=config.training.replay_buffer_size
        )
        
        # Reward coordinator
        print("3. Setting up reward functions...")
        self.reward_coordinator = RewardCoordinator(config.reward)
        
        # RL Agents
        print("4. Initializing RL agents...")

        # PPO Agent (Flow-level)
        if config.training.phase in ['ppo_only', 'combined', 'gwo_hybrid']:
            print("   ✓ Initializing PPO agent...")
            self.ppo_agent = PPOAgent(config.ppo)

            if resume_ppo:
                print(f"   📦 Loading PPO checkpoint: {resume_ppo}")
                self.ppo_agent.load(resume_ppo)
                self.resumed_from_episode = max(self.resumed_from_episode, 
                                                self.ppo_agent.episode_count)
                print(f"   ✅ PPO resumed from episode {self.ppo_agent.episode_count}")
                print(f"      Update count: {self.ppo_agent.update_count}")
                print(f"      Total steps: {self.ppo_agent.total_steps}")
                print(f"      Exploration rate: {self.ppo_agent.exploration_rate:.4f}")
        else:
            print("   ⚠️  PPO agent disabled for this phase")
            self.ppo_agent = None

        # A3C Agent (Edge-level)
        if config.training.phase in ['a3c_only', 'combined', 'gwo_hybrid']:
            print("   ✓ Initializing A3C agent...")
            self.a3c_agent = A3CAgent(config.a3c)

            if resume_a3c:
                print(f"   📦 Loading A3C checkpoint: {resume_a3c}")
                self.a3c_agent.load(resume_a3c)
                self.resumed_from_episode = max(self.resumed_from_episode, 
                                                self.a3c_agent.episode_count)
                print(f"   ✅ A3C resumed from episode {self.a3c_agent.episode_count}")
                print(f"      Update count: {self.a3c_agent.update_count}")
                print(f"      Total steps: {self.a3c_agent.total_steps}")
        else:
            print("   ⚠️  A3C agent disabled for this phase")
            self.a3c_agent = None

        # GWO Optimizer (Global resource allocation)
        if config.training.phase == 'gwo_hybrid':
            print("   ✓ Initializing GWO optimizer...")
            self.gwo_optimizer = GreyWolfOptimizer(config.gwo)
            self.gwo_trigger_count = 0
            self.last_gwo_activation = 0.0
        else:
            print("   ⚠️  GWO optimizer disabled for this phase")
            self.gwo_optimizer = None
        
        # ✅ FIXED: Episode management
        # If resuming, start from the checkpoint episode + 1
        if self.is_resumed_session:
            self.current_episode = self.resumed_from_episode
            self.expected_episode = self.resumed_from_episode + 1
            self.episodes_completed = self.resumed_from_episode
            print(f"\n✅ Resume point established:")
            print(f"   Last completed episode: {self.episodes_completed}")
            print(f"   Next episode will be: {self.expected_episode}")
            
            # ✅ NEW: Load and verify curriculum state if available
            saved_curriculum_state = self._load_curriculum_state_early()
            if saved_curriculum_state:
                print(f"\n📋 Previous curriculum state found:")
                print(f"   Config index: {saved_curriculum_state['current_config_idx']}")
                print(f"   Episodes in config: {saved_curriculum_state['episodes_in_checkpoint']}")
                print(f"   Last saved: {saved_curriculum_state['timestamp']}")
                print(f"   ✅ Curriculum will auto-resume from this point")
        else:
            self.current_episode = 0
            self.expected_episode = 1
            self.episodes_completed = 0
        
        # Episode state
        self.running = False
        self.step_count = 0
        self.training_mode = True
        self.waiting_for_ns3 = False
        
        # Statistics
        self.total_states_received = 0
        self.total_actions_sent = 0
        
        print("="*60)
        print("✅ Initialization Complete")
        print("="*60 + "\n")

    def _train_a3c_offline(self, episode: int):
        """Train A3C agent in offline mode after episode completion"""
        if self.a3c_agent is None or not self.config.training.a3c_offline_training:
            return
        
        self.logger.info("\n🔄 Training A3C agent (Offline Mode)...")
        
        try:
            self.a3c_agent.enable_offline_mode()
            
            a3c_transitions = self.tracker.get_a3c_transitions(episode)
            
            if not a3c_transitions:
                self.logger.info("   No A3C transitions available for training")
                return
            
            self.logger.info(f"   Training with {len(a3c_transitions)} A3C transitions")
            
            train_stats = self.a3c_agent.train_offline(a3c_transitions)
            
            if train_stats:
                self.logger.info(f"   A3C Training Results:")
                self.logger.info(f"     Policy Loss: {train_stats.get('policy_loss', 0):.4f}")
                self.logger.info(f"     Value Loss: {train_stats.get('value_loss', 0):.4f}")
                self.logger.info(f"     Entropy: {train_stats.get('entropy', 0):.4f}")
                self.logger.info(f"     Transitions Used: {train_stats.get('transitions_used', 0)}")
            
        except Exception as e:
            self.logger.error(f"❌ A3C offline training failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.a3c_agent.disable_offline_mode()

    def _flow_at_location(self, flow: FlowState, location: str) -> bool:
        """Determine if flow belongs to edge location"""
        if location == 'india':
            return flow.flow_id % 2 == 1
        elif location == 'uk':
            return flow.flow_id % 2 == 0
        
        return False

    def run_curriculum_training(self, 
                           total_episodes: int = 100,
                           checkpoint_interval: int = 5,
                           ns3_command_template: str = None):
        """
        Run curriculum training with proper checkpoint management
        
        ✅ RESUME SUPPORT: Automatically calculates which curriculum config to use
        based on the episode number we're resuming from.
        """
        import os
        os.makedirs("checkpoints", exist_ok=True)
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("    CURRICULUM TRAINING MODE")
        
        # ✅ CRITICAL: Calculate curriculum position when resuming
        if self.is_resumed_session:
            # Calculate which config we should be on based on episode number
            self.current_config_idx = self.resumed_from_episode // checkpoint_interval
            self.episodes_in_checkpoint = self.resumed_from_episode % checkpoint_interval
            
            self.logger.info(f"    🔄 RESUMING from episode {self.resumed_from_episode}")
            self.logger.info(f"    Session ID: {self.session_id}")
            self.logger.info(f"    Curriculum position: Config {self.current_config_idx}, "
                           f"Episode {self.episodes_in_checkpoint}/{checkpoint_interval}")
        else:
            self.current_config_idx = 0
            self.episodes_in_checkpoint = 0
            self.logger.info(f"    🆕 NEW SESSION")
            self.logger.info(f"    Session ID: {self.session_id}")
        
        self.logger.info("=" * 70)
        self.logger.info(f"  Total Episodes: {total_episodes}")
        self.logger.info(f"  Config Change Interval: {checkpoint_interval}")
        self.logger.info("=" * 70 + "\n")
        
        # Generate curriculum
        generator = CurriculumGenerator(seed=42)
        curriculum = generator.generate_curriculum(
            total_episodes=total_episodes,
            checkpoint_interval=checkpoint_interval
        )
        
        self.logger.info(f"✓ Generated {len(curriculum)} simulation configs")
        self.logger.info(f"  Progression: Easy → Medium → Hard → Extreme\n")
        
        # Store curriculum
        self.curriculum = curriculum
        
        # ✅ NEW: Save curriculum state to checkpoint for verification
        self._save_curriculum_state()
        
        # Start socket
        if not self.start():
            self.logger.error("❌ Failed to start socket")
            return False
        
        # ✅ MAIN TRAINING LOOP
        while self.episodes_completed < total_episodes:
            # Check if we need to change config
            if self.episodes_in_checkpoint == 0:
                self._load_next_curriculum_config()
            
            # Wait for NS-3 to be ready
            self._wait_for_ns3_ready()
            
            # Process one episode (blocks until episode completes)
            self._run_one_episode()
            
            # ✅ CRITICAL: Save checkpoint IMMEDIATELY after episode
            self._save_episode_checkpoint()
            
            # ✅ NEW: Save curriculum state after each episode
            self._save_curriculum_state()
            
            # Update counters
            self.episodes_completed += 1
            self.episodes_in_checkpoint += 1
            
            # Check if checkpoint group complete
            if self.episodes_in_checkpoint >= checkpoint_interval:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Checkpoint Group {self.current_config_idx + 1} Complete!")
                self.logger.info(f"{'='*60}\n")
                self.episodes_in_checkpoint = 0
                self.current_config_idx += 1
        
        self.logger.info("\n✅ All episodes completed!")
        self.stop()
        return True
    
    def _save_curriculum_state(self):
        """
        ✅ NEW: Save curriculum state to a separate file
        
        This allows perfect resumption - we know exactly which config
        we were on and how many episodes into that config we were.
        """
        import json
        import os
        
        state_file = "checkpoints/curriculum_state.json"
        
        state = {
            'session_id': self.session_id,
            'episodes_completed': self.episodes_completed,
            'current_config_idx': self.current_config_idx,
            'episodes_in_checkpoint': self.episodes_in_checkpoint,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            os.makedirs("checkpoints", exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save curriculum state: {e}")

    def _load_curriculum_state_early(self):
        """
        ✅ NEW: Load curriculum state during initialization (before curriculum starts)
        
        This is called early to show the user what state will be resumed.
        """
        import json
        import os
        
        state_file = "checkpoints/curriculum_state.json"
        
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                return state
            return None
        except Exception as e:
            self.logger.warning(f"⚠️  Could not load curriculum state: {e}")
            return None
    
    def _load_curriculum_state(self):
        """
        ✅ NEW: Load curriculum state from file (for verification)
        
        Returns the saved state if it exists, None otherwise.
        """
        import json
        import os
        
        state_file = "checkpoints/curriculum_state.json"
        
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                return state
            return None
        except Exception as e:
            self.logger.warning(f"⚠️  Could not load curriculum state: {e}")
            return None

    def _save_episode_checkpoint(self):
        """
        ✅ FIXED: Save checkpoint with timestamped name to prevent overwrites
        
        Naming convention:
        - New session: episode_{N}_{session_id}_{agent}.pth
        - Resumed session: episode_{N}_{session_id}_resumed_{agent}.pth
        
        This ensures:
        1. No overwrites between different training runs
        2. Clear identification of resumed sessions
        3. Easy to find the latest checkpoint
        """
        # Construct checkpoint name with session ID
        if self.is_resumed_session:
            checkpoint_name = f"episode_{self.episodes_completed}_{self.session_id}_resumed"
        else:
            checkpoint_name = f"episode_{self.episodes_completed}_{self.session_id}"
        
        self.logger.info(f"\n💾 Saving checkpoint: {checkpoint_name}")
        
        saved_any = False
        
        # Save PPO if active
        if self.ppo_agent:
            try:
                ppo_path = f"checkpoints/{checkpoint_name}_ppo.pth"
                self.ppo_agent.save(ppo_path)
                self.logger.info(f"   ✅ PPO saved to {ppo_path}")
                self.logger.info(f"      Episode: {self.ppo_agent.episode_count}")
                self.logger.info(f"      Updates: {self.ppo_agent.update_count}")
                self.logger.info(f"      Total steps: {self.ppo_agent.total_steps}")
                saved_any = True
            except Exception as e:
                self.logger.error(f"   ❌ PPO save failed: {e}")
        
        # Save A3C if active
        if self.a3c_agent:
            try:
                a3c_path = f"checkpoints/{checkpoint_name}_a3c.pth"
                self.a3c_agent.save(a3c_path)
                self.logger.info(f"   ✅ A3C saved to {a3c_path}")
                self.logger.info(f"      Episode: {self.a3c_agent.episode_count}")
                self.logger.info(f"      Updates: {self.a3c_agent.update_count}")
                self.logger.info(f"      Total steps: {self.a3c_agent.total_steps}")
                saved_any = True
            except Exception as e:
                self.logger.error(f"   ❌ A3C save failed: {e}")
        
        if not saved_any:
            self.logger.warning(f"   ⚠️  No agents to save!")
        else:
            self.logger.info(f"✅ Checkpoint {checkpoint_name} saved\n")
            
            # ✅ NEW: Save a "latest" symlink for easy resumption
            self._create_latest_symlink(checkpoint_name)

    def _create_latest_symlink(self, checkpoint_name: str):
        """
        Create a 'latest' symlink for easy resumption
        
        This creates:
        - checkpoints/latest_ppo.pth -> episode_N_sessionid_ppo.pth
        - checkpoints/latest_a3c.pth -> episode_N_sessionid_a3c.pth
        """
        import os
        
        try:
            # PPO symlink
            if self.ppo_agent:
                ppo_target = f"{checkpoint_name}_ppo.pth"
                ppo_link = "checkpoints/latest_ppo.pth"
                
                # Remove existing symlink if it exists
                if os.path.islink(ppo_link):
                    os.unlink(ppo_link)
                elif os.path.exists(ppo_link):
                    os.remove(ppo_link)
                
                # Create new symlink
                os.symlink(ppo_target, ppo_link)
                self.logger.debug(f"   📎 Created symlink: latest_ppo.pth -> {ppo_target}")
            
            # A3C symlink
            if self.a3c_agent:
                a3c_target = f"{checkpoint_name}_a3c.pth"
                a3c_link = "checkpoints/latest_a3c.pth"
                
                if os.path.islink(a3c_link):
                    os.unlink(a3c_link)
                elif os.path.exists(a3c_link):
                    os.remove(a3c_link)
                
                os.symlink(a3c_target, a3c_link)
                self.logger.debug(f"   📎 Created symlink: latest_a3c.pth -> {a3c_target}")
                
        except Exception as e:
            self.logger.warning(f"   ⚠️  Could not create latest symlink: {e}")

    def _load_next_curriculum_config(self):
        """Load next curriculum config and save for NS-3"""
        if self.current_config_idx >= len(self.curriculum):
            self.logger.info("✅ Curriculum complete!")
            return
        
        sim_config = self.curriculum[self.current_config_idx]
        
        checkpoint_start = self.current_config_idx * self.config.training.checkpoint_interval + 1
        checkpoint_end = (self.current_config_idx + 1) * self.config.training.checkpoint_interval
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"CHECKPOINT {self.current_config_idx + 1}/{len(self.curriculum)}")
        self.logger.info("=" * 70)
        self.logger.info(f"  Episodes: {checkpoint_start}-{checkpoint_end}")
        self.logger.info(f"  Difficulty: {sim_config.difficulty_level}")
        self.logger.info(f"  Description: {sim_config.description}")
        self.logger.info(f"  eMBB: {sim_config.num_embb_flows}, "
                        f"URLLC: {sim_config.num_urllc_flows}, "
                        f"mMTC: {sim_config.num_mmtc_devices}, "
                        f"BG: {sim_config.num_background_flows}")
        self.logger.info("=" * 70 + "\n")
        
        # Save config for NS-3
        config_file = f"sim_config_{self.current_config_idx}.json"
        sim_config.save(config_file)
        self.logger.info(f"✓ Saved config to {config_file}\n")

    def _wait_for_ns3_ready(self):
        """Wait for NS-3 to be ready (manual restart in offline mode)"""
        if self.episodes_completed == 0:
            self.logger.info("📋 Waiting for NS-3 to start...")
        else:
            self.logger.info("\n" + "🔴" * 35)
            self.logger.info("    NS-3 RESTART REQUIRED")
            self.logger.info("🔴" * 35)
            self.logger.info(f"\n  Current Episode: {self.episodes_completed}/{self.config.training.max_episodes}")
            self.logger.info(f"  Checkpoint: {self.current_config_idx + 1}/{len(self.curriculum)}")
            
            if self.current_config_idx < len(self.curriculum):
                sim_config = self.curriculum[self.current_config_idx]
                config_file = f"sim_config_{self.current_config_idx}.json"
                
                self.logger.info(f"\n  📄 Config: {config_file}")
                self.logger.info(f"     Difficulty: {sim_config.difficulty_level}")
            
            self.logger.info("\n  ⚠️  ACTION REQUIRED:")
            self.logger.info("     1. Stop current NS-3 (if running)")
            self.logger.info("     2. Restart with new config:")
            self.logger.info(f"        ./ns3 run 'scratch/mec_full_simulation --enableRL=true --config={config_file}'")
            self.logger.info("\n     3. Wait for NS-3 handshake")
            self.logger.info("     4. Press [Enter] to continue...")
            self.logger.info("\n" + "🔴" * 35 + "\n")
            
            input(">>> Press [Enter] when NS-3 is ready: ")
        
        # Reset state for new NS-3 instance
        self.waiting_for_ns3 = True
        self.step_count = 0
        
        self.logger.info("✅ Waiting for NS-3 handshake...\n")

    def _run_one_episode(self):
        """Run one complete episode (blocks until episode ends)"""
        # Start new episode in tracker
        self.expected_episode = self.episodes_completed + 1
        self.current_episode = self.expected_episode
        self.tracker.start_episode(self.current_episode)
        self.step_count = 0
        
        self.logger.info(f"🚀 Episode {self.current_episode} started")
        if self.a3c_agent:
            self.a3c_agent.episode_count = self.current_episode
        
        # Wait for NS-3 to send episode_end signal
        episode_active = True
        
        while episode_active and self.running:
            time.sleep(0.1)
            
            # Check if episode ended (set by _handle_episode_end_signal)
            if not self.tracker.current_episode:
                episode_active = False
        
        self.logger.info(f"✅ Episode {self.current_episode} completed\n")

    def _handle_episode_start(self, data: Dict):
        """Handle episode start signal from NS-3 (unused in offline mode)"""
        pass
    
    def _handle_episode_end_signal(self, data: Dict):
        """Handle episode end signal from NS-3"""
        sim_time = data.get('simTime', 0.0)
        reason = data.get('reason', 'unknown')
        
        self.logger.info(f"⏱️  NS-3 simulation ended at t={sim_time:.1f}s ({reason})")
        
        # End the episode (don't increment counter - that's done in run_curriculum_training)
        self._end_current_episode()

    def start(self):
        """Start the RL controller"""
        print("Starting RL Controller...\n")
        
        if not self.comm.start():
            print("❌ Failed to start socket communication")
            return False
        
        # Register callbacks
        self.comm.register_state_callback(self._process_state)
        self.comm.register_handshake_callback(self._handle_handshake)
        self.comm.register_episode_callbacks(
            start_callback=self._handle_episode_start,
            end_callback=self._handle_episode_end_signal
        )
        
        self.running = True
        print("✅ RL Controller started and waiting for NS-3...\n")
        
        return True
        
    def stop(self):
        """Stop the RL controller"""
        print("\n\nStopping RL Controller...")
        
        self.running = False
        
        if self.tracker.current_episode:
            self.tracker.end_episode()
        
        self.comm.stop()
        self._print_final_statistics()
        
        print("\n✅ RL Controller stopped\n")
        self.action_verifier.print_summary()
    
    def _handle_handshake(self, data: dict):
        """Handle initial handshake from NS-3"""
        self.logger.info("🤝 Handshake received from NS-3")
        self.logger.info(f"   NS-3 Version: {data.get('version', 'unknown')}")
        
        # Mark NS-3 as ready
        self.waiting_for_ns3 = False

    def _process_state(self, state: NetworkState) -> Optional[RLAction]:
        """Main state processing callback"""
        # Skip zero-time states
        if state.timestamp == 0.0:
            self.logger.debug("Skipping zero-time state")
            action = RLAction()
            action.flow_actions = [{'type': 'no_op', 'reason': 'zero_time_state'}]
            return action

        # Log state receipt
        self.logger.info(f"📨 [Step {self.step_count}] State received: "
                        f"t={state.timestamp:.1f}s, flows={len(state.flows)}, queues={len(state.queues)}")
        
        action = RLAction()
        
        # Validate state
        if state is None:
            self.logger.warning("Received None state")
            action.flow_actions = [{'type': 'no_op', 'reason': 'null_state'}]
            return action
        
        self.total_states_received += 1
        self.step_count += 1
        
        # Check if episode should end (60 steps)
        if self.step_count >= self.config.training.max_steps_per_episode:
            self.logger.info(f"🎯 RL Episode {self.current_episode} completed after {self.step_count} steps")
            self._end_current_episode()
            
            # Start next RL episode (simulation continues)
            if self.current_episode < self.config.training.max_episodes:
                self.current_episode += 1
                self.tracker.start_episode(self.current_episode)
                self.step_count = 0
                self.logger.info(f"🔄 Starting RL Episode {self.current_episode} (Simulation continues)")
            else:
                self.logger.info("✅ All RL episodes completed!")
                self.running = False
            
            # Return neutral action for step transition
            action = RLAction()
            action.flow_actions = [{'type': 'episode_transition', 'reason': 'rl_episode_end'}]
            return action

        # Update tracker
        try:
            self.tracker.update(
                flows=state.flows,
                queues=state.queues,
                edges=state.edges or [],
                timestamp=state.timestamp
            )
        except Exception as e:
            self.logger.error(f"Failed to update tracker: {e}")
            action.flow_actions = [{'type': 'error', 'reason': 'tracker_update_failed'}]
            return action
        
        # Get states
        try:
            flow_states = self.tracker.get_flow_states()
            edge_states = self.tracker.get_edge_states()
            global_state = self.tracker.get_global_state()
        except Exception as e:
            self.logger.error(f"Failed to get states: {e}")
            action.flow_actions = [{'type': 'error', 'reason': 'state_retrieval_failed'}]
            return action
        
        # Handle empty flows (early episode)
        if not flow_states:
            self.logger.info(f"⏳ No flows yet at t={state.timestamp:.1f}s")
            action.flow_actions = [{'type': 'waiting', 'reason': 'no_flows_yet'}]
            return action
        
        # Compute rewards
        try:
            ppo_rewards = self.reward_coordinator.compute_ppo_rewards(flow_states)
            a3c_rewards = self.reward_coordinator.compute_a3c_rewards(edge_states)
            gwo_reward, gwo_components = self.reward_coordinator.compute_gwo_reward(global_state)
        except Exception as e:
            self.logger.error(f"Failed to compute rewards: {e}")
            ppo_rewards = [(0.0, {}) for _ in flow_states]
            a3c_rewards = [(0.0, {}) for _ in edge_states]
            gwo_reward = (0.0, {})
        
        # Record previous actions/rewards
        self._record_actions_and_rewards(flow_states, edge_states, ppo_rewards, a3c_rewards)
        
        # Action verification
        for flow_state in flow_states:
            verification_results = self.action_verifier.verify_impact(
                flow_id=flow_state.flow_id,
                state_after={
                    'throughput_mbps': flow_state.throughput_mbps,
                    'rtt_ms': flow_state.rtt_ms,
                    'loss_rate': flow_state.loss_rate
                }
            )
            
            if verification_results:
                for act_id, impact in verification_results.items():
                    if not impact['success']:
                        self.logger.warning(f"⚠️  Action {impact['action']} on Flow {flow_state.flow_id}: {impact['details']}")
        
        # Generate new actions
        action = self._generate_actions(state)
        
        # Safety check
        if not action.flow_actions and not action.edge_actions:
            action.flow_actions = [{'type': 'no_change', 'reason': 'all_stable'}]
        
        # Increment counters
        if action.flow_actions or action.edge_actions:
            self.total_actions_sent += 1
        
        # Periodic logging
        if self.step_count % 10 == 0:
            self._log_progress(flow_states, edge_states, global_state)
        
        # Log action
        self.logger.info(f"📤 [Step {self.step_count}] Sending: "
                        f"{len(action.flow_actions)} flow actions, "
                        f"{len(action.edge_actions)} edge actions")
        
        return action
    
    def _generate_actions(self, state: NetworkState) -> Optional[RLAction]:
        """
        Generate actions - COMPREHENSIVE VERSION with A3C and GWO
        
        Phases:
        - ppo_only: Only PPO flow actions
        - a3c_only: Only A3C edge actions
        - combined: Both PPO and A3C
        - gwo_hybrid: PPO + A3C + GWO (when needed)
        """
        action = RLAction()
        action.flow_actions = []
        action.edge_actions = []

        try:
            # ========== PHASE 1: PPO FLOW-LEVEL ACTIONS ==========
            if self.ppo_agent is not None:
                flow_states = self.tracker.get_flow_states()
                
                # Filter TCP flows
                tcp_flows = [fs for fs in flow_states if fs.protocol == 'TCP']
                
                self.logger.debug(f"Processing {len(tcp_flows)} TCP flows")
                
                for flow_state in tcp_flows:
                    try:
                        # Get PPO action
                        rl_action, log_prob, value = self.ppo_agent.select_action(
                            flow_state,
                            training=self.training_mode
                        )
                        
                        # Store transition
                        self.ppo_agent.store_transition(
                            state=flow_state,
                            action=rl_action,
                            reward=0.0,  # Placeholder
                            next_state=flow_state,
                            done=False,
                            log_prob=log_prob,
                            value=value
                        )
                        
                        # Create action for NS-3
                        action.flow_actions.append({
                            'flow_id': flow_state.flow_id,
                            'action': rl_action.name,
                            'reason': 'ppo_decision',
                            'service_type': flow_state.service_type
                        })
                        
                        # Record action in verifier
                        state_before = {
                            'throughput_mbps': flow_state.throughput_mbps,
                            'rtt_ms': flow_state.rtt_ms,
                            'loss_rate': flow_state.loss_rate
                        }
                        self.action_verifier.record_action(
                            flow_id=flow_state.flow_id,
                            action=rl_action.name,
                            state_before=state_before
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error processing flow {flow_state.flow_id}: {e}")
                        continue
            
            # ========== PHASE 2: A3C EDGE-LEVEL ACTIONS ==========
            if self.a3c_agent is not None:
                edge_states = self.tracker.get_edge_states()
                flow_states = self.tracker.get_flow_states()
                
                self.logger.debug(f"Processing {len(edge_states)} edge servers")
                
                for edge_state in edge_states:
                    try:
                        # Get A3C action
                        edge_action, log_prob, value = self.a3c_agent.select_action(
                            edge_state,
                            training=self.training_mode
                        )
                        
                        # Get flows at this edge location
                        location_flows = [f for f in flow_states if self._flow_at_location(f, edge_state.location)]
                        
                        # ✅ UPDATED ACTION TRANSLATION TO MATCH C++ EDGE ACTIONS
                        if edge_action == EdgeAction.ADJUST_QUEUE_SIZE:
                            # Direct mapping to C++ ADJUST_QUEUE_SIZE
                            action.edge_actions.append({
                                'location': edge_state.location,
                                'action': 'ADJUST_QUEUE_SIZE',
                                'reason': 'a3c_queue_adjustment',
                                'queue_size': 150  # Example: increase queue size to 150 packets
                            })
                            self.logger.info(f"📊 A3C ADJUST_QUEUE_SIZE: {edge_state.location} → 150 packets")
                            
                        elif edge_action == EdgeAction.SET_URLLC_RATE_LIMIT:
                            # Direct mapping to C++ SET_URLLC_RATE_LIMIT
                            action.edge_actions.append({
                                'location': edge_state.location,
                                'action': 'SET_URLLC_RATE_LIMIT',
                                'reason': 'a3c_urllc_rate_control',
                                'rate_limit_mbps': 100.0  # Example: limit URLLC to 100 Mbps
                            })
                            self.logger.info(f"🚦 A3C SET_URLLC_RATE_LIMIT: {edge_state.location} → 100 Mbps")
                            
                        elif edge_action == EdgeAction.ADJUST_PRIORITY_WEIGHTS:
                            # Since we're using CoDel, adjust queue size instead
                            action.edge_actions.append({
                                'location': edge_state.location,
                                'action': 'ADJUST_QUEUE_SIZE',
                                'reason': 'a3c_priority_adjustment_coDel',
                                'queue_size': 150
                            })
                            self.logger.info(f"⚖️ A3C ADJUST_PRIORITY_WEIGHTS (CoDel): {edge_state.location} → queue_size=150")
                            
                        elif edge_action == EdgeAction.ENABLE_ADMISSION_CONTROL:
                            # Direct mapping to C++ ENABLE_ADMISSION_CONTROL
                            action.edge_actions.append({
                                'location': edge_state.location,
                                'action': 'ENABLE_ADMISSION_CONTROL',
                                'reason': 'a3c_admission_control',
                                'enable': True
                            })
                            self.logger.info(f"🛑 A3C ENABLE_ADMISSION_CONTROL: {edge_state.location} → Enabled")
                            
                        elif edge_action == EdgeAction.TUNE_AQM_PARAMETERS:
                            # Direct mapping to C++ TUNE_AQM_PARAMETERS
                            action.edge_actions.append({
                                'location': edge_state.location,
                                'action': 'TUNE_AQM_PARAMETERS',
                                'reason': 'a3c_aqm_tuning',
                                'target_delay_ms': 10.0,  # Reduce target delay for URLLC sensitivity
                                'interval_ms': 100.0      # CoDel interval
                            })
                            self.logger.info(f"🎯 A3C TUNE_AQM_PARAMETERS: {edge_state.location} → target=10ms, interval=100ms")
                            
                        elif edge_action == EdgeAction.NO_CHANGE:
                            # Direct mapping to C++ NO_CHANGE
                            action.edge_actions.append({
                                'location': edge_state.location,
                                'action': 'NO_CHANGE',
                                'reason': 'a3c_no_action_needed'
                            })
                            self.logger.info(f"➡️ A3C NO_CHANGE: {edge_state.location} → No action needed")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing edge {edge_state.location}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # ========== PHASE 3: GWO GLOBAL OPTIMIZATION (HYBRID MODE ONLY) ==========
            if self.gwo_optimizer is not None and self._should_trigger_gwo():
                self.logger.info("🐺 GWO Optimizer Triggered (Lightweight Mode)!")
                
                try:
                    global_state = self.tracker.get_global_state()
                    
                    evaluator = LightweightGWOFitness(
                        target_throughput=400.0,
                        target_delay=30.0,
                        sla_penalty_weight=5.0
                    )
                    
                    # Create fitness function
                    fitness_fn = evaluator.create_fitness_function(global_state)
                    
                    # Get current backbone usage as starting point
                    current_bw = np.array([global_state.backbone_bandwidth_usage])
                    
                    # Run optimization (fast - only 20 iterations)
                    self.gwo_optimizer.config.max_iterations = 20  # Quick optimization
                    best_allocation, best_fitness = self.gwo_optimizer.optimize(
                        fitness_fn=fitness_fn,
                        initial_solution=current_bw,
                        verbose=False  # Silent for real-time
                    )
                    
                    # Decode to full allocation (with heuristics)
                    allocation_dict = self.gwo_optimizer.decode_allocation(best_allocation)
                    
                    # ✅ Send GWO action to NS-3
                    action.gwo_actions = {
                        'allocation': allocation_dict,
                        'reason': 'gwo_backbone_optimization',
                        'fitness': -best_fitness  # Convert back to reward
                    }
                    
                    self.logger.info(f"🐺 GWO Result:")
                    self.logger.info(f"   Backbone BW: {allocation_dict['backbone_bandwidth']:.2%}")
                    self.logger.info(f"   Fitness Improvement: {-best_fitness:+.2f}")
                    
                    self.gwo_trigger_count += 1
                    self.last_gwo_activation = state.timestamp
                    
                except Exception as e:
                    self.logger.error(f"GWO optimization failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Safety check
            if not action.flow_actions and not action.edge_actions and not hasattr(action, 'gwo_actions'):
                action.flow_actions = [{'type': 'no_change', 'reason': 'all_stable'}]
            
            return action

        except Exception as e:
            self.logger.error(f"Action generation error: {e}")
            import traceback
            traceback.print_exc()
            
            action = RLAction()
            action.flow_actions = [{'type': 'error', 'reason': str(e)}]
            return action


    def _should_trigger_gwo(self) -> bool:
        """
        Check if GWO should be triggered
        ✅ UPDATED: More aggressive triggering for lightweight GWO
        """
        if self.gwo_optimizer is None:
            return False
        
        global_state = self.tracker.get_global_state()
        
        # ✅ Trigger 1: SLA violations exceed threshold
        if global_state.total_sla_violations >= self.config.gwo.trigger_on_sla_violations:
            self.logger.debug(f"GWO trigger: SLA violations ({global_state.total_sla_violations})")
            return True
        
        # ✅ Trigger 2: Backbone congestion detected
        # If backbone usage > 85% and delay is high, optimize bandwidth
        if global_state.backbone_bandwidth_usage > 0.85 and global_state.total_delay > 40.0:
            self.logger.debug(f"GWO trigger: Backbone congestion (usage={global_state.backbone_bandwidth_usage:.1%}, delay={global_state.total_delay:.1f}ms)")
            return True
        
        # ✅ Trigger 3: Periodic optimization (every 30 steps)
        if self.step_count > 0 and self.step_count % 30 == 0:
            self.logger.debug(f"GWO trigger: Periodic optimization (step {self.step_count})")
            return True
        
        # ✅ Minimum time between activations (5 seconds - shorter for lightweight)
        if hasattr(self, 'last_gwo_activation'):
            import time
            time_since_last = time.time() - self.last_gwo_activation
            if time_since_last < 5.0:
                return False
        
        return False


    def _simulate_allocation(self, current_state: GWOState, allocation: np.ndarray) -> GWOState:
        """
        Simulate applying resource allocation to current state
        
        Returns modified GWOState for fitness evaluation
        """
        # Create copy of current state
        simulated_state = GWOState(
            total_throughput=current_state.total_throughput,
            total_delay=current_state.total_delay,
            total_loss=current_state.total_loss,
            india_cpu=allocation[0],       # ✅ Apply new allocation
            india_memory=allocation[1],
            uk_cpu=allocation[2],
            uk_memory=allocation[3],
            backbone_bandwidth_usage=allocation[4],
            total_sla_violations=current_state.total_sla_violations,
            urllc_violations=current_state.urllc_violations,
            embb_violations=current_state.embb_violations,
            india_to_uk_flows=current_state.india_to_uk_flows,
            uk_to_india_flows=current_state.uk_to_india_flows,
            to_cloud_flows=current_state.to_cloud_flows
        )
        
        # Simple heuristic: better allocation reduces violations
        allocation_quality = 1.0 - np.mean(np.abs(allocation - 0.7))  # Optimal ~70%
        simulated_state.total_sla_violations = int(
            current_state.total_sla_violations * (1.0 - allocation_quality * 0.3)
        )
        
        return simulated_state
    
    def _record_actions_and_rewards(self, flow_states, edge_states, ppo_rewards, a3c_rewards):
        """
        Record actions and rewards - FULLY FIXED (v3)
        
        Uses PPO agent's new update_reward() method for precise flow_id matching
        """
        
        # ✅ Record PPO flow-level actions
        if self.ppo_agent is not None and flow_states:
            for i, flow_state in enumerate(flow_states):
                if i < len(ppo_rewards):
                    reward, reward_components = ppo_rewards[i]
                    
                    # ✅ CRITICAL FIX: Update reward for this specific flow's last transition
                    success = self.ppo_agent.update_reward(flow_state.flow_id, reward)
                    
                    if not success:
                        # Flow not in buffer yet (first time seeing it)
                        pass
                    
                    # Record in tracker for statistics
                    self.tracker.record_ppo_action(
                        flow_state.flow_id,
                        FlowAction.NO_CHANGE,  # Placeholder
                        reward
                    )
                    
                    # Log reward breakdown every 20 steps
                    if self.step_count % 20 == 0 and i == 0:
                        self.logger.debug(f"Flow {flow_state.flow_id} reward breakdown:")
                        for comp_name, comp_value in reward_components.items():
                            if comp_name != 'total':
                                self.logger.debug(f"  {comp_name}: {comp_value:+.3f}")
        
        # Record A3C edge-level actions
        for i, edge_state in enumerate(edge_states):
            if i < len(a3c_rewards):
                reward, _ = a3c_rewards[i]
                self.tracker.record_a3c_action(
                    edge_state.location,
                    EdgeAction.NO_CHANGE,
                    reward
                )

    
    # def _should_end_episode(self) -> bool:
    #     """Check if current episode should end (safety timeout)"""
    #     if self.step_count >= self.config.training.max_steps_per_episode:
    #         self.logger.warning(f"Max steps ({self.config.training.max_steps_per_episode}) reached!")
    #         return True
        
    #     if self.tracker.current_episode:
    #         duration = self.tracker.current_episode.duration()
    #         if duration >= self.config.training.episode_timeout:
    #             self.logger.warning(f"Episode timeout ({self.config.training.episode_timeout}s) exceeded!")
    #             return True
        
    #     return False

    
    def _end_current_episode(self):
        """End current episode and trigger training updates"""
        if self.tracker.current_episode is None:
            self.logger.warning("Tried to end episode, but no episode is active!")
            return
        
        current_episode_num = self.current_episode
        
        # Mark all transitions as terminal
        self.tracker.mark_episode_end_transitions()
        
        # Get episode statistics
        episode_stats = self.tracker.end_episode()
        
        # Train PPO agent
        if self.ppo_agent is not None:
            self.logger.info("\n🔄 Training PPO agent...")
            
            try:
                train_stats = self.ppo_agent.update()
                
                if train_stats:
                    self.logger.info(f"   Policy Loss: {train_stats['policy_loss']:.4f}")
                    self.logger.info(f"   Value Loss: {train_stats['value_loss']:.4f}")
                    self.logger.info(f"   Entropy: {train_stats['entropy']:.4f}")
                    self.logger.info(f"   KL Divergence: {train_stats['kl_div']:.6f}")
                    self.logger.info(f"   Clip Fraction: {train_stats['clip_fraction']:.2f}")
                
                # Increment episode count
                self.ppo_agent.episode_count += 1
                
                # Save checkpoint periodically
                if current_episode_num % self.config.training.checkpoint_interval == 0:
                    checkpoint_path = f"{self.config.training.checkpoint_dir}/ppo_episode_{current_episode_num}.pth"
                    self.ppo_agent.save(checkpoint_path)
                    self.logger.info(f"✓ PPO Checkpoint saved: {checkpoint_path}")
            
            except Exception as e:
                self.logger.error(f"❌ PPO training failed: {e}")
                import traceback
                traceback.print_exc()
        
        # ✅ FIX: Train A3C in offline mode
        self._train_a3c_offline(current_episode_num)
        
        # Record GWO statistics (if used)
        if self.gwo_optimizer is not None:
            self.logger.info(f"\n🐺 GWO Statistics:")
            self.logger.info(f"   Activations: {self.gwo_trigger_count}")
            
            if self.gwo_trigger_count > 0:
                gwo_stats = self.gwo_optimizer.get_statistics()
                self.logger.info(f"   Best Fitness: {gwo_stats['best_fitness']:.4f}")

        # # Train A3C agent (if enabled)
        # if self.a3c_agent is not None:
        #     self.logger.info("\n🔄 Training A3C agent...")
            
        #     # Note: A3C typically trains asynchronously in separate workers
        #     # For now, we'll just log that it's ready for training
        #     # Full A3C training requires multi-threading (see a3c_worker.py)
            
        #     self.logger.info("   ⚠️  A3C training requires async workers (not implemented in real-time mode)")
        #     self.logger.info("   💡 For full A3C training, use offline training mode")
            
        #     # Increment episode count
        #     self.a3c_agent.episode_count += 1

        # Record GWO statistics (if used)
        
    
    def _log_progress(self, flow_states, edge_states, global_state):
        """Log current progress (called every 10 steps)"""
        self.logger.info(f"\n[Step {self.step_count}] Episode {self.current_episode}")
        self.logger.info(f"  Flows: {len(flow_states)}, Edges: {len(edge_states)}")
        self.logger.info(f"  Global Throughput: {global_state.total_throughput:.1f} Mbps")
        self.logger.info(f"  Global Delay: {global_state.total_delay:.1f} ms")
        self.logger.info(f"  SLA Violations: {global_state.total_sla_violations}")
        
        # Show per-service breakdown
        service_counts = {'eMBB': 0, 'URLLC': 0, 'mMTC': 0, 'Background': 0}
        for fs in flow_states:
            if fs.service_type in service_counts:
                service_counts[fs.service_type] += 1
        
        self.logger.info(f"  Service Mix: eMBB={service_counts['eMBB']}, "
                        f"URLLC={service_counts['URLLC']}, "
                        f"mMTC={service_counts['mMTC']}, "
                        f"BG={service_counts['Background']}")
        
        # Show URLLC health
        urllc_flows = [fs for fs in flow_states if fs.service_type == 'URLLC']
        if urllc_flows:
            urllc_delays = [fs.rtt_ms for fs in urllc_flows]
            avg_urllc_delay = sum(urllc_delays) / len(urllc_delays)
            urllc_violations = sum(1 for d in urllc_delays if d > 70.0)
            
            self.logger.info(f"  URLLC Health: Avg Delay={avg_urllc_delay:.2f}ms, "
                            f"Violations={urllc_violations}/{len(urllc_flows)}")
            
            # ✅ ADD THESE 3 LINES:
            self.logger.info(f"  URLLC Individual Delays (ms): {[f'{d:.1f}' for d in urllc_delays]}")
            urllc_losses = [fs.loss_rate for fs in urllc_flows]
            self.logger.info(f"  URLLC Individual Losses (%): {[f'{l:.2f}' for l in urllc_losses]}")
            
            if urllc_violations > len(urllc_flows) * 0.5:
                self.logger.warning("  ⚠️  >50% URLLC flows violating SLA!")

        embb_flows = [fs for fs in flow_states if fs.service_type == 'eMBB']
        if embb_flows:
            embb_throughputs = [fs.throughput_mbps for fs in embb_flows]
            avg_embb_throughput = sum(embb_throughputs) / len(embb_throughputs)
            # eMBB SLA: Throughput must be > 5 Mbps
            embb_violations = sum(1 for t in embb_throughputs if t < 5.0)

            self.logger.info(f"  eMBB Health: Avg Throughput={avg_embb_throughput:.2f}Mbps, "
                            f"Violations={embb_violations}/{len(embb_flows)}")
            
            # ✅ NEW: Show Flow IDs with their throughput
            embb_flow_details = [f'Flow{fs.flow_id}:{fs.throughput_mbps:.1f}Mbps' for fs in embb_flows]
            self.logger.info(f"  eMBB Flows: {embb_flow_details}")
            
            self.logger.info(f"  eMBB Individual Throughput (Mbps): {[f'{t:.1f}' for t in embb_throughputs]}")
            embb_delays = [fs.rtt_ms for fs in embb_flows]
            self.logger.info(f"  eMBB Individual Delays (ms): {[f'{d:.1f}' for d in embb_delays]}")
            
            if embb_violations > len(embb_flows) * 0.5:
                self.logger.warning("  ⚠️  >50% eMBB flows violating SLA (throughput < 5 Mbps)!")
        
        # ✅ NEW: mMTC Health Check
        mmtc_flows = [fs for fs in flow_states if fs.service_type == 'mMTC']
        if mmtc_flows:
            mmtc_losses = [fs.loss_rate for fs in mmtc_flows]
            avg_mmtc_loss = sum(mmtc_losses) / len(mmtc_losses)
            # mMTC SLA: Loss must be < 5%
            mmtc_violations = sum(1 for l in mmtc_losses if l > 5.0)
            
            self.logger.info(f"  mMTC Health: Avg Loss={avg_mmtc_loss:.2f}%, "
                            f"Violations={mmtc_violations}/{len(mmtc_flows)}")
            
            self.logger.info(f"  mMTC Individual Losses (%): {[f'{l:.2f}' for l in mmtc_losses]}")
            mmtc_delays = [fs.rtt_ms for fs in mmtc_flows]
            self.logger.info(f"  mMTC Individual Delays (ms): {[f'{d:.1f}' for d in mmtc_delays]}")
            
            if mmtc_violations > len(mmtc_flows) * 0.5:
                self.logger.warning("  ⚠️  >50% mMTC flows violating SLA (loss > 5%)!")

    
    def _print_final_statistics(self):
        """Print final statistics"""
        summary = self.tracker.get_summary_stats()
        
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"\nTotal Episodes: {summary.get('total_episodes', 0)}")
        print(f"Total Steps: {summary.get('total_steps', 0)}")
        print(f"Total States Received: {self.total_states_received}")
        print(f"Total Actions Sent: {self.total_actions_sent}")
        print(f"\nAverage Episode Reward:")
        print(f"  PPO: {summary.get('avg_episode_reward_ppo', 0):+.4f}")
        print(f"  A3C: {summary.get('avg_episode_reward_a3c', 0):+.4f}")
        print(f"\nPerformance:")
        print(f"  Avg Throughput: {summary.get('avg_throughput', 0):.2f} Mbps")
        print(f"  Avg Delay: {summary.get('avg_delay', 0):.2f} ms")
        print(f"  Total SLA Violations: {summary.get('total_sla_violations', 0)}")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="5G MEC Multi-Agent RL Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new training
  python main.py --preset dev
  
  # Resume from latest checkpoint (uses symlinks)
  python main.py --preset ppo_only --resume-ppo checkpoints/latest_ppo.pth
  
  # Resume from specific checkpoint
  python main.py --preset standard --resume-ppo checkpoints/episode_50_20241106_143022_ppo.pth
  
  # Resume both agents
  python main.py --preset hybrid --resume-ppo latest_ppo.pth --resume-a3c latest_a3c.pth
        """
    )
    
    parser.add_argument('--preset', type=str, default='dev',
                       choices=['dev', 'quick', 'standard', 'full', 
                               'ppo_only', 'a3c_only', 'hybrid'],
                       help='Configuration preset to use')
    
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration JSON file')
    
    parser.add_argument('--episodes', type=int,
                       help='Override max episodes')
    
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip configuration validation')
    
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum training mode')
    
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Episodes per config change in curriculum mode')
    
    # ✅ UPDATED: Better resume options with symlink support
    parser.add_argument('--resume-ppo', type=str,
                       help='Resume PPO from checkpoint (use "latest" for latest_ppo.pth symlink)')
    
    parser.add_argument('--resume-a3c', type=str,
                       help='Resume A3C from checkpoint (use "latest" for latest_a3c.pth symlink)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = MasterConfig.load(args.config)
    else:
        print(f"Using preset configuration: {args.preset}")
        config = get_config_preset(args.preset)
    
    # Override episodes if specified
    if args.episodes:
        config.training.max_episodes = args.episodes
        print(f"Override: max_episodes = {args.episodes}")
    
    # Validate configuration
    if not args.no_validate:
        warnings = validate_config(config)
        if warnings:
            print("\n⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"  - {warning}")
            print()
    
    # Print configuration summary
    config.print_summary()

     # ✅ NEW: Handle "latest" shorthand for resume
    resume_ppo = None
    resume_a3c = None
    
    if args.resume_ppo:
        if args.resume_ppo.lower() == 'latest':
            resume_ppo = 'checkpoints/latest_ppo.pth'
            print(f"📦 Resuming PPO from latest checkpoint")
        else:
            resume_ppo = args.resume_ppo
            print(f"📦 Resuming PPO from {resume_ppo}")
    
    if args.resume_a3c:
        if args.resume_a3c.lower() == 'latest':
            resume_a3c = 'checkpoints/latest_a3c.pth'
            print(f"📦 Resuming A3C from latest checkpoint")
        else:
            resume_a3c = args.resume_a3c
            print(f"📦 Resuming A3C from {resume_a3c}")
    
    # Create controller
    # ✅ NEW: Pass checkpoint paths to controller
    controller = MEC_RL_Controller(
        config, 
        resume_ppo=args.resume_ppo,
        resume_a3c=args.resume_a3c
    )
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n🛑 Interrupt received...")
        controller.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start controller
    if not controller.start():
        print("❌ Failed to start controller")
        return 1
    
    print("🚀 Controller running. Waiting for NS-3 simulation...")
    print("   Press Ctrl+C to stop\n")
    
    # ✅ Curriculum training mode
    if args.curriculum:
        controller.run_curriculum_training(
            total_episodes=config.training.max_episodes,
            checkpoint_interval=args.checkpoint_interval
        )
    else:
        if not controller.start():
            return 1
        
        print("🚀 Controller running. Press Ctrl+C to stop\n")
        # Regular training
        try:
            while controller.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    controller.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())