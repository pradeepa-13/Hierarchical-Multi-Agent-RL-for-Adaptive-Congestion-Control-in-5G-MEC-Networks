#!/usr/bin/env python3
"""
verify_resume.py - Checkpoint Verification Tool

Verifies that checkpoint resumption is working correctly by comparing:
1. Checkpoint metadata (episode, updates, steps)
2. Curriculum state (config index, episodes in config)
3. Session IDs (new vs resumed)

Usage:
    python verify_resume.py                          # Verify latest checkpoints
    python verify_resume.py --checkpoint episode_50  # Verify specific checkpoint
    python verify_resume.py --compare before after   # Compare two checkpoints
"""

import argparse
import json
import os
import torch
from datetime import datetime
from pathlib import Path


class CheckpointVerifier:
    """Verify checkpoint consistency and resumption"""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
    
    def load_checkpoint_metadata(self, checkpoint_path):
        """Load checkpoint and extract metadata"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return {
                'episode_count': checkpoint.get('episode_count', 'N/A'),
                'update_count': checkpoint.get('update_count', 'N/A'),
                'total_steps': checkpoint.get('total_steps', 'N/A'),
                'exploration_rate': checkpoint.get('exploration_rate', 'N/A'),
            }
        except Exception as e:
            print(f"❌ Error loading {checkpoint_path}: {e}")
            return None
    
    def load_curriculum_state(self):
        """Load curriculum state file"""
        state_file = self.checkpoint_dir / "curriculum_state.json"
        try:
            if state_file.exists():
                with open(state_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"❌ Error loading curriculum state: {e}")
            return None
    
    def parse_checkpoint_filename(self, filename):
        """Extract metadata from checkpoint filename"""
        # Format: episode_N_sessionid_[resumed_]agent.pth
        parts = filename.replace('.pth', '').split('_')
        
        info = {
            'filename': filename,
            'episode': None,
            'session_id': None,
            'is_resumed': False,
            'agent': None
        }
        
        try:
            if parts[0] == 'episode':
                info['episode'] = int(parts[1])
                
                # Find session_id (timestamp format: YYYYMMDD_HHMMSS)
                for i, part in enumerate(parts[2:], start=2):
                    if len(part) == 8 and part.isdigit():  # YYYYMMDD
                        # Session ID is this part + next part (HHMMSS)
                        if i + 1 < len(parts):
                            info['session_id'] = f"{part}_{parts[i+1]}"
                            
                            # Check if resumed
                            if i + 2 < len(parts) and parts[i+2] == 'resumed':
                                info['is_resumed'] = True
                                info['agent'] = parts[i+3]
                            else:
                                info['agent'] = parts[i+2]
                        break
        except Exception as e:
            print(f"⚠️  Could not fully parse {filename}: {e}")
        
        return info
    
    def find_latest_checkpoint(self, agent='ppo'):
        """Find the most recent checkpoint for an agent"""
        pattern = f"*_{agent}.pth"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def verify_latest(self):
        """Verify the latest checkpoints"""
        print("\n" + "="*70)
        print("    LATEST CHECKPOINT VERIFICATION")
        print("="*70 + "\n")
        
        # Check PPO
        ppo_checkpoint = self.find_latest_checkpoint('ppo')
        if ppo_checkpoint:
            print("📦 PPO Checkpoint:")
            self._print_checkpoint_details(ppo_checkpoint)
        else:
            print("⚠️  No PPO checkpoint found")
        
        print()
        
        # Check A3C
        a3c_checkpoint = self.find_latest_checkpoint('a3c')
        if a3c_checkpoint:
            print("📦 A3C Checkpoint:")
            self._print_checkpoint_details(a3c_checkpoint)
        else:
            print("⚠️  No A3C checkpoint found")
        
        print()
        
        # Check curriculum state
        curriculum_state = self.load_curriculum_state()
        if curriculum_state:
            print("📋 Curriculum State:")
            print(f"   Session ID: {curriculum_state['session_id']}")
            print(f"   Episodes completed: {curriculum_state['episodes_completed']}")
            print(f"   Current config index: {curriculum_state['current_config_idx']}")
            print(f"   Episodes in config: {curriculum_state['episodes_in_checkpoint']}")
            print(f"   Last saved: {curriculum_state['timestamp']}")
        else:
            print("⚠️  No curriculum state found")
        
        print("\n" + "="*70 + "\n")
    
    def _print_checkpoint_details(self, checkpoint_path):
        """Print detailed checkpoint information"""
        # Parse filename
        filename_info = self.parse_checkpoint_filename(checkpoint_path.name)
        
        print(f"   File: {checkpoint_path.name}")
        print(f"   Episode: {filename_info['episode']}")
        print(f"   Session ID: {filename_info['session_id']}")
        print(f"   Is resumed: {'Yes' if filename_info['is_resumed'] else 'No'}")
        print(f"   Agent: {filename_info['agent']}")
        
        # Load metadata
        metadata = self.load_checkpoint_metadata(checkpoint_path)
        if metadata:
            print(f"   Episode count: {metadata['episode_count']}")
            print(f"   Update count: {metadata['update_count']}")
            print(f"   Total steps: {metadata['total_steps']}")
            print(f"   Exploration rate: {metadata['exploration_rate']:.4f}")
    
    def compare_checkpoints(self, checkpoint1_name, checkpoint2_name):
        """Compare two checkpoints to verify progression"""
        print("\n" + "="*70)
        print("    CHECKPOINT COMPARISON")
        print("="*70 + "\n")
        
        cp1_path = self.checkpoint_dir / f"{checkpoint1_name}_ppo.pth"
        cp2_path = self.checkpoint_dir / f"{checkpoint2_name}_ppo.pth"
        
        if not cp1_path.exists():
            print(f"❌ Checkpoint 1 not found: {cp1_path}")
            return
        
        if not cp2_path.exists():
            print(f"❌ Checkpoint 2 not found: {cp2_path}")
            return
        
        print(f"📦 Comparing:")
        print(f"   Before: {cp1_path.name}")
        print(f"   After:  {cp2_path.name}\n")
        
        # Load metadata
        meta1 = self.load_checkpoint_metadata(cp1_path)
        meta2 = self.load_checkpoint_metadata(cp2_path)
        
        if not meta1 or not meta2:
            print("❌ Could not load checkpoint metadata")
            return
        
        # Parse filenames
        info1 = self.parse_checkpoint_filename(cp1_path.name)
        info2 = self.parse_checkpoint_filename(cp2_path.name)
        
        # Compare
        print("Comparison Results:")
        print("-" * 70)
        
        # Episode count
        ep_delta = meta2['episode_count'] - meta1['episode_count']
        print(f"Episode count:     {meta1['episode_count']} → {meta2['episode_count']} "
              f"(+{ep_delta})")
        
        # Update count
        update_delta = meta2['update_count'] - meta1['update_count']
        print(f"Update count:      {meta1['update_count']} → {meta2['update_count']} "
              f"(+{update_delta})")
        
        # Total steps
        steps_delta = meta2['total_steps'] - meta1['total_steps']
        print(f"Total steps:       {meta1['total_steps']} → {meta2['total_steps']} "
              f"(+{steps_delta})")
        
        # Exploration rate
        exp_delta = meta2['exploration_rate'] - meta1['exploration_rate']
        print(f"Exploration rate:  {meta1['exploration_rate']:.4f} → "
              f"{meta2['exploration_rate']:.4f} ({exp_delta:+.4f})")
        
        # Session ID
        print(f"\nSession IDs:")
        print(f"   Before: {info1['session_id']}")
        print(f"   After:  {info2['session_id']}")
        
        if info1['session_id'] != info2['session_id']:
            print("   ✅ Different session IDs (resumed training detected)")
        else:
            print("   ⚠️  Same session ID (continuous training)")
        
        # Verification
        print("\n" + "="*70)
        print("Verification:")
        
        if ep_delta > 0:
            print(f"   ✅ Episode count increased (+{ep_delta})")
        else:
            print(f"   ❌ Episode count did NOT increase!")
        
        if update_delta > 0:
            print(f"   ✅ Update count increased (+{update_delta})")
        else:
            print(f"   ⚠️  Update count did not increase (no training?)")
        
        if steps_delta > 0:
            print(f"   ✅ Total steps increased (+{steps_delta})")
        else:
            print(f"   ❌ Total steps did NOT increase!")
        
        if exp_delta < 0:
            print(f"   ✅ Exploration rate decreased (expected)")
        else:
            print(f"   ⚠️  Exploration rate increased or stayed same")
        
        print("="*70 + "\n")
    
    def list_all_checkpoints(self):
        """List all available checkpoints"""
        print("\n" + "="*70)
        print("    ALL CHECKPOINTS")
        print("="*70 + "\n")
        
        checkpoints = sorted(self.checkpoint_dir.glob("episode_*.pth"))
        
        if not checkpoints:
            print("No checkpoints found")
            return
        
        # Group by session
        sessions = {}
        for cp in checkpoints:
            info = self.parse_checkpoint_filename(cp.name)
            session_id = info['session_id']
            
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append((cp, info))
        
        # Print by session
        for session_id, cps in sorted(sessions.items()):
            print(f"📅 Session: {session_id}")
            
            for cp_path, info in sorted(cps, key=lambda x: x[1]['episode']):
                resumed_marker = " (RESUMED)" if info['is_resumed'] else ""
                print(f"   Episode {info['episode']:3d} - {info['agent']} "
                      f"{resumed_marker}")
            
            print()
        
        print("="*70 + "\n")
    
    def verify_resume_ready(self):
        """Check if environment is ready for resumption"""
        print("\n" + "="*70)
        print("    RESUME READINESS CHECK")
        print("="*70 + "\n")
        
        # Check for latest symlinks
        latest_ppo = self.checkpoint_dir / "latest_ppo.pth"
        latest_a3c = self.checkpoint_dir / "latest_a3c.pth"
        
        print("Symlink Status:")
        if latest_ppo.exists():
            target = os.readlink(latest_ppo)
            print(f"   ✅ latest_ppo.pth → {target}")
        else:
            print(f"   ❌ latest_ppo.pth not found")
        
        if latest_a3c.exists():
            target = os.readlink(latest_a3c)
            print(f"   ✅ latest_a3c.pth → {target}")
        else:
            print(f"   ❌ latest_a3c.pth not found")
        
        # Check curriculum state
        print("\nCurriculum State:")
        curriculum_state = self.load_curriculum_state()
        if curriculum_state:
            print(f"   ✅ curriculum_state.json found")
            print(f"      Episodes completed: {curriculum_state['episodes_completed']}")
        else:
            print(f"   ⚠️  curriculum_state.json not found")
        
        # Recommend resume command
        print("\n" + "="*70)
        print("Resume Command:")
        
        if latest_ppo.exists() and latest_a3c.exists():
            print("   python main.py --preset hybrid --curriculum \\")
            print("       --resume-ppo latest --resume-a3c latest")
        elif latest_ppo.exists():
            print("   python main.py --preset ppo_only --curriculum \\")
            print("       --resume-ppo latest")
        elif latest_a3c.exists():
            print("   python main.py --preset a3c_only --curriculum \\")
            print("       --resume-a3c latest")
        else:
            print("   ⚠️  No checkpoints available for resumption")
        
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify checkpoint resumption",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str,
                       help='Verify specific checkpoint (e.g., episode_50_20241106_143022)')
    
    parser.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'),
                       help='Compare two checkpoints')
    
    parser.add_argument('--list', action='store_true',
                       help='List all checkpoints')
    
    parser.add_argument('--ready', action='store_true',
                       help='Check if ready for resumption')
    
    args = parser.parse_args()
    
    verifier = CheckpointVerifier()
    
    if args.list:
        verifier.list_all_checkpoints()
    elif args.compare:
        verifier.compare_checkpoints(args.compare[0], args.compare[1])
    elif args.ready:
        verifier.verify_resume_ready()
    elif args.checkpoint:
        # Verify specific checkpoint
        print(f"Verifying checkpoint: {args.checkpoint}")
        # TODO: Implement specific checkpoint verification
    else:
        # Default: verify latest
        verifier.verify_latest()
        print("\nOptions:")
        print("  --list      List all checkpoints")
        print("  --compare   Compare two checkpoints")
        print("  --ready     Check if ready for resumption")


if __name__ == "__main__":
    main()