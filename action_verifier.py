#!/usr/bin/env python3
"""
action_verifier.py - Verify RL actions are actually applied in NS-3
Place in: ai_controller/action_verifier.py
"""

import numpy as np
from typing import Dict
from collections import defaultdict
import time
import uuid


class ActionVerifier:
    """Tracks actions and verifies their impact on network metrics"""
    
    def __init__(self, verification_window: int = 3):
        self.verification_window = verification_window
        self.pending_verifications = defaultdict(list)
        self.pending_actions = {}      # Add this
        self.executed_actions = {}     # Add this
        self.total_actions = 0
        self.verified_actions = 0
        self.failed_verifications = 0
        self.action_impacts = defaultdict(list)
    
    def record_action(self, flow_id: int, action: str, state_before: Dict):
        action_id = str(uuid.uuid4())
        self.pending_actions[action_id] = {
            'flow_id': flow_id,
            'action': action,
            'state_before': state_before.copy(),
            'timestamp': time.time(),
            'executed': False,
            'verified': False
        }
        self.total_actions += 1
        return action_id

    def verify_execution(self, action_id: str, ack_data: Dict):
        """Verify that NS-3 executed the action"""
        if action_id not in self.pending_actions:
            return False
            
        action = self.pending_actions[action_id]
        action['executed'] = ack_data['success']
        action['execution_details'] = ack_data['details']
        
        if action['executed']:
            self.executed_actions[action_id] = action
            
        return action['executed']

    def verify_impact(self, flow_id: int, state_after: Dict) -> Dict:
        """Verify if previous actions had expected impact"""
        results = {}

        for action_id, action in self.executed_actions.items():
            if action['flow_id'] != flow_id or action['verified']:
                continue
                
            impact = self._check_impact(
                action['action'],
                action['state_before'],
                state_after
            )
            
            action['verified'] = True
            action['impact'] = impact
            
            if impact['success']:
                self.verified_actions += 1
            else:
                self.failed_verifications += 1
                
            self.action_impacts[action['action']].append(impact)
            results[action_id] = impact
        
        return results
    
    def _check_impact(self, action: str, before: Dict, after: Dict) -> Dict:
        """Check if action had expected impact"""
        impact = {'action': action, 'success': False, 'details': ''}
        
        # # INCREASE_PRIORITY
        # if action == "INCREASE_PRIORITY":
        #     tput_before = before.get('throughput_mbps', 0)
        #     tput_after = after.get('throughput_mbps', 0)
        #     delay_before = before.get('rtt_ms', 0)
        #     delay_after = after.get('rtt_ms', 0)
            
        #     if tput_after > tput_before * 1.05 or delay_after < delay_before * 0.95:
        #         impact['success'] = True
        #         impact['details'] = f"✅ Throughput or delay improved"
        #     else:
        #         impact['details'] = f"❌ No improvement (tput: {tput_before:.1f}→{tput_after:.1f}, delay: {delay_before:.1f}→{delay_after:.1f})"

        # # DECREASE_PRIORITY
        # elif action == "DECREASE_PRIORITY":
        #     delay_before = before.get('rtt_ms', 0)
        #     delay_after = after.get('rtt_ms', 0)
            
        #     if delay_after > delay_before:
        #         impact['success'] = True
        #         impact['details'] = f"✅ Priority decreased as expected"
        #     else:
        #         impact['details'] = f"❌ Delay did not increase as expected ({delay_before:.1f}→{delay_after:.1f}ms)"

        # INCREASE_RATE
        if action == "INCREASE_RATE":
            tput_before = before.get('throughput_mbps', 0)
            tput_after = after.get('throughput_mbps', 0)
            
            if tput_after > tput_before * 1.1:  # 10% increase
                impact['success'] = True
                impact['details'] = f"✅ Rate increased ({tput_before:.1f}→{tput_after:.1f} Mbps)"
            else:
                impact['details'] = f"❌ Rate did not increase enough ({tput_before:.1f}→{tput_after:.1f} Mbps)"
        
        elif action == "DECREASE_RATE":
            tput_before = before.get('throughput_mbps', 0)
            tput_after = after.get('throughput_mbps', 0)
            
            if tput_after < tput_before * 0.9:  # 10% decrease
                impact['success'] = True
                impact['details'] = f"✅ Rate decreased ({tput_before:.1f}→{tput_after:.1f} Mbps)"
            else:
                impact['details'] = f"❌ Rate did not decrease enough ({tput_before:.1f}→{tput_after:.1f} Mbps)"
        
        elif action == "INCREASE_BURST_SIZE":
            # Verify burst size increase through throughput spike
            tput_before = before.get('throughput_mbps', 0)
            tput_after = after.get('throughput_mbps', 0)
            
            if tput_after > tput_before * 1.2:  # Expect higher impact
                impact['success'] = True
                impact['details'] = f"✅ Burst size increase effective"
            else:
                impact['details'] = f"❌ Burst size increase not effective"
        
        elif action == "DECREASE_BURST_SIZE":
            tput_before = before.get('throughput_mbps', 0)
            tput_after = after.get('throughput_mbps', 0)
            
            if tput_after < tput_before * 0.9:
                impact['success'] = True
                impact['details'] = f"✅ Burst size decrease effective"
            else:
                impact['details'] = f"❌ Burst size decrease not effective"
        
        elif action == "NO_CHANGE":
            # For no-op, consider it successful if metrics didn't change dramatically
            tput_change = abs(after.get('throughput_mbps', 0) - before.get('throughput_mbps', 0))
            delay_change = abs(after.get('rtt_ms', 0) - before.get('rtt_ms', 0))
            
            if tput_change < 5 and delay_change < 10:
                impact['success'] = True
                impact['details'] = f"✅ Metrics stable as expected"
            else:
                impact['details'] = f"❌ Unexpected changes for NO_OP"
        else:
            impact['details'] = f"❌ Unknown action type: {action}"
        
        return impact
    
    def print_summary(self):
        """Print verification summary"""
        print("\n" + "="*70)
        print("ACTION VERIFICATION SUMMARY")
        print("="*70)
        print(f"  Total Actions:     {self.total_actions}")
        print(f"  Verified:          {self.verified_actions}")
        print(f"  Failed:            {self.failed_verifications}")
        
        if self.verified_actions + self.failed_verifications > 0:
            rate = self.verified_actions / (self.verified_actions + self.failed_verifications) * 100
            print(f"  Success Rate:      {rate:.1f}%")
        
        print("="*70 + "\n")