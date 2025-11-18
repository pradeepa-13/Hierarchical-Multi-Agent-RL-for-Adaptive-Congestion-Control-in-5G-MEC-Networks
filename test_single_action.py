#!/usr/bin/env python3
"""
test_single_action.py - Force specific A3C action for testing
Place in: ai_controller/test_single_action.py

Tests individual A3C actions to verify they work correctly

Usage:
    python3 test_single_action.py --action SCALE_UP
    python3 test_single_action.py --action OFFLOAD_TO_CLOUD
"""

import argparse
import sys
import time
from typing import Optional

from rl_spaces import EdgeAction
from config import get_config_preset
from socket_comm import SocketCommunicator, NetworkState, RLAction, SocketConfig

class SingleActionTester:
    """Force a specific A3C action for testing"""
    
    def __init__(self, forced_action: EdgeAction):
        self.forced_action = forced_action
        self.action_count = 0
        self.max_actions = 10  # Apply action 10 times then observe
        
        config = get_config_preset('dev')
        socket_config = SocketConfig(
            host=config.socket.host,
            port=config.socket.port,
            verbose=True
        )
        
        self.comm = SocketCommunicator(socket_config)
        
        print("\n" + "="*70)
        print(f"    TESTING A3C ACTION: {forced_action.name}")
        print("="*70)
        print(f"\nForced action: {forced_action.name} ({forced_action.value})")
        print(f"Will apply action {self.max_actions} times then stop")
        print("="*70 + "\n")
    
    def start(self):
        """Start tester"""
        if not self.comm.start():
            print("❌ Failed to start socket")
            return False
        
        self.comm.register_state_callback(self._process_state)
        print("✅ Tester started, waiting for NS-3...\n")
        return True
    
    def _process_state(self, state: NetworkState) -> Optional[RLAction]:
        """Process state and return forced action"""
        
        if state.timestamp == 0.0:
            action = RLAction()
            action.flow_actions = [{'type': 'no_op', 'reason': 'zero_time'}]
            return action
        
        # Stop after max_actions
        if self.action_count >= self.max_actions:
            print(f"\n✅ Applied action {self.max_actions} times, now observing...")
            action = RLAction()
            action.flow_actions = [{'type': 'observe', 'reason': 'max_actions_reached'}]
            return action
        
        action = RLAction()
        
        # Get flows (simulate getting location flows)
        tcp_flows = [f for f in state.flows if f.get('protocol') == 'TCP']
        
        if not tcp_flows:
            print(f"⏳ [t={state.timestamp:.1f}s] No TCP flows yet")
            action.flow_actions = [{'type': 'waiting', 'reason': 'no_flows'}]
            return action
        
        # Determine edge location (simplified: first flow's source)
        first_flow = tcp_flows[0]
        src_ip = first_flow.get('src', '')
        
        if src_ip.startswith('10.'):
            edge_location = 'india'
        elif src_ip.startswith('20.'):
            edge_location = 'uk'
        else:
            edge_location = 'unknown'
        
        print(f"\n📋 [t={state.timestamp:.1f}s] Action {self.action_count + 1}/{self.max_actions}")
        print(f"   Applying {self.forced_action.name} to {len(tcp_flows)} flows at {edge_location}")
        
        # ✅ APPLY FORCED ACTION
        if self.forced_action == EdgeAction.SCALE_UP_RESOURCES:
            for flow in tcp_flows:
                action.flow_actions.extend([
                    {
                        'flow_id': flow['id'],
                        'action': 'INCREASE_RATE',
                        'reason': f'test_scale_up_{edge_location}'
                    },
                    {
                        'flow_id': flow['id'],
                        'action': 'INCREASE_BURST_SIZE',
                        'reason': f'test_scale_up_burst_{edge_location}'
                    }
                ])
        
        elif self.forced_action == EdgeAction.SCALE_DOWN_RESOURCES:
            for flow in tcp_flows:
                if flow.get('service_type') != 'URLLC':
                    action.flow_actions.extend([
                        {
                            'flow_id': flow['id'],
                            'action': 'DECREASE_RATE',
                            'reason': f'test_scale_down_{edge_location}'
                        },
                        {
                            'flow_id': flow['id'],
                            'action': 'DECREASE_BURST_SIZE',
                            'reason': f'test_scale_down_burst_{edge_location}'
                        }
                    ])
        
        elif self.forced_action == EdgeAction.OFFLOAD_TO_CLOUD:
            for flow in tcp_flows:
                service_type = flow.get('service_type', 'Unknown')
                if service_type in ['Background', 'mMTC']:
                    action.flow_actions.append({
                        'flow_id': flow['id'],
                        'action': 'DECREASE_RATE',
                        'reason': f'test_offload_{edge_location}'
                    })
                elif service_type == 'eMBB':
                    action.flow_actions.append({
                        'flow_id': flow['id'],
                        'action': 'DECREASE_BURST_SIZE',
                        'reason': f'test_partial_offload_{edge_location}'
                    })
        
        elif self.forced_action == EdgeAction.ACTIVATE_CACHING:
            for flow in tcp_flows:
                if flow.get('service_type') == 'eMBB':
                    action.flow_actions.append({
                        'flow_id': flow['id'],
                        'action': 'INCREASE_BURST_SIZE',
                        'reason': f'test_caching_{edge_location}'
                    })
        
        elif self.forced_action == EdgeAction.ADJUST_QUEUE_SIZE:
            # Simulate congestion check
            sla_violations = sum(1 for f in tcp_flows if f.get('rtt_ms', 0) > 50)
            
            if sla_violations > len(tcp_flows) // 2:
                # Congested: increase URLLC rate
                for flow in tcp_flows:
                    if flow.get('service_type') == 'URLLC':
                        action.flow_actions.append({
                            'flow_id': flow['id'],
                            'action': 'INCREASE_RATE',
                            'reason': f'test_queue_clear_{edge_location}'
                        })
            else:
                # Low congestion: reduce buffering
                for flow in tcp_flows:
                    action.flow_actions.append({
                        'flow_id': flow['id'],
                        'action': 'DECREASE_BURST_SIZE',
                        'reason': f'test_queue_reduce_{edge_location}'
                    })
        
        elif self.forced_action == EdgeAction.ENABLE_LOAD_BALANCING:
            # Calculate average throughput
            tputs = [f.get('throughput_mbps', 0) for f in tcp_flows]
            avg_tput = sum(tputs) / len(tputs) if tputs else 0
            
            for flow in tcp_flows:
                flow_tput = flow.get('throughput_mbps', 0)
                
                if flow_tput < avg_tput * 0.8:
                    action.flow_actions.append({
                        'flow_id': flow['id'],
                        'action': 'INCREASE_RATE',
                        'reason': f'test_balance_up_{edge_location}'
                    })
                elif flow_tput > avg_tput * 1.2:
                    action.flow_actions.append({
                        'flow_id': flow['id'],
                        'action': 'DECREASE_RATE',
                        'reason': f'test_balance_down_{edge_location}'
                    })
        
        self.action_count += 1
        
        print(f"   → Sending {len(action.flow_actions)} flow actions")
        
        return action
    
    def stop(self):
        """Stop tester"""
        self.comm.stop()
        
        print("\n" + "="*70)
        print("TEST COMPLETE")
        print("="*70)
        print(f"\nActions applied: {self.action_count}")
        print("\n📊 Check results:")
        print("   1. Review mec_metrics.csv for metric changes")
        print("   2. Check NS-3 console for action execution")
        print("   3. Run: python3 diagnosis_script.py")
        print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test individual A3C actions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_single_action.py --action SCALE_UP
  python3 test_single_action.py --action OFFLOAD_TO_CLOUD
  python3 test_single_action.py --action LOAD_BALANCE
  
Available actions:
  - NO_CHANGE
  - SCALE_UP_RESOURCES
  - SCALE_DOWN_RESOURCES
  - OFFLOAD_TO_CLOUD
  - ACTIVATE_CACHING
  - ADJUST_QUEUE_SIZE
  - ENABLE_LOAD_BALANCING
        """
    )
    
    parser.add_argument('--action', type=str, required=True,
                       help='A3C action to test')
    
    parser.add_argument('--count', type=int, default=10,
                       help='Number of times to apply action')
    
    args = parser.parse_args()
    
    # Parse action
    try:
        forced_action = EdgeAction[args.action]
    except KeyError:
        print(f"❌ Invalid action: {args.action}")
        print(f"\nAvailable actions: {[a.name for a in EdgeAction]}")
        return 1
    
    # Create tester
    tester = SingleActionTester(forced_action)
    tester.max_actions = args.count
    
    if not tester.start():
        return 1
    
    print("🚀 Tester running...")
    print("   Start NS-3 simulation now:")
    print("   ./ns3 run 'scratch/mec_full_simulation --enableRL=true'\n")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    
    tester.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())