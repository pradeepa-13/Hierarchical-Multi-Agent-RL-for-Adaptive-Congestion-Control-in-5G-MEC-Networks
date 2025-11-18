#!/usr/bin/env python3
"""
diagnosis_script.py - Verify A3C is actually being used
Place in: ai_controller/diagnosis_script.py

Run this DURING simulation to check if A3C actions are happening
"""

import json
import time

def analyze_metrics_csv(csv_file="mec_metrics.csv"):
    """Check if metrics are changing over time (indicating actions are working)"""
    import pandas as pd
    
    print("\n🔍 ANALYZING NETWORK METRICS FOR A3C IMPACT")
    print("="*70)
    
    try:
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print("❌ No metrics data yet!")
            return
        
        # Group by flow_id and check variance over time
        print("\n📊 Per-Flow Variance Analysis:")
        print("-"*70)
        
        for flow_id in df['flow_id'].unique():
            flow_data = df[df['flow_id'] == flow_id]
            
            if len(flow_data) < 3:
                continue
            
            # Calculate variance in key metrics
            tput_var = flow_data['throughput_mbps'].var()
            delay_var = flow_data['avg_delay_ms'].var()
            cwnd_var = flow_data['cwnd'].var()
            
            print(f"\nFlow {flow_id}:")
            print(f"  Throughput variance: {tput_var:.2f}")
            print(f"  Delay variance: {delay_var:.2f}")
            print(f"  CWND variance: {cwnd_var:.2f}")
            
            # If variance is near zero, actions aren't working
            if tput_var < 0.01 and delay_var < 0.01 and cwnd_var < 0.01:
                print(f"  ⚠️  WARNING: Metrics barely changing! Actions may not be applied.")
            else:
                print(f"  ✅ Metrics are changing (actions likely working)")
        
        # Check CWND specifically (should change if TCP actions work)
        print("\n📈 CWND Analysis (TCP Action Indicator):")
        print("-"*70)
        
        tcp_flows = df[df['cwnd'] > 0]
        
        if tcp_flows.empty:
            print("❌ No TCP flows with CWND data!")
        else:
            for flow_id in tcp_flows['flow_id'].unique():
                flow_cwnd = tcp_flows[tcp_flows['flow_id'] == flow_id]['cwnd']
                
                cwnd_changes = flow_cwnd.diff().abs().sum()
                
                print(f"Flow {flow_id}: CWND changed {cwnd_changes:.0f} segments total")
                
                if cwnd_changes < 10:
                    print(f"  ⚠️  Suspiciously stable CWND (may indicate actions not applied)")
                else:
                    print(f"  ✅ CWND is dynamic")
        
    except Exception as e:
        print(f"❌ Error analyzing metrics: {e}")


def check_action_log():
    """Check if actions are being sent and acknowledged"""
    print("\n🔍 CHECKING ACTION ACKNOWLEDGMENTS")
    print("="*70)
    
    # This would need to be implemented in socket_comm.py
    # For now, manually check console output for:
    # - "Action sent" messages
    # - "Action acknowledged" messages
    # - Actual changes in NS-3 (e.g., "Set data rate to X")
    
    print("""
    ⚠️  Manual Check Required:
    
    Look for these patterns in console output:
    
    ✅ GOOD SIGNS:
      - "💫 Action executed: INCREASE_RATE for Flow X"
      - "✅ Flow X: Increased rate to Y Mbps"
      - CWND values changing in metrics
      
    ❌ BAD SIGNS:
      - Only "NO_CHANGE" actions
      - "⚠️ Flow X not found in port mapping"
      - Metrics completely flat/unchanging
    """)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("    A3C DIAGNOSIS TOOL")
    print("="*70)
    
    print("\n⏳ Waiting 30 seconds for simulation to generate data...")
    time.sleep(30)
    
    analyze_metrics_csv()
    check_action_log()
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    
    print("""
    🔧 NEXT STEPS IF ACTIONS NOT WORKING:
    
    1. Check g_flowIdToPort mapping in NS-3:
       - Is it populated?
       - Are flow IDs correct?
    
    2. Check action execution in NS-3:
       - Add debug prints in ApplyAction()
       - Verify OnOffApplication attributes are changing
    
    3. Check socket communication:
       - Are actions reaching NS-3?
       - Are ACKs coming back?
    
    4. Verify A3C is being used:
       - Check main.py logs for "A3C action" messages
       - Confirm phase is 'a3c_only' or 'combined'
    """)