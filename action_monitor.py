#!/usr/bin/env python3
"""
action_monitor.py - Real-time action effectiveness dashboard
Place in: ai_controller/action_monitor.py

Monitors mec_metrics.csv and tracks if actions are having effect

Usage:
    python action_monitor.py
"""

import pandas as pd
import time
from collections import defaultdict, deque
import os

class ActionMonitor:
    def __init__(self, csv_file="../ns-3.45/mec_metrics.csv"):
        self.csv_file = csv_file
        self.last_size = 0
        self.action_history = defaultdict(lambda: deque(maxlen=10))
        self.metric_history = defaultdict(lambda: deque(maxlen=10))
        
    def monitor(self, interval=5.0):
        """Monitor metrics and detect action effects"""
        print("\n" + "="*70)
        print("    REAL-TIME ACTION EFFECTIVENESS MONITOR")
        print("="*70)
        print("\nMonitoring actions and their effects...")
        print("Press Ctrl+C to stop\n")
        
        while True:
            try:
                if not os.path.exists(self.csv_file):
                    print("⏳ Waiting for metrics file...")
                    time.sleep(interval)
                    continue
                
                # Check if file updated
                current_size = os.path.getsize(self.csv_file)
                if current_size == self.last_size:
                    time.sleep(interval)
                    continue
                
                self.last_size = current_size
                
                # Read latest data
                df = pd.read_csv(self.csv_file)
                
                if df.empty:
                    time.sleep(interval)
                    continue
                
                # Analyze recent changes
                self._analyze_changes(df)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n\n✅ Monitoring stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(interval)
    
    def _analyze_changes(self, df):
        """Analyze if metrics are changing (indicating actions work)"""
        print("\n" + "─"*70)
        print(f"📊 Analysis at t={df['timestamp'].max():.1f}s")
        print("─"*70)
        
        # Group by flow
        flow_changes = {}
        
        for flow_id in df['flow_id'].unique():
            flow_data = df[df['flow_id'] == flow_id].tail(5)
            
            if len(flow_data) < 2:
                continue
            
            # Calculate metric changes
            tput_change = flow_data['throughput_mbps'].diff().abs().sum()
            delay_change = flow_data['avg_delay_ms'].diff().abs().sum()
            cwnd_change = flow_data['cwnd'].diff().abs().sum()
            
            flow_changes[flow_id] = {
                'throughput_change': tput_change,
                'delay_change': delay_change,
                'cwnd_change': cwnd_change,
                'service': flow_data['service_type'].iloc[-1]
            }
        
        # Display results
        active_flows = 0
        static_flows = 0
        
        for flow_id, changes in flow_changes.items():
            total_change = (changes['throughput_change'] + 
                          changes['delay_change']/10 + 
                          changes['cwnd_change']/100)
            
            if total_change > 0.5:
                active_flows += 1
                status = "✅ DYNAMIC"
            else:
                static_flows += 1
                status = "⚠️  STATIC"
            
            print(f"Flow {flow_id:3d} ({changes['service']:6s}): {status} "
                  f"(Δtput={changes['throughput_change']:.1f}, "
                  f"Δdelay={changes['delay_change']:.1f}, "
                  f"Δcwnd={changes['cwnd_change']:.0f})")
        
        # Summary
        print("\n" + "─"*70)
        print(f"Summary: {active_flows} dynamic, {static_flows} static")
        
        if static_flows > active_flows:
            print("⚠️  WARNING: Most flows are static! Actions may not be applied.")
        else:
            print("✅ Actions appear to be working")
        
        print("─"*70)


if __name__ == "__main__":
    monitor = ActionMonitor()
    monitor.monitor(interval=3.0)