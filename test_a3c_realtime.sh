#!/bin/bash
# test_a3c_realtime.sh - Comprehensive A3C testing
# Place in: ai_controller/

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          A3C REAL-TIME EFFECTIVENESS TEST                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Clean previous runs
echo "1️⃣  Cleaning previous data..."
rm -f mec_metrics.csv mec_flowmon.xml
rm -f ~/Desktop/ns-allinone-3.45/ns-3.45/mec_*.pcap
echo "   ✅ Cleaned"
echo ""

# Step 2: Start action monitor
echo "2️⃣  Starting action monitor..."
python3 action_monitor.py &
MONITOR_PID=$!
sleep 2
echo "   ✅ Monitor running (PID: $MONITOR_PID)"
echo ""

# Step 3: Start AI controller with A3C
echo "3️⃣  Starting AI controller (A3C only mode)..."
python3 main.py --preset a3c_only --episodes 1 &
AI_PID=$!
sleep 3
echo "   ✅ AI controller running (PID: $AI_PID)"
echo ""

# Step 4: Prompt to start NS-3
echo "4️⃣  Ready to start NS-3 simulation"
echo ""
echo "   Run this command in another terminal:"
echo "   ┌─────────────────────────────────────────────────────────────┐"
echo "   │ cd ~/Desktop/ns-allinone-3.45/ns-3.45                       │"
echo "   │ ./ns3 run 'scratch/mec_full_simulation --enableRL=true'    │"
echo "   └─────────────────────────────────────────────────────────────┘"
echo ""
echo "   Press [Enter] when NS-3 has FINISHED..."
read -r

# Step 5: Wait for completion
echo ""
echo "5️⃣  Waiting for AI controller to finish..."
wait $AI_PID
echo "   ✅ AI controller stopped"
echo ""

# Step 6: Kill monitor
echo "6️⃣  Stopping monitor..."
kill $MONITOR_PID 2>/dev/null
echo "   ✅ Monitor stopped"
echo ""

# Step 7: Analyze results
echo "7️⃣  Analyzing results..."
echo ""
python3 diagnosis_script.py
echo ""

# Step 8: Compare with baseline
echo "8️⃣  Generate comparison report..."
echo ""
python3 - <<EOF
import pandas as pd
import numpy as np

print("="*70)
print("    FINAL ANALYSIS: A3C vs BASELINE")
print("="*70)

try:
    df = pd.read_csv('mec_metrics.csv')
    
    # Calculate key metrics
    avg_throughput = df['throughput_mbps'].mean()
    avg_delay = df['avg_delay_ms'].mean()
    avg_loss = df['loss_rate'].mean()
    cwnd_variance = df['cwnd'].var()
    
    print(f"\n📊 Network Performance:")
    print(f"   Avg Throughput: {avg_throughput:.2f} Mbps")
    print(f"   Avg Delay: {avg_delay:.2f} ms")
    print(f"   Avg Loss Rate: {avg_loss:.2f}%")
    print(f"   CWND Variance: {cwnd_variance:.2f}")
    
    # Check if metrics changed over time
    timestamps = df['timestamp'].unique()
    if len(timestamps) > 1:
        early_data = df[df['timestamp'] <= timestamps[len(timestamps)//3]]
        late_data = df[df['timestamp'] >= timestamps[2*len(timestamps)//3]]
        
        early_tput = early_data['throughput_mbps'].mean()
        late_tput = late_data['throughput_mbps'].mean()
        
        improvement = ((late_tput - early_tput) / early_tput) * 100
        
        print(f"\n📈 Learning Progress:")
        print(f"   Early Throughput: {early_tput:.2f} Mbps")
        print(f"   Late Throughput: {late_tput:.2f} Mbps")
        print(f"   Improvement: {improvement:+.2f}%")
        
        if improvement > 5:
            print(f"   ✅ A3C appears to be learning!")
        elif improvement < -5:
            print(f"   ⚠️  Performance degraded (may need more training)")
        else:
            print(f"   ⚠️  No significant change (check if actions applied)")
    
    # Action effectiveness check
    print(f"\n🔧 Action Effectiveness:")
    
    flow_changes = {}
    for flow_id in df['flow_id'].unique():
        flow_data = df[df['flow_id'] == flow_id]
        
        if len(flow_data) < 5:
            continue
        
        tput_std = flow_data['throughput_mbps'].std()
        cwnd_std = flow_data['cwnd'].std()
        
        flow_changes[flow_id] = (tput_std, cwnd_std)
    
    static_flows = sum(1 for (t, c) in flow_changes.values() if t < 0.5 and c < 5)
    dynamic_flows = len(flow_changes) - static_flows
    
    print(f"   Dynamic Flows: {dynamic_flows}/{len(flow_changes)}")
    print(f"   Static Flows: {static_flows}/{len(flow_changes)}")
    
    if static_flows > dynamic_flows:
        print(f"   ❌ PROBLEM: Actions may not be reaching NS-3!")
        print(f"      → Check g_flowIdToPort in NS-3")
        print(f"      → Verify ApplyAction() is called")
        print(f"      → Check socket ACKs")
    else:
        print(f"   ✅ Actions appear to be applied")
    
    print("\n" + "="*70)
    
except Exception as e:
    print(f"❌ Analysis failed: {e}")
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    TEST COMPLETE                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Output files:"
echo "   - mec_metrics.csv (check CWND column for changes)"
echo "   - mec_flowmon.xml (NS-3 statistics)"
echo ""
echo "🔍 Next: Compare with baseline (run without --enableRL)"