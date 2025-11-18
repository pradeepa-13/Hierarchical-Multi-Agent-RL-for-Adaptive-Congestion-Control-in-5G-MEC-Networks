#!/usr/bin/env python3
"""
compare_all_actions.py - Test all A3C actions and compare results
Place in: ai_controller/compare_all_actions.py

Runs multiple simulations, each with a different forced action,
then generates comparison report

Usage:
    python3 compare_all_actions.py --auto
"""

import subprocess
import time
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from rl_spaces import EdgeAction

class ActionComparator:
    """Compare effects of all A3C actions"""
    
    def __init__(self, output_dir="action_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.actions_to_test = [
            EdgeAction.NO_CHANGE,
            EdgeAction.SCALE_UP_RESOURCES,
            EdgeAction.SCALE_DOWN_RESOURCES,
            EdgeAction.OFFLOAD_TO_CLOUD,
            EdgeAction.ACTIVATE_CACHING,
            EdgeAction.ADJUST_QUEUE_SIZE,
            EdgeAction.ENABLE_LOAD_BALANCING
        ]
        
        self.results = {}
    
    def run_test(self, action: EdgeAction, manual=False):
        """Run test for single action"""
        print("\n" + "="*70)
        print(f"Testing: {action.name}")
        print("="*70)
        
        # Start tester
        tester_cmd = f"python3 test_single_action.py --action {action.name} --count 15"
        
        if manual:
            print(f"\n📋 Manual Test Instructions:")
            print(f"   1. Terminal 1: {tester_cmd}")
            print(f"   2. Terminal 2: ./ns3 run 'scratch/mec_full_simulation --enableRL=true'")
            print(f"   3. Wait for completion")
            print(f"\n   Press [Enter] when done...")
            input()
        else:
            print(f"⚠️  Auto mode: Start NS-3 in another terminal!")
            print(f"   Run: ./ns3 run 'scratch/mec_full_simulation --enableRL=true'")
            print(f"\n   Press [Enter] to start tester...")
            input()
            
            # Start tester in background
            print(f"Starting tester...")
            proc = subprocess.Popen(tester_cmd, shell=True)
            
            print(f"⏳ Waiting for simulation to complete (90 seconds)...")
            time.sleep(90)
            
            # Kill tester
            proc.terminate()
            proc.wait()
        
        # Copy results
        result_file = self.output_dir / f"{action.name}_metrics.csv"
        subprocess.run(f"cp mec_metrics.csv {result_file}", shell=True)
        
        print(f"✅ Results saved to {result_file}")
        
        # Analyze
        self._analyze_result(action, result_file)
    
    def _analyze_result(self, action: EdgeAction, csv_file: Path):
        """Analyze single test result"""
        try:
            df = pd.read_csv(csv_file)
            
            # Calculate key metrics
            metrics = {
                'action': action.name,
                'avg_throughput': df['throughput_mbps'].mean(),
                'throughput_std': df['throughput_mbps'].std(),
                'avg_delay': df['avg_delay_ms'].mean(),
                'delay_std': df['avg_delay_ms'].std(),
                'avg_loss': df['loss_rate'].mean(),
                'cwnd_variance': df['cwnd'].var(),
                'sla_violations': sum(
                    1 for _, row in df.iterrows() 
                    if row['service_type'] == 'URLLC' and row['avg_delay_ms'] > 50
                ),
                'total_samples': len(df)
            }
            
            # Calculate improvement over time
            timestamps = df['timestamp'].unique()
            if len(timestamps) > 2:
                mid_point = len(timestamps) // 2
                
                early = df[df['timestamp'] <= timestamps[mid_point]]
                late = df[df['timestamp'] > timestamps[mid_point]]
                
                metrics['early_throughput'] = early['throughput_mbps'].mean()
                metrics['late_throughput'] = late['throughput_mbps'].mean()
                metrics['throughput_improvement'] = (
                    (metrics['late_throughput'] - metrics['early_throughput']) / 
                    metrics['early_throughput'] * 100
                )
            
            self.results[action.name] = metrics
            
            print(f"\n📊 Results for {action.name}:")
            print(f"   Avg Throughput: {metrics['avg_throughput']:.2f} ± {metrics['throughput_std']:.2f} Mbps")
            print(f"   Avg Delay: {metrics['avg_delay']:.2f} ± {metrics['delay_std']:.2f} ms")
            print(f"   Avg Loss: {metrics['avg_loss']:.2f}%")
            print(f"   CWND Variance: {metrics['cwnd_variance']:.2f}")
            print(f"   SLA Violations: {metrics['sla_violations']}")
            
            if 'throughput_improvement' in metrics:
                print(f"   Throughput Change: {metrics['throughput_improvement']:+.2f}%")
        
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            self.results[action.name] = {'error': str(e)}
    
    def generate_report(self):
        """Generate comparison report"""
        print("\n" + "="*70)
        print("    COMPREHENSIVE ACTION COMPARISON REPORT")
        print("="*70)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results).T
        
        # Print summary table
        print("\n📊 Summary Table:")
        print("─"*70)
        print(f"{'Action':<25} {'Throughput':>12} {'Delay':>10} {'SLA Viol':>10}")
        print("─"*70)
        
        for action_name, metrics in self.results.items():
            if 'error' in metrics:
                continue
            
            print(f"{action_name:<25} "
                  f"{metrics['avg_throughput']:>10.2f} Mbps "
                  f"{metrics['avg_delay']:>8.2f} ms "
                  f"{metrics['sla_violations']:>10d}")
        
        print("─"*70)
        
        # Best performers
        print("\n🏆 Best Performers:")
        
        best_throughput = max(
            (m for m in self.results.values() if 'error' not in m),
            key=lambda x: x['avg_throughput']
        )
        print(f"   Highest Throughput: {best_throughput['action']} "
              f"({best_throughput['avg_throughput']:.2f} Mbps)")
        
        best_delay = min(
            (m for m in self.results.values() if 'error' not in m),
            key=lambda x: x['avg_delay']
        )
        print(f"   Lowest Delay: {best_delay['action']} "
              f"({best_delay['avg_delay']:.2f} ms)")
        
        best_sla = min(
            (m for m in self.results.values() if 'error' not in m),
            key=lambda x: x['sla_violations']
        )
        print(f"   Fewest SLA Violations: {best_sla['action']} "
              f"({best_sla['sla_violations']} violations)")
        
        # Most dynamic
        most_dynamic = max(
            (m for m in self.results.values() if 'error' not in m),
            key=lambda x: x['cwnd_variance']
        )
        print(f"   Most Dynamic (CWND): {most_dynamic['action']} "
              f"(variance={most_dynamic['cwnd_variance']:.2f})")
        
        # Save report
        report_file = self.output_dir / "comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ACTION COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            for action_name, metrics in self.results.items():
                if 'error' in metrics:
                    continue
                
                f.write(f"\n{action_name}\n")
                f.write("-"*50 + "\n")
                for key, value in metrics.items():
                    if key != 'action':
                        f.write(f"  {key}: {value}\n")
        
        print(f"\n✅ Full report saved to {report_file}")
        
        # Save CSV
        csv_file = self.output_dir / "comparison_metrics.csv"
        df.to_csv(csv_file)
        print(f"✅ CSV data saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare all A3C actions")
    parser.add_argument('--auto', action='store_true',
                       help='Run automatically (still needs manual NS-3 restart)')
    parser.add_argument('--actions', nargs='+',
                       help='Specific actions to test (default: all)')
    
    args = parser.parse_args()
    
    comparator = ActionComparator()
    
    # Determine which actions to test
    if args.actions:
        actions_to_test = [EdgeAction[a] for a in args.actions]
    else:
        actions_to_test = comparator.actions_to_test
    
    print("\n" + "="*70)
    print("    A3C ACTION COMPARISON SUITE")
    print("="*70)
    print(f"\nWill test {len(actions_to_test)} actions:")
    for action in actions_to_test:
        print(f"  - {action.name}")
    
    print("\n⚠️  IMPORTANT:")
    print("   - Each test requires NS-3 restart")
    print("   - Simulation runs for ~60-90 seconds")
    print("   - Results saved to action_comparison/")
    print("\n" + "="*70 + "\n")
    
    input("Press [Enter] to begin...")
    
    # Run tests
    for i, action in enumerate(actions_to_test, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST {i}/{len(actions_to_test)}")
        print(f"{'='*70}")
        
        comparator.run_test(action, manual=(not args.auto))
        
        if i < len(actions_to_test):
            print(f"\n✅ Test {i} complete. Prepare for next test:")
            print(f"   1. Stop current NS-3 (Ctrl+C)")
            print(f"   2. Clean: rm mec_metrics.csv")
            print(f"   3. Press [Enter] to continue...")
            input()
    
    # Generate report
    comparator.generate_report()
    
    print("\n" + "="*70)
    print("    ALL TESTS COMPLETE")
    print("="*70)
    print(f"\n📁 Results in: {comparator.output_dir}/")
    print("\n✅ You now have data to compare all A3C actions!")


if __name__ == "__main__":
    main()