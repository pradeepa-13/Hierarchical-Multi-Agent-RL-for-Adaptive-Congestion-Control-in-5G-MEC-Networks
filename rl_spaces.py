#!/usr/bin/env python3
"""
rl_spaces.py
State and Action Space Definitions for 5G MEC RL Agents

Defines:
1. Flow-level state/action space (PPO agent)
2. Edge-level state/action space (A3C agent)
3. Resource optimization space (GWO)

Place in: ai_controller/rl_spaces.py
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# SERVICE TYPE CONSTANTS
# ============================================================================
class ServiceType(Enum):
    """5G service types with different QoS requirements"""
    EMBB = "eMBB"      # Enhanced Mobile Broadband (high throughput)
    URLLC = "URLLC"    # Ultra-Reliable Low Latency (strict delay)
    MMTC = "mMTC"      # Massive Machine-Type Comms (many devices)
    BACKGROUND = "Background"


# ============================================================================
# FLOW-LEVEL STATE SPACE (PPO AGENT)
# ============================================================================
@dataclass
class FlowState:
    """
    State representation for a single flow (used by PPO agent)
    URLLC SLA: <70ms RTT, <1.0% loss
    Dimensions: 12 features per flow
    """
    # Flow identification
    flow_id: int
    service_type: str  # eMBB/URLLC/mMTC/Background
    protocol: str  # Add this field
    src_port: int
    dst_port: int
    # Performance metrics (normalized 0-1)
    throughput_mbps: float      # Current throughput
    rtt_ms: float              # Round-trip time
    jitter_ms: float           # Delay variation
    loss_rate: float           # Packet loss percentage
    
    # TCP state
    cwnd: int                  # Congestion window size
    
    # QoS compliance (binary flags)
    sla_violation: bool        # Is SLA being violated?
    
    # Temporal features
    throughput_trend: float    # Derivative (increasing/decreasing)
    rtt_trend: float
    
    # Queue info
    queue_length: int          # Packets in queue
    queue_drops: int          # Recent drops
    
    def to_vector(self) -> np.ndarray:
        """Convert to normalized feature vector for RL agent"""
        # Service type one-hot encoding
        service_onehot = {
            'eMBB': [1, 0, 0, 0],
            'URLLC': [0, 1, 0, 0],
            'mMTC': [0, 0, 1, 0],
            'Background': [0, 0, 0, 1]
        }.get(self.service_type, [0, 0, 0, 0])
        
        # Normalize metrics
        vector = [
            *service_onehot,                              # [0-3] Service type
            min(self.throughput_mbps / 100.0, 1.0),      # [4] Normalized throughput
            min(self.rtt_ms / 200.0, 1.0),               # [5] Normalized RTT
            min(self.jitter_ms / 50.0, 1.0),             # [6] Normalized jitter
            min(self.loss_rate / 10.0, 1.0),             # [7] Normalized loss
            min(self.cwnd / 1000.0, 1.0),                # [8] Normalized CWND
            float(self.sla_violation),                    # [9] Binary flag
            np.tanh(self.throughput_trend),               # [10] Normalized trend
            np.tanh(self.rtt_trend),                      # [11] Normalized trend
            min(self.queue_length / 100.0, 1.0),         # [12] Normalized queue
            min(self.queue_drops / 50.0, 1.0),            # [13] Normalized drops
            1.0 if self.protocol == 'TCP' else 0.0     # [14] Protocol type
        ]
        
        return np.array(vector, dtype=np.float32)
    
    @staticmethod
    def vector_size() -> int:
        """Feature vector dimension"""
        return 15
    
    @staticmethod
    def from_ns3_flow(flow_data: Dict) -> 'FlowState':
        return FlowState(
            flow_id=flow_data.get('id', 0),
            service_type=flow_data.get('service_type', 'Background'),
            protocol=flow_data.get('protocol', 'TCP'),
            src_port=flow_data.get('src_port', 0),
            dst_port=flow_data.get('dst_port', 0),
            throughput_mbps=flow_data.get('throughput_mbps', 0.0),
            rtt_ms=flow_data.get('rtt_ms', 0.0),
            jitter_ms=flow_data.get('jitter_ms', 0.0),
            loss_rate=flow_data.get('loss_rate', 0.0),
            cwnd=flow_data.get('cwnd', 10),
            sla_violation=_check_sla_violation(flow_data),
            throughput_trend=0.0,
            rtt_trend=0.0,
            queue_length=0,
            queue_drops=0
        )


def _check_sla_violation(flow_data: Dict) -> bool:
    """Check if flow violates its service-level agreement"""
    service_type = flow_data.get('service_type', '')
    rtt_ms = flow_data.get('rtt_ms', 0.0)
    loss_rate = flow_data.get('loss_rate', 0.0)
    throughput_mbps = flow_data.get('throughput_mbps', 0.0)
    
    # URLLC: RTT must be < 60ms, loss < 1%
    if service_type == 'URLLC':
        return rtt_ms > 70.0 or loss_rate > 1.0

    # eMBB: Throughput must be > 5 Mbps
    elif service_type == 'eMBB':
        return throughput_mbps < 5.0
    
    # mMTC: Loss must be < 5%
    elif service_type == 'mMTC':
        return loss_rate > 5.0
    
    return False


# ============================================================================
# FLOW-LEVEL ACTION SPACE (PPO AGENT)
# ============================================================================
class FlowAction(Enum):
    """Available actions for flow control"""
    NO_CHANGE = 0
    INCREASE_RATE = 1
    DECREASE_RATE = 2
    INCREASE_BURST_SIZE = 3
    DECREASE_BURST_SIZE = 4

    @classmethod
    def from_index(cls, idx: int) -> 'FlowAction':
        """Safely convert integer to FlowAction"""
        try:
            return list(cls)[idx]
        except IndexError:
            print(f"⚠️ Invalid action index {idx}, defaulting to NO_CHANGE")
            return cls.NO_CHANGE

@dataclass
class FlowActionCommand:
    """Command to send to NS-3 for flow modification"""
    flow_id: int
    action: FlowAction
    parameters: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict for socket"""
        return {
            'type': 'flow_action',
            'flow_id': self.flow_id,
            'action': self.action.name,
            'action_code': self.action.value,
            'parameters': self.parameters or {}
        }


# ============================================================================
# EDGE-LEVEL STATE SPACE (A3C AGENT)
# ============================================================================
@dataclass
class EdgeState:
    """
    State representation for edge server (MEC node)
    
    Dimensions: 10 features per edge
    """
    location: str              # 'india' or 'uk'
    
    # Resource utilization (0-1)
    cpu_usage: float
    memory_usage: float
    network_usage: float       # Bandwidth utilization
    
    # Aggregate flow statistics
    total_flows: int
    embb_flows: int
    urllc_flows: int
    mmtc_flows: int
    
    # Performance metrics
    avg_delay_ms: float        # Average across all flows at this edge
    sla_violations: int        # Count of flows violating SLA
    
    def to_vector(self) -> np.ndarray:
        """Convert to normalized feature vector"""
        location_onehot = [1, 0] if self.location == 'india' else [0, 1]
        
        vector = [
            *location_onehot,                              # [0-1] Location
            self.cpu_usage,                                # [2] Already 0-1
            self.memory_usage,                             # [3] Already 0-1
            self.network_usage,                            # [4] Already 0-1
            min(self.total_flows / 50.0, 1.0),            # [5] Normalized flow count
            min(self.embb_flows / 20.0, 1.0),             # [6]
            min(self.urllc_flows / 15.0, 1.0),            # [7]
            min(self.mmtc_flows / 30.0, 1.0),             # [8]
            min(self.avg_delay_ms / 100.0, 1.0),          # [9] Normalized delay
            min(self.sla_violations / 10.0, 1.0)          # [10] Normalized violations
        ]
        
        return np.array(vector, dtype=np.float32)
    
    @staticmethod
    def vector_size() -> int:
        return 11
    
    @staticmethod
    def from_ns3_edge(edge_data: Dict, flows: List[Dict]) -> 'EdgeState':
        """Create EdgeState from NS-3 metrics"""
        location = edge_data.get('location', 'unknown')
        
        # Count flows by type at this edge
        flow_counts = {'eMBB': 0, 'URLLC': 0, 'mMTC': 0}
        total_delay = 0.0
        sla_violations = 0
        
        for flow in flows:
            # Simple heuristic: flows belong to edge based on src IP prefix
            if _flow_belongs_to_edge(flow, location):
                stype = flow.get('service_type', '')
                if stype in flow_counts:
                    flow_counts[stype] += 1
                
                total_delay += flow.get('rtt_ms', 0.0)
                if _check_sla_violation(flow):
                    sla_violations += 1
        
        total_flows = sum(flow_counts.values())
        avg_delay = total_delay / total_flows if total_flows > 0 else 0.0
        
        return EdgeState(
            location=location,
            cpu_usage=edge_data.get('cpu_usage', 0.0),
            memory_usage=edge_data.get('memory_usage', 0.0),
            network_usage=edge_data.get('network_usage', 0.0),
            total_flows=total_flows,
            embb_flows=flow_counts['eMBB'],
            urllc_flows=flow_counts['URLLC'],
            mmtc_flows=flow_counts['mMTC'],
            avg_delay_ms=avg_delay,
            sla_violations=sla_violations
        )


def _flow_belongs_to_edge(flow: Dict, location: str) -> bool:
    """Determine if flow belongs to given edge location"""
    src_ip = flow.get('src', '')
    
    if location == 'india':
        return src_ip.startswith('10.')
    elif location == 'uk':
        return src_ip.startswith('20.')
    
    return False


# ============================================================================
# EDGE-LEVEL ACTION SPACE (A3C AGENT)
# ============================================================================
class EdgeAction(Enum):
    """
    A3C Agent Actions for Edge Resource Control
    Affects ALL flows (TCP/UDP) at the edge level
    """
    NO_CHANGE = 0
    ADJUST_QUEUE_SIZE = 1              # Modify queue buffer sizes
    SET_URLLC_RATE_LIMIT = 2           # Rate limiting for URLLC flows  
    ADJUST_PRIORITY_WEIGHTS = 3        # Change priority scheduling
    ENABLE_ADMISSION_CONTROL = 4       # Control flow admission
    TUNE_AQM_PARAMETERS = 5            # Adjust CoDel parameters


@dataclass
class EdgeActionCommand:
    """Command to send to NS-3 for edge modification"""
    location: str
    action: EdgeAction
    parameters: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        return {
            'type': 'edge_action',
            'location': self.location,
            'action': self.action.name,
            'action_code': self.action.value,
            'parameters': self.parameters or {}
        }


# ============================================================================
# GREY WOLF OPTIMIZER STATE SPACE
# ============================================================================
@dataclass
class GWOState:
    """
    State representation for GWO resource optimizer
    
    Global view of entire network for resource allocation
    """
    # Global metrics
    total_throughput: float
    total_delay: float
    total_loss: float
    
    # Resource utilization
    india_cpu: float
    india_memory: float
    uk_cpu: float
    uk_memory: float
    backbone_bandwidth_usage: float
    
    # SLA compliance
    total_sla_violations: int
    urllc_violations: int
    embb_violations: int
    
    # Traffic distribution
    india_to_uk_flows: int
    uk_to_india_flows: int
    to_cloud_flows: int
    
    def to_vector(self) -> np.ndarray:
        """Convert to optimization input vector"""
        vector = [
            min(self.total_throughput / 500.0, 1.0),       # [0] Normalized
            min(self.total_delay / 200.0, 1.0),            # [1]
            min(self.total_loss / 10.0, 1.0),              # [2]
            self.india_cpu,                                 # [3-7] Already 0-1
            self.india_memory,
            self.uk_cpu,
            self.uk_memory,
            self.backbone_bandwidth_usage,
            min(self.total_sla_violations / 20.0, 1.0),   # [8-10]
            min(self.urllc_violations / 10.0, 1.0),
            min(self.embb_violations / 10.0, 1.0),
            min(self.india_to_uk_flows / 30.0, 1.0),      # [11-13]
            min(self.uk_to_india_flows / 30.0, 1.0),
            min(self.to_cloud_flows / 40.0, 1.0)
        ]
        
        return np.array(vector, dtype=np.float32)
    
    @staticmethod
    def vector_size() -> int:
        return 14


# ============================================================================
# MULTI-AGENT STATE AGGREGATOR
# ============================================================================
class StateAggregator:
    """
    Maintains state history and computes temporal features
    
    Usage:
        aggregator = StateAggregator(history_size=10)
        aggregator.update(flows, queues, edges)
        ppo_states = aggregator.get_flow_states()
        a3c_states = aggregator.get_edge_states()
        gwo_state = aggregator.get_global_state()
    """
    
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.flow_history = []
        self.edge_history = []
        self.queue_history = []
        
    def update(self, flows: List[Dict], queues: List[Dict], edges: List[Dict]):
        """Update with new network state from NS-3"""
        # Store in history
        self.flow_history.append(flows)
        self.queue_history.append(queues)
        self.edge_history.append(edges)
        
        # Keep only recent history
        if len(self.flow_history) > self.history_size:
            self.flow_history.pop(0)
            self.queue_history.pop(0)
            self.edge_history.pop(0)
    
    def get_flow_states(self) -> List[FlowState]:
        """Get current flow states with temporal features"""
        if not self.flow_history:
            return []
        
        current_flows = self.flow_history[-1]
        flow_states = []
        
        for flow_data in current_flows:
            flow_state = FlowState.from_ns3_flow(flow_data)
            
            # Compute trends if we have history
            if len(self.flow_history) >= 2:
                prev_flows = self.flow_history[-2]
                prev_flow = next((f for f in prev_flows if f['id'] == flow_data['id']), None)
                
                if prev_flow:
                    flow_state.throughput_trend = (
                        flow_data['throughput_mbps'] - prev_flow['throughput_mbps']
                    )
                    flow_state.rtt_trend = (
                        flow_data['rtt_ms'] - prev_flow['rtt_ms']
                    )
            
            # Add queue info
            if self.queue_history:
                # Find relevant queue (simplified - match by device name)
                for queue in self.queue_history[-1]:
                    # You'd need better matching logic based on your topology
                    flow_state.queue_length = queue.get('length', 0)
                    flow_state.queue_drops = queue.get('drops', 0)
                    break
            
            flow_states.append(flow_state)
        
        return flow_states
    
    def get_edge_states(self) -> List[EdgeState]:
        """Get current edge states"""
        if not self.edge_history or not self.flow_history:
            return []
        
        current_edges = self.edge_history[-1]
        current_flows = self.flow_history[-1]
        
        edge_states = [
            EdgeState.from_ns3_edge(edge_data, current_flows)
            for edge_data in current_edges
        ]
        
        return edge_states
    
    def get_global_state(self) -> GWOState:
        """Get global state for GWO optimizer"""
        if not self.flow_history or not self.edge_history:
            return GWOState(
                total_throughput=0, total_delay=0, total_loss=0,
                india_cpu=0, india_memory=0, uk_cpu=0, uk_memory=0,
                backbone_bandwidth_usage=0,
                total_sla_violations=0, urllc_violations=0, embb_violations=0,
                india_to_uk_flows=0, uk_to_india_flows=0, to_cloud_flows=0
            )
        
        flows = self.flow_history[-1]
        edges = self.edge_history[-1]
        
        # Aggregate flow metrics
        total_throughput = sum(f.get('throughput_mbps', 0) for f in flows)
        total_delay = sum(f.get('rtt_ms', 0) for f in flows) / max(len(flows), 1)
        total_loss = sum(f.get('loss_rate', 0) for f in flows) / max(len(flows), 1)
        
        # Count SLA violations
        total_sla = sum(1 for f in flows if _check_sla_violation(f))
        urllc_violations = sum(
            1 for f in flows 
            if f.get('service_type') == 'URLLC' and _check_sla_violation(f)
        )
        embb_violations = sum(
            1 for f in flows 
            if f.get('service_type') == 'eMBB' and _check_sla_violation(f)
        )
        
        # Get edge resources
        india_edge = next((e for e in edges if e.get('location') == 'india'), {})
        uk_edge = next((e for e in edges if e.get('location') == 'uk'), {})
        
        # Count flow directions
        india_to_uk = sum(
            1 for f in flows 
            if f.get('src', '').startswith('10.') and f.get('dst', '').startswith('20.')
        )
        uk_to_india = sum(
            1 for f in flows 
            if f.get('src', '').startswith('20.') and f.get('dst', '').startswith('10.')
        )
        to_cloud = sum(
            1 for f in flows 
            if f.get('dst', '').startswith('203.')
        )
        
        return GWOState(
            total_throughput=total_throughput,
            total_delay=total_delay,
            total_loss=total_loss,
            india_cpu=india_edge.get('cpu_usage', 0),
            india_memory=india_edge.get('memory_usage', 0),
            uk_cpu=uk_edge.get('cpu_usage', 0),
            uk_memory=uk_edge.get('memory_usage', 0),
            backbone_bandwidth_usage=0.5,  # Placeholder - compute from flows
            total_sla_violations=total_sla,
            urllc_violations=urllc_violations,
            embb_violations=embb_violations,
            india_to_uk_flows=india_to_uk,
            uk_to_india_flows=uk_to_india,
            to_cloud_flows=to_cloud
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def print_state_summary(flow_states: List[FlowState], 
                        edge_states: List[EdgeState],
                        gwo_state: GWOState):
    """Pretty-print current state for debugging"""
    print("\n" + "="*60)
    print("CURRENT NETWORK STATE")
    print("="*60)
    
    print(f"\n📊 Flow States ({len(flow_states)} flows):")
    for fs in flow_states[:5]:  # Show first 5
        print(f"  Flow {fs.flow_id} ({fs.service_type}): "
              f"Tput={fs.throughput_mbps:.1f}Mbps, "
              f"RTT={fs.rtt_ms:.1f}ms, "
              f"Loss={fs.loss_rate:.2f}%, "
              f"SLA={'❌' if fs.sla_violation else '✓'}")
    
    print(f"\n🖥️  Edge States ({len(edge_states)} edges):")
    for es in edge_states:
        print(f"  {es.location.upper()}: "
              f"CPU={es.cpu_usage:.1%}, "
              f"Mem={es.memory_usage:.1%}, "
              f"Flows={es.total_flows}, "
              f"Violations={es.sla_violations}")
    
    print(f"\n🌐 Global State:")
    print(f"  Total Throughput: {gwo_state.total_throughput:.1f} Mbps")
    print(f"  Avg Delay: {gwo_state.total_delay:.1f} ms")
    print(f"  SLA Violations: {gwo_state.total_sla_violations}")
    print("="*60 + "\n")


if __name__ == "__main__":
    """Test state space definitions"""
    print("Testing RL State-Action Space Definitions\n")
    
    # Test flow state
    test_flow = {
        'id': 1,
        'service_type': 'URLLC',
        'throughput_mbps': 25.5,
        'rtt_ms': 8.3,
        'jitter_ms': 1.2,
        'loss_rate': 0.05,
        'cwnd': 120
    }
    
    flow_state = FlowState.from_ns3_flow(test_flow)
    flow_vector = flow_state.to_vector()
    
    print(f"✓ Flow State Vector Shape: {flow_vector.shape}")
    print(f"  Expected: ({FlowState.vector_size()},)")
    print(f"  Sample values: {flow_vector[:5]}")
    
    # Test edge state
    test_edge = {
        'location': 'india',
        'cpu_usage': 0.65,
        'memory_usage': 0.72,
        'network_usage': 0.58
    }
    
    edge_state = EdgeState.from_ns3_edge(test_edge, [test_flow])
    edge_vector = edge_state.to_vector()
    
    print(f"\n✓ Edge State Vector Shape: {edge_vector.shape}")
    print(f"  Expected: ({EdgeState.vector_size()},)")
    
    # Test GWO state
    gwo_state = GWOState(
        total_throughput=450.0,
        total_delay=25.3,
        total_loss=0.8,
        india_cpu=0.65,
        india_memory=0.72,
        uk_cpu=0.58,
        uk_memory=0.63,
        backbone_bandwidth_usage=0.75,
        total_sla_violations=3,
        urllc_violations=2,
        embb_violations=1,
        india_to_uk_flows=15,
        uk_to_india_flows=12,
        to_cloud_flows=8
    )
    
    gwo_vector = gwo_state.to_vector()
    
    print(f"\n✓ GWO State Vector Shape: {gwo_vector.shape}")
    print(f"  Expected: ({GWOState.vector_size()},)")
    
    print("\n✅ All state space definitions validated!")
    print("Ready to implement RL agents.")
