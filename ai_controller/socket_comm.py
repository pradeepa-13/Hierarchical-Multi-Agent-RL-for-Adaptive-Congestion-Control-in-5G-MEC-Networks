#!/usr/bin/env python3
"""
socket_comm.py
Python-side socket communication module for NS-3 RL integration
Place in: ai_controller/socket_comm.py

Handles bidirectional UDP communication with NS-3 simulation.
Designed for multi-agent RL architecture (PPO, A3C, GWO).
"""

import socket
import json
import threading
import queue
import time
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict


@dataclass
class SocketConfig:
    """Configuration for socket communication"""
    host: str = "127.0.0.1"
    port: int = 5000
    buffer_size: int = 16384
    timeout: float = 1.0
    max_queue_size: int = 1000
    verbose: bool = False


@dataclass
class NetworkState:
    """Structured representation of network state from NS-3"""
    timestamp: float
    flows: list
    queues: list
    edges: list = None
    raw_json: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'flows': self.flows,
            'queues': self.queues,
            'edges': self.edges or []
        }


@dataclass
class RLAction:
    """Structured representation of RL actions to send to NS-3"""
    flow_actions: list = None
    queue_actions: list = None
    edge_actions: list = None
    gwo_actions: dict = None 
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.flow_actions is None:
            self.flow_actions = []
        if self.queue_actions is None:
            self.queue_actions = []
        if self.edge_actions is None:
            self.edge_actions = []
        if self.gwo_actions is None:
            self.gwo_actions = {} 
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        payload = {
            'flow_actions': self.flow_actions,
            'queue_actions': self.queue_actions,
            'edge_actions': self.edge_actions,
            'timestamp': self.timestamp
        }
        
        # ✅ Only include gwo_actions if not empty
        if self.gwo_actions:
            payload['gwo_actions'] = self.gwo_actions
        
        return json.dumps(payload)


class SocketCommunicator:
    """
    Robust socket communicator for NS-3 <-> Python AI communication.
    
    Features:
    - Non-blocking receive with threading
    - Queue-based message handling
    - Automatic reconnection
    - Statistics tracking
    - Callback system for state updates
    
    Usage:
        comm = SocketCommunicator(config)
        comm.start()
        comm.register_state_callback(my_rl_agent.process_state)
        # ... simulation runs ...
        comm.stop()
    """
    
    def __init__(self, config: SocketConfig = None):
        self.config = config or SocketConfig()
        # Add threading lock
        self._stats_lock = threading.Lock()
        self._action_lock = threading.Lock()
        
        # Socket setup
        self.socket = None
        self.running = False
        self.connected = False
        
        # Threading
        self.receive_thread = None
        self.state_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.action_queue = queue.Queue(maxsize=self.config.max_queue_size)
        self.pending_actions = {}  # track sent actions
        self.action_acks = {}      # track acknowledged actions

        # Callbacks
        self.state_callbacks = []
        self.handshake_callback = None
        self.episode_start_callback = None
        self.episode_end_callback = None
        
        # Statistics
        self.stats = {
            'start_time': time.time(),
            'states_received': 0,
            'actions_sent': 0,
            'bytes_received': 0,
            'bytes_sent': 0,
            'errors': 0,
            'last_message_time': 0
        }
        
        # Logging
        self.logger = self._setup_logger()
        self._stats_lock = threading.Lock()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging with appropriate format"""
        logger = logging.getLogger('SocketCommunicator')
        
        if self.config.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start(self) -> bool:
        """
        Start the socket communicator
        Returns True if successful
        """
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.config.host, self.config.port))
            self.socket.settimeout(self.config.timeout)
            
            self.logger.info(f"✓ Socket bound to {self.config.host}:{self.config.port}")
            
            # Start receive thread
            self.running = True
            self.stats['start_time'] = time.time()
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
            self.logger.info("✓ Receive thread started")
            self.logger.info("✓ Waiting for NS-3 connection...")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start socket: {e}")
            self.stats['errors'] += 1
            return False
    
    def stop(self):
        """Stop the socket communicator gracefully"""
        self.logger.info("Stopping socket communicator...")
        
        self.running = False
        
        if self.receive_thread:
            self.receive_thread.join(timeout=2.0)
        
        if self.socket:
            self.socket.close()
        
        self.print_statistics()
        self.logger.info("✓ Socket communicator stopped")
    
    def register_state_callback(self, callback: Callable[[NetworkState], RLAction]):
        """
        Register a callback function to process incoming states
        Callback should accept NetworkState and return RLAction
        """
        self.state_callbacks.append(callback)
        self.logger.info(f"✓ Registered state callback: {callback.__name__}")
    
    def register_handshake_callback(self, callback: Callable[[Dict], None]):
        """Register callback for handshake messages"""
        self.handshake_callback = callback
    
    def register_episode_callbacks(self, 
                                   start_callback: Callable[[int], None],
                                   end_callback: Callable[[Dict], None]):
        """Register callbacks for episode start/end signals"""
        self.episode_start_callback = start_callback
        self.episode_end_callback = end_callback

    def _handle_action_ack(self, ack_data: dict):
        """Handle acknowledgment of action execution from NS-3"""
        action_id = ack_data.get('action_id')
        if action_id in self.pending_actions:
            self.pending_actions[action_id]['acknowledged'] = True
            self.action_acks[action_id] = ack_data
            
            # ✅ NEW: Log success/failure
            success = ack_data.get('success', False)
            details = ack_data.get('details', '')
            
            if success:
                self.logger.debug(f"✅ Action {action_id} executed successfully: {details}")
            else:
                self.logger.warning(f"❌ Action {action_id} failed: {details}")
            
            # Clean up old acknowledged actions periodically
            self._cleanup_old_actions()
        else:
            self.logger.warning(f"⚠️  Received ACK for unknown action: {action_id}")
    
    def send_action(self, action: RLAction, addr: tuple) -> bool:
        """
        Send action to NS-3
        Returns True if successful
        """
        try:
            # Generate unique ID for this action
            action_id = str(uuid.uuid4())
            
            # Track the pending action
            self.pending_actions[action_id] = {
                'timestamp': time.time(),
                'data': action,
                'acknowledged': False,
                'addr': addr
            }
            
            # Add action_id to the action before sending
            action_dict = json.loads(action.to_json())
            action_dict['action_id'] = action_id
            action_json = json.dumps(action_dict)
            
            sent = self.socket.sendto(action_json.encode('utf-8'), addr)
            
            self.stats['actions_sent'] += 1
            self.stats['bytes_sent'] += sent
            
            self.logger.debug(f"Action sent: {sent} bytes to {addr} (id: {action_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send action: {e}")
            self.stats['errors'] += 1
            return False
    
    def _receive_loop(self):
        """Main receive loop (runs in separate thread)"""
        self.logger.info("Receive loop started")
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(self.config.buffer_size)
                
                if not self.connected:
                    self.connected = True
                    self.logger.info(f"✓ Connected to NS-3 at {addr}")
                
                self.stats['last_message_time'] = time.time()
                self.stats['states_received'] += 1
                self.stats['bytes_received'] += len(data)
                
                # Parse message
                msg = data.decode('utf-8').strip()
                self._process_message(msg, addr)
                
            except socket.timeout:
                # Normal timeout, continue
                continue
            
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    self.logger.error(f"Receive error: {e}")
                    self.stats['errors'] += 1
                    time.sleep(0.1)  # Brief pause before retry
    
    def _process_message(self, msg: str, addr: tuple):
        """Process incoming message from NS-3"""
        try:
            data = json.loads(msg)
            msg_type = data.get('type', 'state')

            if msg_type == 'action_ack':
                self._handle_action_ack(data)
            
            if msg_type == 'handshake':
                self._handle_handshake(data, addr)
            
            elif msg_type == 'episode_start':
                self._handle_episode_start(data)
            
            elif msg_type == 'episode_end':
                self._handle_episode_end(data)
            
            elif msg_type == 'goodbye':
                self._handle_goodbye(data)
            
            else:
                # Regular state update
                self._handle_state(data, addr)
        
        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON received: {e}")
            self.logger.debug(f"Raw message: {msg[:200]}...")
            self.stats['errors'] += 1
        
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            self.stats['errors'] += 1
    
    # def _handle_action_ack(self, ack_data: dict):
    #     """Handle acknowledgment of action execution from NS-3"""
    #     action_id = ack_data.get('action_id')
    #     if action_id in self.pending_actions:
    #         self.pending_actions[action_id]['acknowledged'] = True
    #         self.action_acks[action_id] = ack_data
    #         self.logger.debug(f"Action {action_id} acknowledged: {ack_data.get('success', False)}")
            
    #         # Clean up old acknowledged actions periodically
    #         self._cleanup_old_actions()

    def _cleanup_old_actions(self, max_age: float = 300):
        """Remove old acknowledged actions (older than max_age seconds)"""
        current_time = time.time()
        for action_id in list(self.pending_actions.keys()):
            action = self.pending_actions[action_id]
            if action['acknowledged'] and (current_time - action['timestamp']) > max_age:
                del self.pending_actions[action_id]
                if action_id in self.action_acks:
                    del self.action_acks[action_id]

    def _handle_handshake(self, data: Dict, addr: tuple):
        """Handle handshake message from NS-3"""
        self.logger.info("═" * 60)
        self.logger.info("HANDSHAKE RECEIVED FROM NS-3")
        self.logger.info("═" * 60)
        self.logger.info(f"  Version: {data.get('version', 'unknown')}")
        self.logger.info(f"  Message: {data.get('message', '')}")
        self.logger.info(f"  Agents: {', '.join(data.get('agents', []))}")
        self.logger.info(f"  Sim Time: {data.get('timestamp', 0):.2f}s")
        self.logger.info("═" * 60)
        
        if self.handshake_callback:
            self.handshake_callback(data)
        
        # Send acknowledgment
        ack = RLAction()
        ack.flow_actions = [{'type': 'handshake_ack', 'status': 'ready'}]
        self.send_action(ack, addr)
    
    def _handle_episode_start(self, data: Dict):
        """Handle episode start signal"""
        episode_id = data.get('episode_id', 0)
        self.logger.info(f"━━━ Episode {episode_id} Started ━━━")
        
        if self.episode_start_callback:
            self.episode_start_callback(episode_id)
    
    def _handle_episode_end(self, data: Dict):
        """Handle episode end signal from NS-3"""
        reason = data.get('reason', 'unknown')
        sim_time = data.get('simTime', 0.0)  # ← Get actual sim time from NS-3
        
        self.logger.info(f"╔═══ Episode Ended at t={sim_time:.1f}s: {reason} ═══╗")
        
        if self.episode_end_callback:
            self.episode_end_callback(data)
    
    def _handle_goodbye(self, data: Dict):
        """Handle goodbye message from NS-3"""
        self.logger.info("═" * 60)
        self.logger.info("GOODBYE MESSAGE FROM NS-3")
        self.logger.info("═" * 60)
        ns3_stats = data.get('stats', {})
        self.logger.info(f"  NS-3 messages sent: {ns3_stats.get('messages_sent', 0)}")
        self.logger.info(f"  NS-3 messages received: {ns3_stats.get('messages_received', 0)}")
        self.logger.info(f"  NS-3 errors: {ns3_stats.get('errors', 0)}")
        self.logger.info("═" * 60)
    
    def _handle_state(self, data: Dict, addr: tuple):
        """Handle state update and invoke callbacks"""
        try:
            # Create NetworkState object
            state = NetworkState(
                timestamp=data.get('timestamp', 0.0),
                flows=data.get('flows', []),
                queues=data.get('queues', []),
                edges=data.get('edges', []),
                raw_json=json.dumps(data)
            )
            
            self.logger.debug(f"State received: {len(state.flows)} flows, "
                            f"{len(state.queues)} queues at t={state.timestamp:.2f}s")
            
            # ✅ NEW CODE (ALWAYS SEND):
            for callback in self.state_callbacks:
                try:
                    action = callback(state)
                    
                    # ✅ Always send action (even if empty) to acknowledge receipt
                    if action is None:
                        # Callback returned None - create empty action
                        action = RLAction()
                        action.flow_actions = [{'type': 'no_action', 'reason': 'callback_returned_none'}]
                    
                    if not isinstance(action, RLAction):
                        self.logger.error(f"Callback returned invalid type: {type(action)}")
                        action = RLAction()
                        action.flow_actions = [{'type': 'error', 'reason': 'invalid_action_type'}]
                    
                    # ✅ Send action (even if empty)
                    success = self.send_action(action, addr)
                    
                    if success:
                        self.logger.debug(f"✓ Action sent: {len(action.flow_actions)} flow actions, "
                                        f"{len(action.edge_actions)} edge actions")
                    else:
                        self.logger.warning(f"⚠️  Failed to send action!")
                        
                except Exception as e:
                    self.logger.error(f"Callback error ({callback.__name__}): {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # ✅ Send error acknowledgment
                    error_action = RLAction()
                    error_action.flow_actions = [{'type': 'error', 'reason': str(e)}]
                    self.send_action(error_action, addr)
            
        except Exception as e:
            self.logger.error(f"Error handling state: {e}")
            self.stats['errors'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:

        """Get current statistics"""
        # Remove the super() call and return stats directly
        with self._stats_lock:
            return {
                'states_received': self.stats['states_received'],
                'actions_sent': self.stats['actions_sent'],
                'bytes_received': self.stats['bytes_received'],
                'bytes_sent': self.stats['bytes_sent'],
                'errors': self.stats['errors'],
                'uptime': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0,
                'last_message_time': self.stats['last_message_time']
            }
            # # Calculate runtime
            # if stats['start_time'] > 0:
            #     stats['runtime_seconds'] = time.time() - stats['start_time']
            # else:
            #     stats['runtime_seconds'] = 0.0

            # # Add action acknowledgment stats if tracking enabled
            # if hasattr(self, 'pending_actions'):
            #     total_actions = len(self.pending_actions)
            #     acked_actions = sum(1 for a in self.pending_actions.values() if a['acknowledged'])
            #     stats.update({
            #         'actions_pending': total_actions - acked_actions,
            #         'actions_acknowledged': acked_actions,
            #         'ack_rate': acked_actions / total_actions if total_actions > 0 else 0
            #     })
            
            # return stats
    
    def print_statistics(self):
        """Print current statistics to console"""
        try:
            stats = self.get_statistics()
            print("\n📊 Socket Communication Statistics:")
            print(f"States Received: {stats['states_received']}")
            print(f"Actions Sent: {stats['actions_sent']}")
            print(f"Bytes Received: {stats['bytes_received']/1024:.2f} KB")
            print(f"Bytes Sent: {stats['bytes_sent']/1024:.2f} KB")
            print(f"Errors: {stats['errors']}")
            print(f"Uptime: {stats['uptime']:.1f} seconds")
            if stats['last_message_time'] > 0:
                print(f"Last Message: {time.strftime('%H:%M:%S', time.localtime(stats['last_message_time']))}")
            print("━" * 50)
        except Exception as e:
            print(f"Error printing statistics: {e}")

def create_test_callback():
    """
    Create a simple test callback for development
    This will be replaced by actual RL agents (PPO, A3C, GWO)
    """
    def test_callback(state: NetworkState) -> RLAction:
        """Simple rule-based action for testing"""
        action = RLAction()
        
        # Example rule: if any URLLC flow has high delay, boost priority
        for flow in state.flows:
            if flow.get('service_type') == 'URLLC' and flow.get('rtt_ms', 0) > 70:
                action.flow_actions.append({
                    'flow_id': flow.get('id'),
                    'priority': 'high',
                    'reason': 'urllc_sla_violation'
                })
        
        # Example rule: if queue too full, reduce limit
        for q in state.queues:
            if q.get('length', 0) > 50:
                action.queue_actions.append({
                    'device': q.get('device'),
                    'limit_packets': 70,
                    'reason': 'congestion_control'
                })
        
        return action
    
    return test_callback


if __name__ == "__main__":
    """Test the socket communicator"""
    print("\n" + "═" * 60)
    print("     PYTHON RL SOCKET COMMUNICATOR TEST")
    print("═" * 60 + "\n")
    
    config = SocketConfig(verbose=True)
    comm = SocketCommunicator(config)
    
    # Register test callback
    comm.register_state_callback(create_test_callback())
    
    if comm.start():
        print("\n✓ Socket started successfully")
        print("✓ Waiting for NS-3 connection...")
        print("  (Run NS-3 simulation with RL socket enabled)\n")
        
        try:
            # Keep running until Ctrl+C
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n✓ Test interrupted by user")
    
    else:
        print("\n❌ Failed to start socket")
    
    comm.stop()
