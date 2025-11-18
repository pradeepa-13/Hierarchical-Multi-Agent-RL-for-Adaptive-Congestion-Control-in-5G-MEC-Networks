# 5G MEC Multi-Agent RL for Congestion Control

Deep reinforcement learning approach for intelligent congestion management in 5G Mobile Edge Computing networks.

## Overview

This project implements a hierarchical multi-agent RL system that dynamically optimizes network performance across different service types (eMBB, URLLC, mMTC) in realistic 5G MEC environments.

**Key Components:**
- **PPO Agent**: Flow-level traffic control
- **A3C Agent**: Edge resource management  
- **Grey Wolf Optimizer**: Global resource allocation
- **NS-3 Simulation**: Realistic network topology with randomized traffic patterns

## Architecture

```
NS-3 Simulation (C++) ←→ UDP Socket ←→ AI Controller (Python)
                                           ├── PPO (Flow Control)
                                           ├── A3C (Edge Management)
                                           └── GWO (Resource Optimization)
```

## Features

- Real-time socket communication between NS-3 and RL agents
- Curriculum learning with progressive difficulty
- Service-aware QoS optimization (URLLC latency, eMBB throughput, mMTC scalability)
- Comparative analysis with TCP variants (BBR, Cubic, Vegas, Reno)
- Dynamic traffic generation to prevent pattern memorization

## Requirements

- NS-3.45 with Python bindings
- PyTorch
- NumPy, RapidJSON

## Quick Start

**1. Build NS-3 simulation:**
```bash
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

**2. Run training:**
```bash
# Terminal 1: Start AI controller
cd ai_controller
python main.py --preset dev --curriculum

# Terminal 2: Run NS-3 simulation
./ns3 run 'scratch/mec_full_simulation --enableRL=true --config=sim_config_0.json'
```

**3. Visualize results:**
```bash
python visualize_metrics.py
```

## Results

Performance metrics tracked:
- Throughput (Mbps)
- End-to-end delay (ms)
- Packet loss rate (%)
- SLA compliance (%)
- Fairness index

