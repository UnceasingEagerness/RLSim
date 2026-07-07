# RLSim: DeepSet Multi-Agent RL Simulator

RLSim is a high-performance, fully decentralized Multi-Agent Reinforcement Learning (MARL) simulator for heterogeneous maritime swarms (Surface Vessels and Autonomous Underwater Vehicles). It features highly optimized Numba JIT-compiled physics and complies with the PettingZoo Parallel Environment API.

## Features
- **Heterogeneous Agents**: Supports both 3-DOF USVs (Surface Vessels) and 6-DOF AUVs (Torpedo-shaped).
- **Blazing Fast Physics**: Custom Runge-Kutta 4th Order (RK4) integration accelerated via Numba JIT.
- **DeepSet MARL Architecture**: Shared decentralized Soft-Actor Critic (SAC) policy utilizing Deep Sets to process an arbitrary number of neighbors and obstacles.
- **Potential-Based Reward Shaping (PBRS)**: Mathematically airtight dense reward signals for optimal navigation.
- **Global Command Center**: An interactive PyGame visualizer with zooming, panning, and depth rendering for underwater agents.

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

- `envs/`: Contains the PettingZoo compliant `MultiAgentNavEnv`, physical dynamics models (`usv_dynamics.py`, `auv_dynamics.py`), and Numba JIT utilities.
- `algorithms/`: Contains the `cleanrl_sac.py` training loop and the prioritized experience replay buffers.
- `visualization/`: Contains the `visualiser.py` Global Command Center and UI rendering logic.

## Usage

To launch a training run with 2 parallel environments and 6 agents per environment:

```bash
python algorithms/cleanrl_sac.py --num_worlds 2 --num_agents 6 --total_timesteps 40000
```

To enable the Global Visualizer during training:

```bash
python algorithms/cleanrl_sac.py --show_visualizer True
```
