# Addon Suggestions from Paper Analysis
**Paper:** Distributed Pursuit-Evasion Game of Limited Perception USV Swarm Based on Multiagent Proximal Policy Optimization

---

## What the Paper Does vs What We Have

| Feature | Paper | Ours |
|---|---|---|
| Algorithm | MAPPO (on-policy) | SAC (off-policy) |
| Task | Pursuit-Evasion Game | Navigation + Collision Avoidance |
| Sensors | Limited perception radius | 64-beam LiDAR + entity tracking |
| Dynamics | USV 3-DOF model | USV 3-DOF model (same) |
| Entity Encoding | Mean Pool | **EntityMaxEncoder (Max Pool + 70m filter)** |
| Temporal | None | LSTM + 10-frame stacking |
| Obstacle Type | Static only | Static only |

---

## Addons You Can Pick From

---

### 🔴 ADDON 1: Pursuit-Evasion Game Mode
**What it is:** Add a second role — "Evader" USVs — that actively try to escape the "Pursuer" swarm. Pursuers try to surround and capture the evader within a radius. Gives the simulation a competitive game-theoretic dimension.

**Difficulty:** High  
**Training:** You'd need to retrain both pursuer and evader policies, possibly with a co-evolution loop.  
**What changes:** New `EnvState` with `evader_pos`, new reward function for both roles, new observation features (relative position to evader), new termination condition (capture radius).  
**Why it's valuable:** Demonstrates adversarial robustness and strategic emergent behavior. Papers on this show that pursuers self-organize into encirclement formations purely from decentralized RL.

---

### 🟠 ADDON 2: Dynamic/Moving Obstacles
**What it is:** Obstacles that move with constant or random velocities (simulating debris, boats, sea traffic). Currently our obstacles are fully static.

**Difficulty:** Medium  
**Training:** Would require retraining with moving obstacles included in experience.  
**What changes:** `EnvState.obstacles` gains a velocity component `[x, y, r, vx, vy]`. `jax_usv_env.py` updates obstacle positions every step. The observation already has a `moving_obstacles` slot in the layout — **it's unused right now!** We just need to populate it.  
**Why it's valuable:** Real ocean environments have dynamic obstacles. This directly increases real-world applicability and is a massive upgrade to show your prof.

---

### 🟠 ADDON 3: Formation Control Mode
**What it is:** Instead of independent goals, agents are given *relative offset goals* from a virtual leader point. The swarm maintains a rigid geometric formation (e.g., diamond, V-shape, line) while navigating.

**Difficulty:** Medium  
**Training:** New reward component for formation error (`current_formation_error - prev_formation_error`).  
**What changes:** New `env_params.formation_offsets` array. Each agent's "goal" is `leader_pos + formation_offset[i]`. Leader position can be a pre-planned waypoint trajectory.  
**Why it's valuable:** Formation control is a classic benchmark in cooperative MARL. Looks stunning in a GIF — you'd see the whole swarm moving as a single coordinated entity.

---

### 🟡 ADDON 4: Communication / Message Passing
**What it is:** Allow agents to broadcast a learned "message vector" to their neighbors (within range). Each agent's policy takes both its own observations AND received messages as input.

**Difficulty:** High  
**Training:** Requires full retraining with communication channels.  
**What changes:** New `CommNet` or `QMIX`-style message passing layer. At each step, each agent generates a message via a small MLP, messages are aggregated using our existing `EntityMaxEncoder`, and the result is concatenated into the actor's input.  
**Why it's valuable:** This is the state-of-the-art in cooperative MARL. Directly referenced in the paper as a future direction for their framework.

---

### 🟡 ADDON 5: Centralized Training, Decentralized Execution (CTDE) Critic
**What it is:** The critic (during training only) gets access to the **global state** — positions of ALL agents. The actor only uses local observations. This is the MAPPO paradigm from the paper.

**Difficulty:** Medium  
**Training:** New critic input shape. Actor stays the same, so deployed policy is unchanged.  
**What changes:** Modify `SoftQNetwork` to accept the concatenated full state of all agents in addition to the current agent's observation. Only affects training, not inference.  
**Why it's valuable:** The paper uses this and shows it significantly improves coordination. It's theoretically sound — agents learn better by training with global knowledge, even if they act with only local knowledge.

---

### 🟢 ADDON 6: Heterogeneous Agent Roles (Scout + Heavy)
**What it is:** Two types of USVs: fast/agile Scouts (smaller radius, higher speed) and slow/heavy Heavies (larger radius, slower). Each type has its own dynamics parameters.

**Difficulty:** Low-Medium  
**Training:** Modify `USVParams` to be agent-indexed. Existing architecture handles this naturally.  
**What changes:** `USVParams` becomes a batched array. `rk4_step` is already vmapped, so it just needs to receive different params per agent.  
**Why it's valuable:** Real naval fleets are heterogeneous. Shows that our architecture generalizes across different physical constraints.

---

### 🟢 ADDON 7: Waypoint Navigation (Multi-Goal Sequential)
**What it is:** Instead of a single goal, each agent has a **sequence of waypoints** they must visit in order. Once waypoint 1 is reached, goal automatically switches to waypoint 2, etc.

**Difficulty:** Low  
**Training:** Can be done as a curriculum — train on 1 waypoint first, then 2, then 3.  
**What changes:** `EnvState.goal_pos` becomes `EnvState.waypoints[N, K, 2]` with a `current_waypoint_idx[N]` counter. When an agent reaches its current waypoint, the counter increments.  
**Why it's valuable:** Much more realistic mission profile. Shows the policy is not just a simple "go to X" but a true path planner.

---

### 🟢 ADDON 8: Reward Shaping — Cooperative Bonus
**What it is:** Add a small reward bonus when the **whole team** reaches their goals, not just the individual. This incentivizes faster agents to slow down or help slower teammates.

**Difficulty:** Very Low (just reward engineering)  
**Training:** Requires retraining to absorb the new reward signal.  
**What changes:** Add `r_team = 100.0 * jnp.all(reached_goal)` to the reward computation in `jax_usv_env.py`.  
**Why it's valuable:** Transforms purely selfish agents into genuinely cooperative teammates. Would likely reduce the "fast agents sit idle near their goal" behavior we see in some runs.

---

### 🟢 ADDON 9: XAI — Saliency Map of Attention Weights
**What it is:** Visualize which neighboring agents and obstacles the policy was **attending to** at each timestep during inference. Export the attention matrix from `FlaxSpatioTemporalAttentionEncoder` and overlay it on the GIF.

**Difficulty:** Low (no retraining needed)  
**Training:** None — inference only change.  
**What changes:** Modify `FlaxSpatioTemporalAttentionEncoder` to return attention weights alongside output. Render the highest-attention neighbor with a highlighted circle in the GIF.  
**Why it's valuable:** Extremely powerful for a professor presentation. Shows the policy is interpretable and that attention is focusing on the most dangerous/relevant agents.

---

### 🔵 ADDON 10: Curriculum Learning Framework
**What it is:** Automate the training curriculum — start with 2 agents, 0 obstacles, short range. Every N episodes where success rate > 80%, automatically increase difficulty (more agents, more obstacles, farther goals).

**Difficulty:** Medium  
**Training:** This IS the training change — rewrites the training loop.  
**What changes:** New `CurriculumManager` class that monitors episode stats and updates `EnvParams` automatically during training.  
**Why it's valuable:** This is how our model achieved its current performance. Formalizing it makes it reproducible, publishable, and dramatically speeds up training for new experiments.

---

## My Top Recommendations for Your Thesis

| Priority | Addon | Reason |
|---|---|---|
| ⭐⭐⭐ | **#9 XAI Attention Viz** | Zero retraining, immediate visual impact for presentation |
| ⭐⭐⭐ | **#2 Moving Obstacles** | Already partially implemented! The `moving_obstacles` slot exists |
| ⭐⭐ | **#1 Pursuit-Evasion** | Directly from the paper, massive novelty |
| ⭐⭐ | **#7 Waypoint Nav** | Low effort, high demo value |
| ⭐ | **#5 CTDE Critic** | Principled algorithm upgrade, publishable contribution |
