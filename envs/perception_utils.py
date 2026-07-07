import numpy as np

# ============================================================
#                   KALMAN FILTER
# ============================================================

class KinematicKalmanFilter:
    """
    Constant Velocity Kalman Filter
    State: [x, y, vx, vy]
    """
    def __init__(self, dt, process_noise=1e-2, measurement_noise=1e-1):
        self.dt = dt
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.eye(4, dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(4, dtype=np.float32) * measurement_noise
        self.x = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)
        self.initialized = False

    def reset(self):
        self.x = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)
        self.initialized = False

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement.copy()
            self.initialized = True
            return self.x

        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        return self.x

# ============================================================
#                   ENTITY TRACKER
# ============================================================

class EntityTracker:
    """Persistent temporal model for ONE entity."""
    def __init__(self, entity_id, entity_type, dt):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.kf = KinematicKalmanFilter(dt)
        self.last_state = None

    def update(self, measurement):
        self.kf.predict()
        filtered = self.kf.update(measurement)
        self.last_state = filtered
        return filtered

# ============================================================
#                   PERCEPTION MODULE
# ============================================================

class PerceptionModule:
    """
    Multi-agent perception engine.
    Responsibilities:
        - KF smoothing of raw simulator dynamics
    """
    def __init__(self, dt=0.5, map_size=600.0):
        self.dt = dt
        self.map_size = map_size
        self.entity_trackers = {}
        self.current_states = {}

    def reset(self):
        self.entity_trackers.clear()
        self.current_states.clear()

    def get_tracker(self, entity_id, entity_type):
        if entity_id not in self.entity_trackers:
            self.entity_trackers[entity_id] = EntityTracker(
                entity_id=entity_id,
                entity_type=entity_type,
                dt=self.dt
            )
        return self.entity_trackers[entity_id]

    def update_entities(self, entity_states):
        """Smooth and cache all dynamic entities once per simulator step."""
        smoothed_states = {}
        for entity_id, state in entity_states.items():
            entity_type = state.get("class", "moving_obstacle")
            if entity_type == "static_obstacle":
                continue

            tracker = self.get_tracker(entity_id, entity_type)
            measurement = np.array([
                state["pos"][0],
                state["pos"][1],
                state["vel"][0],
                state["vel"][1]
            ], dtype=np.float32)

            smoothed = tracker.update(measurement)
            smoothed_states[entity_id] = {
                "pos": smoothed[:2],
                "vel": smoothed[2:4],
                "yaw": float(state.get("yaw", 0.0)),
                "class": entity_type
            }

        self.current_states = smoothed_states
        return smoothed_states

    def extract_entity_context(self, ego_id, target_classes=None):
        """
        Return entity rows relative to ego_id.
        Row schema: [active, rel_x, rel_y, rel_vx, rel_vy]
        """
        if target_classes is not None:
            target_classes = set(target_classes)

        if ego_id not in self.current_states:
            return np.zeros((0, 5), dtype=np.float32)

        ego_state = self.current_states[ego_id]
        ego_pos = ego_state["pos"]
        ego_vel = ego_state["vel"]
        ego_yaw = float(ego_state.get("yaw", 0.0))

        entity_features = []
        for target_id, target_state in self.current_states.items():
            if target_id == ego_id:
                continue

            if target_classes is not None and target_state["class"] not in target_classes:
                continue

            rel_pos_world = target_state["pos"] - ego_pos
            rel_vel_world = target_state["vel"] - ego_vel
            rel_pos = self._world_to_ego_frame(rel_pos_world, ego_yaw)
            rel_vel = self._world_to_ego_frame(rel_vel_world, ego_yaw)

            entity_features.append([
                1.0,                                            # is_active
                rel_pos[0] / self.map_size,                     # rel_x
                rel_pos[1] / self.map_size,                     # rel_y
                rel_vel[0] / 10.0,                              # rel_vx
                rel_vel[1] / 10.0,                              # rel_vy
            ])

        if not entity_features:
            return np.zeros((0, 5), dtype=np.float32)

        return np.array(entity_features, dtype=np.float32)

    def _make_feature(self, rel_pos_world, rel_vel_world, ego_yaw):
        rel_pos = self._world_to_ego_frame(rel_pos_world, ego_yaw)
        rel_vel = self._world_to_ego_frame(rel_vel_world, ego_yaw)

        return np.array([
            1.0,
            np.clip(rel_pos[0] / self.map_size, -1.0, 1.0),
            np.clip(rel_pos[1] / self.map_size, -1.0, 1.0),
            np.clip(rel_vel[0] / 20.0, -1.0, 1.0),
            np.clip(rel_vel[1] / 20.0, -1.0, 1.0),
        ], dtype=np.float32)

    @staticmethod
    def _world_to_ego_frame(vec_xy, ego_yaw):
        c = np.cos(ego_yaw)
        s = np.sin(ego_yaw)
        x_body = c * vec_xy[0] + s * vec_xy[1]
        y_body = -s * vec_xy[0] + c * vec_xy[1]
        return np.array([x_body, y_body], dtype=np.float32)

    def extract_swarm_context(self, ego_id, all_agent_states=None):
        if all_agent_states is not None:
            self.update_entities(all_agent_states)
        return self.extract_entity_context(ego_id, target_classes=("auv",))
