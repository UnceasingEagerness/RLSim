import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple

class DeepSetOAB(nn.Module):
    """PointNet-style LiDAR processor for JAX."""
    num_points: int = 64
    out_features: int = 64

    @nn.compact
    def __call__(self, x):
        # x is flat LiDAR array [B, num_points * 2]
        B = x.shape[0]
        x = x.reshape((B, self.num_points, 2))
        
        # Shared MLP across points
        x = nn.Dense(32)(x)
        x = nn.relu(x)  # Using standard ReLU in Flax instead of Mish for speed unless Mish is needed
        x = nn.Dense(self.out_features)(x)
        x = nn.relu(x)
        
        # Max pooling across points
        pooled = jnp.max(x, axis=1)
        return pooled

class EntitySetEncoder(nn.Module):
    """Deep Sets Encoder for permutation-invariant entity sets."""
    embed_dim: int = 64

    @nn.compact
    def __call__(self, entities, query_features=None):
        # entities shape: [B, num_entities, feature_dim]
        # features: [active_flag, rx, ry, rvx, rvy]
        mask = entities[:, :, 0] > 0.5
        features = entities[:, :, 1:]
        
        # phi network
        h = nn.Dense(self.embed_dim)(features)
        h = nn.relu(h)
        h = nn.Dense(self.embed_dim)(h)
        h = nn.relu(h)
        
        # mask out inactive entities
        h = h * jnp.expand_dims(mask, -1)
        
        # mean pool
        active_counts = jnp.clip(jnp.sum(mask, axis=1, keepdims=True), a_min=1.0)
        pooled = jnp.sum(h, axis=1) / active_counts
        
        # rho network
        out = nn.Dense(self.embed_dim)(pooled)
        out = nn.relu(out)
        out = nn.Dense(self.embed_dim)(out)
        
        # If no entities exist, zero out the feature
        has_entities = jnp.expand_dims(jnp.any(mask, axis=1), -1)
        return jnp.where(has_entities, out, jnp.zeros_like(out))

class ActorBackbone(nn.Module):
    """Decentralized Actor Feature Extractor."""
    layout: dict

    @nn.compact
    def __call__(self, x):
        ego_spec = self.layout["ego"]
        goal_spec = self.layout["goal"]
        lidar_spec = self.layout["lidar"]
        auv_spec = self.layout["auv_entities"]
        mob_spec = self.layout["moving_obstacles"]
        
        def slice_vector(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            return x[:, start:start+dim]
            
        def slice_entities(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            count = self.layout[name]["count"]
            feat_dim = self.layout[name]["feature_dim"]
            flat = x[:, start:start+dim]
            return flat.reshape((x.shape[0], count, feat_dim))

        kin_feat = nn.Dense(64)(slice_vector("ego"))
        kin_feat = nn.LayerNorm()(kin_feat)
        kin_feat = nn.relu(kin_feat)
        kin_feat = nn.Dense(64)(kin_feat)
        kin_feat = nn.relu(kin_feat)
        
        goal_feat = nn.Dense(32)(slice_vector("goal"))
        goal_feat = nn.LayerNorm()(goal_feat)
        goal_feat = nn.relu(goal_feat)
        
        auv_feat = EntitySetEncoder(embed_dim=64)(slice_entities("auv_entities"))
        moving_feat = EntitySetEncoder(embed_dim=64)(slice_entities("moving_obstacles"))
        lidar_feat = DeepSetOAB(num_points=lidar_spec["dim"]//2, out_features=64)(slice_vector("lidar"))
        
        fused = nn.LayerNorm()(jnp.concatenate([kin_feat, lidar_feat], axis=1))
        combined = jnp.concatenate([fused, goal_feat, auv_feat, moving_feat], axis=1)
        
        # No LSTM for now to keep JAX port pure and simple
        out = nn.Dense(256)(combined)
        out = nn.relu(out)
        return out

class SoftQNetwork(nn.Module):
    layout: dict
    
    @nn.compact
    def __call__(self, x, a):
        # We use the ActorBackbone structure for the critic as well (since this is decentralised execution)
        features = ActorBackbone(layout=self.layout)(x)
        x = jnp.concatenate([features, a], axis=1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class Actor(nn.Module):
    layout: dict
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    
    @nn.compact
    def __call__(self, x):
        features = ActorBackbone(layout=self.layout)(x)
        mean = nn.Dense(self.action_dim)(features)
        log_std = nn.Dense(self.action_dim)(features)
        log_std = jnp.clip(log_std, -5.0, 2.0)
        return mean, log_std

    def get_action(self, x, key):
        mean, log_std = self(x)
        std = jnp.exp(log_std)
        normal = mean + std * jax.random.normal(key, mean.shape)
        action = jnp.tanh(normal)
        action_env = action * self.action_scale + self.action_bias
        # We need the log_prob
        log_prob = jax.scipy.stats.norm.logpdf(normal, mean, std) - jnp.log(1 - action**2 + 1e-6)
        log_prob = jnp.sum(log_prob, axis=1, keepdims=True)
        return action_env, log_prob
