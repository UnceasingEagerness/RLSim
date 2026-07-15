import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Any

# -------------------------------------------------------------------------
# Feature Embedding Block (FEB)
# -------------------------------------------------------------------------
class FEB(nn.Module):
    """
    Processes variable/multiple homogeneous inputs (like N teammates).
    Applies shared weights, then Column-wise Max-Pooling (CMP) and Avg-Pooling (CAP).
    """
    out_dim: int = 32
    
    @nn.compact
    def __call__(self, x):
        # x is expected to be [Batch, Seq, Num_Neighbors, Feat_Dim]
        # or simply [Batch, Num_Neighbors, Feat_Dim]
        
        # 1. Shared Feature Extraction for each neighbor
        h = nn.Dense(64)(x)
        h = nn.swish(h)
        h = nn.Dense(32)(h)
        h = nn.swish(h)
        
        # 2. CMP & CAP Aggregation across the Num_Neighbors axis
        max_pool = jnp.max(h, axis=-2)
        avg_pool = jnp.mean(h, axis=-2)
        
        # 3. Concatenate and map to final dimension
        concat_pool = jnp.concatenate([max_pool, avg_pool], axis=-1)
        
        z_var = nn.Dense(self.out_dim)(concat_pool)
        z_var = nn.swish(z_var)
        return z_var

# -------------------------------------------------------------------------
# Obstacle Awareness Block (OAB)
# -------------------------------------------------------------------------
class OAB(nn.Module):
    """
    IEEE Paper Implementation: Sigmoid Gating Mechanism for LiDAR.
    Projects 64-beam LiDAR into a semantic embedding using a gated pathway.
    """
    out_dim: int = 32
    
    @nn.compact
    def __call__(self, lidar_scans):
        # x is [..., 64]
        x = nn.Dense(64)(lidar_scans)
        gate = nn.sigmoid(x)
        feat = nn.swish(x)
        
        # Element-wise gating
        gated_feat = gate * feat
        
        out = nn.Dense(32)(gated_feat)
        return nn.swish(out)

# -------------------------------------------------------------------------
# Masked Self-Attention (MSA) Module
# -------------------------------------------------------------------------
class MSA(nn.Module):
    """
    Processes historical frames using Multi-Head Self-Attention.
    """
    num_heads: int = 4
    out_dim: int = 32
    
    @nn.compact
    def __call__(self, seq_features):
        # seq_features: [Batch, Time, Feat_Dim]
        
        # Attention Layer
        attn_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            inputs_q=seq_features, 
            inputs_kv=seq_features
        )
        
        # Residual + LayerNorm
        h = seq_features + attn_out
        h = nn.LayerNorm()(h)
        
        # Take the most recent timestep as the temporal summary
        temporal_summary = h[:, -1, :] 
        
        z_out = nn.Dense(self.out_dim)(temporal_summary)
        z_out = nn.swish(z_out)
        return z_out

# -------------------------------------------------------------------------
# MASAC PURSUER ARCHITECTURE
# -------------------------------------------------------------------------
class PursuerActorBackbone(nn.Module):
    @nn.compact
    def __call__(self, obs_history):
        # obs_history is [Batch, Time, 77]
        batch_size, time_steps, obs_dim = obs_history.shape
        
        # 1. Slice observations
        # O_out: 3 | O_in: 10 | LiDAR: 64
        O_out = obs_history[:, :, :3]
        O_in = obs_history[:, :, 3:13]
        lidar = obs_history[:, :, 13:77]
        
        # Reshape O_in to [Batch, Time, 2_neighbors, 5_features]
        O_in = O_in.reshape((batch_size, time_steps, 2, 5))
        
        # 2. Process features independently (vectorized across time)
        # O_out map
        z_out = nn.Dense(32)(O_out)
        z_out = nn.swish(z_out)
        
        # FEB
        z_var = FEB(out_dim=32)(O_in)
        
        # OAB
        z_obs = OAB(out_dim=32)(lidar)
        
        # 3. Concatenate modalities
        concat_features = jnp.concatenate([z_out, z_var, z_obs], axis=-1) # [Batch, Time, 96]
        
        # 4. Temporal Fusion (MSA)
        temporal_features = MSA(num_heads=4, out_dim=64)(concat_features) # [Batch, 64]
        
        return temporal_features

class PursuerActor(nn.Module):
    action_dim: int = 2
    
    @nn.compact
    def __call__(self, obs_history):
        features = PursuerActorBackbone()(obs_history)
        
        h = nn.Dense(64)(features)
        h = nn.swish(h)
        
        means = nn.Dense(self.action_dim)(h)
        log_stds = nn.Dense(self.action_dim)(h)
        log_stds = jnp.clip(log_stds, -20.0, 2.0)
        
        return means, log_stds

# -------------------------------------------------------------------------
# MASAC EVADER ARCHITECTURE
# -------------------------------------------------------------------------
class EvaderActorBackbone(nn.Module):
    @nn.compact
    def __call__(self, obs_history):
        # obs_history is [Batch, Time, 77]
        batch_size, time_steps, obs_dim = obs_history.shape
        
        # Evader uses: O_out_e (6) + LiDAR (64) + 7 Padding
        O_out_e = obs_history[:, :, :6]
        lidar = obs_history[:, :, 6:70]
        
        # Reshape O_out_e to [Batch, Time, 3_pursuers, 2_features]
        O_out_e = O_out_e.reshape((batch_size, time_steps, 3, 2))
        
        # Evader FEB processes the pursuers
        z_var = FEB(out_dim=32)(O_out_e)
        z_obs = OAB(out_dim=32)(lidar)
        
        concat_features = jnp.concatenate([z_var, z_obs], axis=-1) # [Batch, Time, 64]
        
        temporal_features = MSA(num_heads=4, out_dim=64)(concat_features)
        return temporal_features

class EvaderActor(nn.Module):
    action_dim: int = 2
    
    @nn.compact
    def __call__(self, obs_history):
        features = EvaderActorBackbone()(obs_history)
        
        h = nn.Dense(64)(features)
        h = nn.swish(h)
        
        means = nn.Dense(self.action_dim)(h)
        log_stds = nn.Dense(self.action_dim)(h)
        log_stds = jnp.clip(log_stds, -20.0, 2.0)
        
        return means, log_stds

# -------------------------------------------------------------------------
# CTDE CRITICS
# -------------------------------------------------------------------------
class PursuerCritic(nn.Module):
    @nn.compact
    def __call__(self, global_obs_history, joint_actions):
        # global_obs_history: [Batch, Time, num_agents * 77]
        # joint_actions: [Batch, num_agents * 2]
        
        # Simplistic global fusion for the CTDE critic
        x = jnp.concatenate([global_obs_history[:, -1, :], joint_actions], axis=-1)
        
        # Q1
        q1 = nn.Dense(256)(x)
        q1 = nn.LayerNorm()(q1)
        q1 = nn.swish(q1)
        q1 = nn.Dense(128)(q1)
        q1 = nn.LayerNorm()(q1)
        q1 = nn.swish(q1)
        q1 = nn.Dense(1)(q1)
        
        # Q2
        q2 = nn.Dense(256)(x)
        q2 = nn.LayerNorm()(q2)
        q2 = nn.swish(q2)
        q2 = nn.Dense(128)(q2)
        q2 = nn.LayerNorm()(q2)
        q2 = nn.swish(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2

class EvaderCritic(nn.Module):
    @nn.compact
    def __call__(self, global_obs_history, joint_actions):
        # Symmetrical structure for the evader critic
        x = jnp.concatenate([global_obs_history[:, -1, :], joint_actions], axis=-1)
        
        # Q1
        q1 = nn.Dense(256)(x)
        q1 = nn.LayerNorm()(q1)
        q1 = nn.swish(q1)
        q1 = nn.Dense(128)(q1)
        q1 = nn.LayerNorm()(q1)
        q1 = nn.swish(q1)
        q1 = nn.Dense(1)(q1)
        
        # Q2
        q2 = nn.Dense(256)(x)
        q2 = nn.LayerNorm()(q2)
        q2 = nn.swish(q2)
        q2 = nn.Dense(128)(q2)
        q2 = nn.LayerNorm()(q2)
        q2 = nn.swish(q2)
        q2 = nn.Dense(1)(q2)
        
        return q1, q2
