import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple

class DeepSetOAB(nn.Module):
    """LiDAR processor for JAX. (Original Basic MLP)"""
    num_points: int = 64
    out_features: int = 64

    @nn.compact
    def __call__(self, x):
        # x is flat LiDAR array [B, num_points] containing normalized ranges
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_features)(x)
        x = nn.relu(x)
        return x

class SigmoidOAB(nn.Module):
    """IEEE Paper Implementation: Sigmoid Gating Mechanism for LiDAR."""
    num_points: int = 64
    out_features: int = 64

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(64)(x)
        gate = nn.sigmoid(h)
        feat = nn.relu(h)
        gated_feat = gate * feat
        out = nn.Dense(self.out_features)(gated_feat)
        return nn.relu(out)

class CNNOAB(nn.Module):
    """IEEE Paper Implementation: 2-Frame CNN LiDAR Encoder."""
    out_features: int = 64

    @nn.compact
    def __call__(self, x_seq):
        # x_seq shape expected: [B, 2, 64] (2 frames, 64 horizontal LiDAR beams)
        # Transpose to channel-last format for Conv1D: [B, 64, 2]
        x = jnp.transpose(x_seq, (0, 2, 1))
        
        # Conv Layer 1: 32 channels, kernel 5, stride 2, circular padding
        # Pad width = (kernel_size - 1) // 2 = 2
        x_pad = jnp.pad(x, ((0,0), (2,2), (0,0)), mode='wrap')
        h1 = nn.Conv(features=32, kernel_size=(5,), strides=(2,), padding='VALID')(x_pad)
        h1 = nn.relu(h1)
        
        # Conv Layer 2: 64 channels, kernel 3, stride 2, circular padding
        # Pad width = (kernel_size - 1) // 2 = 1
        x_pad2 = jnp.pad(h1, ((0,0), (1,1), (0,0)), mode='wrap')
        h2 = nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='VALID')(x_pad2)
        h2 = nn.relu(h2)
        
        # Flatten and map to output features
        flat = h2.reshape((x.shape[0], -1))
        out = nn.Dense(self.out_features)(flat)
        return nn.relu(out)

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
        active_counts = jnp.clip(jnp.sum(mask, axis=1, keepdims=True), 1.0, None)
        pooled = jnp.sum(h, axis=1) / active_counts
        
        # rho network
        out = nn.Dense(self.embed_dim)(pooled)
        out = nn.relu(out)
        out = nn.Dense(self.embed_dim)(out)
        
        # If no entities exist, zero out the feature
        has_entities = jnp.expand_dims(jnp.any(mask, axis=1), -1)
        return jnp.where(has_entities, out, jnp.zeros_like(out))

class GNNAttentionEncoder(nn.Module):
    """Graph Neural Network Encoder using Multi-Head Cross-Attention."""
    embed_dim: int = 64
    num_heads: int = 4

    @nn.compact
    def __call__(self, entities, query_features):
        # entities shape: [B, num_entities, feature_dim]
        # query_features shape: [B, query_dim]
        mask = entities[:, :, 0] > 0.5
        features = entities[:, :, 1:]
        
        # Embed the neighbor features (Keys and Values)
        kv = nn.Dense(self.embed_dim)(features)
        kv = nn.relu(kv)
        kv = nn.Dense(self.embed_dim)(kv)
        
        # Embed the query (Ego Kinematics)
        q = nn.Dense(self.embed_dim)(query_features)
        # Add sequence dimension for attention [B, 1, embed_dim]
        q = jnp.expand_dims(q, axis=1)
        
        # Cross-Attention mask: True where valid. shape: [B, 1, 1, num_entities]
        attn_mask = jnp.expand_dims(mask, axis=(1, 2))
        
        # MultiHeadDotProductAttention computes weighted sum of neighbors based on relevance to Ego
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.embed_dim, 
            out_features=self.embed_dim
        )(inputs_q=q, inputs_kv=kv, mask=attn_mask)
        
        # Squeeze the sequence dimension back out -> [B, embed_dim]
        attn_out = jnp.squeeze(attn_out, axis=1)
        
        # Final non-linear processing
        out = nn.Dense(self.embed_dim)(attn_out)
        out = nn.relu(out)
        out = nn.Dense(self.embed_dim)(out)
        
        # If no entities exist, zero out the feature
        has_entities = jnp.expand_dims(jnp.any(mask, axis=1), -1)
        return jnp.where(has_entities, out, jnp.zeros_like(out))

class ActorBackbone(nn.Module):
    """[LEGACY (Mean Pool)] Decentralized Actor Feature Extractor."""
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
        
        out = nn.Dense(256)(combined)
        out = nn.relu(out)
        return out

# =========================================================================================
# VARIANT 2: SPATIO-TEMPORAL LSTM ARCHITECTURE (70m Max Pool + Frame Stacking)
# =========================================================================================

class EntityMaxEncoder(nn.Module):
    """Deep Sets Encoder using Max Pooling and a strict 70m LiDAR Range Filter."""
    embed_dim: int = 64

    @nn.compact
    def __call__(self, entities, query_features=None):
        # entities shape: [B, num_entities, feature_dim]
        # features: [active_flag, rx, ry, rvx, rvy]
        mask = entities[:, :, 0] > 0.5
        features = entities[:, :, 1:]
        
        # 1. The 70m Observability Filter
        rx = features[:, :, 0]
        ry = features[:, :, 1]
        dist = jnp.sqrt(rx**2 + ry**2)
        in_range_mask = dist <= 70.0
        
        final_mask = jnp.logical_and(mask, in_range_mask)
        
        # phi network
        h = nn.Dense(self.embed_dim)(features)
        h = nn.relu(h)
        h = nn.Dense(self.embed_dim)(h)
        h = nn.relu(h)
        
        # Mask out inactive OR out-of-range entities.
        # By setting them to a large negative number, MaxPool will naturally ignore them.
        h = jnp.where(jnp.expand_dims(final_mask, -1), h, -1e9)
        
        # Max Pool (replaces the flawed Mean Pool)
        # We must handle the edge case where the entity array is strictly size 0 (e.g. moving_obstacles)
        if h.shape[1] == 0:
            pooled = jnp.zeros((h.shape[0], h.shape[2]))
        else:
            pooled = jnp.max(h, axis=1)
        
        # rho network
        out = nn.Dense(self.embed_dim)(pooled)
        out = nn.relu(out)
        out = nn.Dense(self.embed_dim)(out)
        
        # If no entities were in range, zero out the feature
        has_entities = jnp.expand_dims(jnp.any(final_mask, axis=1), -1)
        return jnp.where(has_entities, out, jnp.zeros_like(out))

class TemporalActorBackbone(nn.Module):
    """Spatio-Temporal Actor using Frame Stacking and an LSTM."""
    layout: dict
    seq_len: int = 10

    @nn.compact
    def __call__(self, x_flat):
        B = x_flat.shape[0]
        x_seq = x_flat.reshape((B, self.seq_len, 92))
        
        class SpatialExtractor(nn.Module):
            layout: dict
            @nn.compact
            def __call__(self, x):
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
                
                # Use our NEW EntityMaxEncoder for strict spatial awareness
                auv_feat = EntityMaxEncoder(embed_dim=64)(slice_entities("auv_entities"))
                moving_feat = EntityMaxEncoder(embed_dim=64)(slice_entities("moving_obstacles"))
                
                lidar_feat = DeepSetOAB(num_points=self.layout["lidar"]["dim"]//2, out_features=64)(slice_vector("lidar"))
                
                fused = nn.LayerNorm()(jnp.concatenate([kin_feat, lidar_feat], axis=1))
                combined = jnp.concatenate([fused, goal_feat, auv_feat, moving_feat], axis=1)
                
                out = nn.Dense(128)(combined)
                out = nn.relu(out)
                return out

        # Vectorize across the seq_len dimension using vmap
        VmappedSpatial = nn.vmap(
            SpatialExtractor,
            variable_axes={'params': None}, 
            split_rngs={'params': False},
            in_axes=1,  
            out_axes=1
        )
        
        spatial_seq = VmappedSpatial(layout=self.layout)(x_seq) # [B, seq_len, 128]
        
        # Temporal Memory (LSTM)
        LSTM = nn.RNN(nn.OptimizedLSTMCell(features=128), return_carry=True)
        (lstm_carry, lstm_hidden), lstm_out = LSTM(spatial_seq)
        
        out = nn.Dense(256)(lstm_hidden)
        out = nn.relu(out)
        return out

class SoftQNetwork(nn.Module):
    layout: dict
    
    @nn.compact
    def __call__(self, x, a):
        # We use the STAE Actor Backbone for the critic as well
        features = STAE_ActorBackbone(layout=self.layout)(x)
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
        features = STAE_ActorBackbone(layout=self.layout)(x)
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

# =========================================================================================
# VARIANT 3: SPATIO-TEMPORAL ATTENTION ENCODER (STAE)
# =========================================================================================

class FlaxSpatioTemporalAttentionEncoder(nn.Module):
    """Flax Spatio-Temporal Attention Encoder based on the Reference Diagram."""
    embed_dim: int = 64
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, ego_seq, entity_seqs):
        # ego_seq: [B, seq_len, 8]  (Kinematics)
        # entity_seqs: [B, seq_len, num_entities, 5] (Traffic/Neighbors)
        B, seq_len, num_entities, _ = entity_seqs.shape
        
        # 1. Ego Temporal Encoding (Query)
        ego_emb = nn.Dense(self.embed_dim)(ego_seq)
        ego_emb = nn.relu(ego_emb)
        LSTM_ego = nn.RNN(nn.OptimizedLSTMCell(self.hidden_dim), return_carry=True)
        (ego_carry, ego_hidden), _ = LSTM_ego(ego_emb)
        ego_q = jnp.expand_dims(ego_hidden, axis=1) # [B, 1, hidden_dim] -> Query Q
        
        # 2. Entity Temporal Encoding (Keys/Values)
        entity_flat = entity_seqs.reshape((B * num_entities, seq_len, -1))
        # Mask out inactive entities, but keep their features for now
        entity_feat = entity_flat[:, :, 1:] # [B*num, seq_len, 4]
        
        entity_emb = nn.Dense(self.embed_dim)(entity_feat)
        entity_emb = nn.relu(entity_emb)
        LSTM_shared = nn.RNN(nn.OptimizedLSTMCell(self.hidden_dim), return_carry=True)
        (ent_carry, ent_hidden), _ = LSTM_shared(entity_emb)
        ent_kv = ent_hidden.reshape((B, num_entities, self.hidden_dim)) # [B, num_entities, hidden_dim] -> Keys K / Values V
        
        # Attention Mask (True if active/valid). entity_seqs[..., 0] is the active flag.
        key_mask = entity_seqs[:, -1, :, 0] > 0.5 # [B, num_entities]
        
        # Handle case where num_entities is 0
        if num_entities == 0:
            attn_output = jnp.zeros_like(ego_q)
        else:
            # MultiHeadDotProductAttention requires mask shape: [batch, num_heads, q_seq_len, kv_seq_len]
            # We expand to [B, 1, 1, num_entities] to broadcast across heads and q_seq_len (which is 1)
            key_mask = jnp.expand_dims(key_mask, axis=(1, 2))
            # 3. Cross-Attention
            attn_output = nn.MultiHeadDotProductAttention(num_heads=4, qkv_features=self.hidden_dim)(
                inputs_q=ego_q, inputs_kv=ent_kv, mask=key_mask
            )
        
        # 4. Add & Norm (Residual)
        ego_interactive_enc = nn.LayerNorm()(jnp.squeeze(ego_q, 1) + jnp.squeeze(attn_output, 1)) # [B, hidden_dim]
        
        # 5. FC Layer
        z_t = nn.Dense(128)(ego_interactive_enc)
        z_t = nn.relu(z_t)
        
        return z_t

class STAE_ActorBackbone(nn.Module):
    """Integrates STAE with Context Encoders (LiDAR & Goal)."""
    layout: dict
    seq_len: int = 10

    @nn.compact
    def __call__(self, x_flat):
        B = x_flat.shape[0]
        x_seq = x_flat.reshape((B, self.seq_len, 92))
        
        def slice_vector_seq(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            return x_seq[:, :, start:start+dim]
            
        def slice_entities_seq(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            count = self.layout[name]["count"]
            feat_dim = self.layout[name]["feature_dim"]
            flat = x_seq[:, :, start:start+dim]
            return flat.reshape((B, self.seq_len, count, feat_dim))
            
        # TOP BRANCH: STAE (Spatio-Temporal Attention Encoder)
        ego_seq = slice_vector_seq("ego") # [B, seq_len, 8]
        auv_seqs = slice_entities_seq("auv_entities") # [B, seq_len, 4, 5]
        moving_seqs = slice_entities_seq("moving_obstacles") # [B, seq_len, 0, 5]
        
        # Combine all entities into one traffic array
        traffic_seqs = jnp.concatenate([auv_seqs, moving_seqs], axis=2)
        
        z_t = FlaxSpatioTemporalAttentionEncoder(embed_dim=64, hidden_dim=128)(ego_seq, traffic_seqs) # [B, 128]
        
        # BOTTOM BRANCH: Context Encoders (Using ONLY current frame t=0, which is the last frame in the sequence)
        current_frame = x_seq[:, -1, :]
        
        def slice_current(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            return current_frame[:, start:start+dim]
            
        # Goal MLP
        goal_feat = nn.Dense(32)(slice_current("goal"))
        goal_feat = nn.LayerNorm()(goal_feat)
        goal_feat = nn.relu(goal_feat)
        
        # LiDAR Dual-Stream OAB -> replaced by CNNOAB (Temporal 2-Frame Stack)
        # We slice the last 2 frames from the LiDAR sequence
        lidar_seq = slice_vector_seq("lidar")[:, -2:, :] # [B, 2, 64]
        lidar_feat = CNNOAB(out_features=64)(lidar_seq)
        
        # Fusion C_t
        c_t = jnp.concatenate([goal_feat, lidar_feat], axis=1) # [B, 96]
        
        # Final Fusion S_t
        s_t = jnp.concatenate([z_t, c_t], axis=1) # [B, 128 + 96] = [B, 224]
        
        out = nn.Dense(256)(s_t)
        out = nn.relu(out)
        return out

# =========================================================================================
# CHAPTER 7 ARCHITECTURE: SWARM TRANSFORMER + GRU
# -----------------------------------------------------------------------------------------
# NOTE: This architecture relies on Self-Attention (all entities attend to all entities)
# and processes temporal sequences using a Gated Recurrent Unit (GRU).
# The input 'x' must have shape [B, seq_len, feature_dim].
# =========================================================================================
#
# class TransformerBlock(nn.Module):
#     embed_dim: int = 64
#     num_heads: int = 4
#     
#     @nn.compact
#     def __call__(self, x, mask=None):
#         # x shape: [B, num_tokens, embed_dim]
#         # mask shape: [B, 1, num_tokens]
#         
#         # 1. Multi-Head Self Attention
#         attn_out = nn.MultiHeadDotProductAttention(
#             num_heads=self.num_heads, 
#             qkv_features=self.embed_dim, 
#             out_features=self.embed_dim
#         )(inputs_q=x, inputs_kv=x, mask=mask)
#         
#         # 2. Residual Add + LayerNorm
#         x = nn.LayerNorm()(x + attn_out)
#         
#         # 3. Feed Forward Network (FFN)
#         ffn_out = nn.Dense(self.embed_dim * 4)(x)
#         ffn_out = nn.relu(ffn_out)
#         ffn_out = nn.Dense(self.embed_dim)(ffn_out)
#         
#         # 4. Residual Add + LayerNorm
#         out = nn.LayerNorm()(x + ffn_out)
#         return out
#
# class SwarmTransformer(nn.Module):
#     embed_dim: int = 64
#     num_heads: int = 4
#     num_layers: int = 2
#     
#     @nn.compact
#     def __call__(self, ego_features, entity_features, entity_mask):
#         # Concatenate Ego token as the first token in the sequence
#         # tokens shape: [B, 1 + num_entities, embed_dim]
#         ego_emb = nn.Dense(self.embed_dim)(ego_features)
#         ego_emb = jnp.expand_dims(ego_emb, axis=1)
#         tokens = jnp.concatenate([ego_emb, entity_features], axis=1)
#         
#         # Create Self-Attention Mask (Ego is always True)
#         ego_mask = jnp.ones((entity_mask.shape[0], 1), dtype=bool)
#         full_mask = jnp.concatenate([ego_mask, entity_mask], axis=1)
#         attn_mask = jnp.expand_dims(full_mask, axis=1)
#         
#         x = tokens
#         for _ in range(self.num_layers):
#             x = TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)(x, mask=attn_mask)
#             
#         # The updated Ego token contains the global spatial awareness of the entire swarm
#         ego_out = x[:, 0, :]
#         return ego_out
#
# class TransformerGRUActor(nn.Module):
#     embed_dim: int = 64
#     
#     @nn.compact
#     def __call__(self, ego_seq, entity_seqs):
#         # ego_seq: [B, seq_len, ego_dim]
#         # entity_seqs: [B, seq_len, num_entities, feature_dim]
#         B, seq_len, num_entities, _ = entity_seqs.shape
#         
#         # Vectorize the SwarmTransformer across the time (seq_len) dimension using nn.vmap
#         VmappedTransformer = nn.vmap(
#             SwarmTransformer,
#             variable_axes={'params': None}, # Share weights across all time steps
#             split_rngs={'params': False},
#             in_axes=(1, 1, 1),
#             out_axes=1
#         )
#         
#         ent_mask = entity_seqs[:, :, :, 0] > 0.5
#         ent_feat = entity_seqs[:, :, :, 1:]
#         ent_emb = nn.Dense(self.embed_dim)(ent_feat)
#         ent_emb = nn.relu(ent_emb)
#         
#         # spatial_features shape: [B, seq_len, embed_dim]
#         spatial_features = VmappedTransformer(embed_dim=self.embed_dim)(ego_seq, ent_emb, ent_mask)
#         
#         # Process the spatial sequence through a GRU (Gated Recurrent Unit)
#         GRU = nn.RNN(nn.GRUCell(features=128), return_carry=True)
#         (gru_carry, gru_hidden), gru_out = GRU(spatial_features)
#         
#         # The final hidden state contains the complete Spatio-Temporal awareness
#         out = nn.Dense(256)(gru_hidden)
#         out = nn.relu(out)
#         
#         return out
