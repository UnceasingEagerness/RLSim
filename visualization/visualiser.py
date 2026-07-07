"""
visualiser.py
=============
PyGame-based Multi-Domain Visualizer for USV and AUV swarms.

Features:
- [TAB] toggles between 'Grid POV' mode and 'Global Map' mode.
- Global Map supports WASD panning and Mouse Scroll zooming.
- AUVs are rendered with depth-shading (darker = deeper) and depth tags.
- USVs are rendered on the surface.
"""

import pygame
import numpy as np
import math

class Swarm2DVisualizer:
    def __init__(self, num_agents=4, local_view_range=80.0, lidar_range=50.0):
        pygame.init()
        self.num_agents       = num_agents
        self.lidar_range      = lidar_range
        self.local_view_range = local_view_range

        # POV Grid layout
        self.cols = math.ceil(math.sqrt(max(num_agents, 1)))
        self.rows = math.ceil(max(num_agents, 1) / self.cols)

        self.panel_size = 400
        self.pov_scale  = self.panel_size / local_view_range

        self.screen_w = self.cols * self.panel_size
        self.screen_h = self.rows * self.panel_size

        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Multi-Domain Nav — Global Command Center")
        self.font  = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 18)
        self.clock = pygame.time.Clock()

        # View state
        self.mode = "GLOBAL"  # 'GLOBAL' or 'POV'
        self.global_offset_x = 0.0
        self.global_offset_y = 0.0
        self.global_scale = 1.0

        # Colour palette
        self.bg_color              = (15,  45,  65)
        self.grid_color            = (30,  70,  90)
        self.agent_color           = (255, 165,  0)   # orange — USV
        self.auv_base_color        = (0, 150, 255)    # blue — AUV
        self.nbor_color            = (200, 100, 255)  # purple — other USVs
        self.moving_obstacle_color = (255, 220,  70)  # yellow — moving obstacles
        self.static_obs_color      = (160,  80,  30)  # brown  — static obstacles
        self.lidar_hit_color       = (0,   255, 100)
        self.lidar_free_color      = (30,  100,  50)
        self.goal_color            = (255,  50,  50)
        self.border_color          = (200, 200, 200)

    def _get_auv_color(self, depth):
        # Base: (0, 150, 255)
        # Depth is negative. At z=0 -> bright blue. At z=-50 -> dark blue.
        intensity = max(0.2, min(1.0, 1.0 + (depth / 50.0)))
        return (0, int(150 * intensity), int(255 * intensity))

    # ── POV Coords ────────────────────────────────────────────────────────────
    def _to_panel(self, tx, ty, ex, ey):
        dx =  (tx - ex) * self.pov_scale + self.panel_size / 2
        dy = -(ty - ey) * self.pov_scale + self.panel_size / 2
        return int(dx), int(dy)

    # ── Global Coords ─────────────────────────────────────────────────────────
    def _to_global(self, tx, ty):
        dx =  (tx - self.global_offset_x) * self.global_scale + self.screen_w / 2
        dy = -(ty - self.global_offset_y) * self.global_scale + self.screen_h / 2
        return int(dx), int(dy)

    def _draw_entity_global(self, surface, x, y, z, is_auv, color, radius, label=None):
        gx, gy = self._to_global(x, y)
        if 0 <= gx < self.screen_w and 0 <= gy < self.screen_h:
            draw_radius = max(3, int(radius * self.global_scale))
            if is_auv:
                c = self._get_auv_color(z)
                pygame.draw.circle(surface, c, (gx, gy), draw_radius)
                pygame.draw.circle(surface, (255, 255, 255), (gx, gy), draw_radius, 1) # outline
                if label:
                    text = self.small_font.render(f"{label} [{z:.1f}m]", True, (255, 255, 255))
                    surface.blit(text, (gx + draw_radius + 2, gy - draw_radius))
            else:
                pygame.draw.circle(surface, color, (gx, gy), draw_radius)
                if label:
                    text = self.small_font.render(label, True, (255, 255, 255))
                    surface.blit(text, (gx + draw_radius + 2, gy - draw_radius))

    def _handle_inputs(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.mode = "POV" if self.mode == "GLOBAL" else "GLOBAL"
            
            if self.mode == "GLOBAL":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4: # Scroll Up
                        self.global_scale *= 1.2
                    elif event.button == 5: # Scroll Down
                        self.global_scale /= 1.2
        
        # Continuous keys for panning
        if self.mode == "GLOBAL":
            keys = pygame.key.get_pressed()
            pan_speed = 20.0 / self.global_scale
            if keys[pygame.K_w] or keys[pygame.K_UP]:    self.global_offset_y += pan_speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:  self.global_offset_y -= pan_speed
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:  self.global_offset_x -= pan_speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.global_offset_x += pan_speed

        return True

    def render(self, obs_dict, info_dict, goals):
        if not self._handle_inputs():
            return

        self.screen.fill((0, 0, 0))

        if self.mode == "GLOBAL":
            self._render_global(info_dict, goals)
        else:
            self._render_pov(obs_dict, info_dict, goals)

        pygame.display.flip()
        self.clock.tick(30)

    # ── GLOBAL RENDER ─────────────────────────────────────────────────────────
    def _render_global(self, info_dict, goals):
        self.screen.fill(self.bg_color)
        
        # Grid
        grid_sp = 50.0
        start_x = (int((self.global_offset_x - (self.screen_w/2)/self.global_scale) / grid_sp) - 1) * grid_sp
        end_x   = (int((self.global_offset_x + (self.screen_w/2)/self.global_scale) / grid_sp) + 1) * grid_sp
        start_y = (int((self.global_offset_y - (self.screen_h/2)/self.global_scale) / grid_sp) - 1) * grid_sp
        end_y   = (int((self.global_offset_y + (self.screen_h/2)/self.global_scale) / grid_sp) + 1) * grid_sp
        
        for i in np.arange(start_x, end_x, grid_sp):
            p1 = self._to_global(i, start_y)
            p2 = self._to_global(i, end_y)
            pygame.draw.line(self.screen, self.grid_color, p1, p2)
        for i in np.arange(start_y, end_y, grid_sp):
            p1 = self._to_global(start_x, i)
            p2 = self._to_global(end_x, i)
            pygame.draw.line(self.screen, self.grid_color, p1, p2)

        # Draw Entities
        for aid, info in info_dict.items():
            if not isinstance(info, dict) or "pos" not in info:
                continue
            
            x, y, z = info["pos"]
            is_auv = "auv" in aid
            is_mob = info.get("type") == "moving_obstacle"
            color = self.moving_obstacle_color if is_mob else (self.auv_base_color if is_auv else self.agent_color)
            
            # Goal line
            if aid in goals:
                gx, gy = goals[aid][0], goals[aid][1]
                gsx, gsy = self._to_global(gx, gy)
                ex, ey = self._to_global(x, y)
                pygame.draw.line(self.screen, self.goal_color, (ex, ey), (gsx, gsy), 1)
                pygame.draw.circle(self.screen, self.goal_color, (gsx, gsy), max(3, int(2*self.global_scale)))
            
            # Body
            self._draw_entity_global(self.screen, x, y, z, is_auv, color, 4.0, label=aid)

        # Mode Text
        text = self.font.render("GLOBAL MAP MODE (WASD to pan, Scroll to zoom, TAB to switch)", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

    # ── POV RENDER ────────────────────────────────────────────────────────────
    def _render_pov(self, obs_dict, info_dict, goals):
        panel_idx = 0
        for ego_id, obs in obs_dict.items():
            if ego_id not in info_dict or "pos" not in info_dict[ego_id]:
                continue
                
            panel = pygame.Surface((self.panel_size, self.panel_size))
            panel.fill(self.bg_color)
            
            ego_pos = info_dict[ego_id]["pos"]
            ex, ey, ez = float(ego_pos[0]), float(ego_pos[1]), float(ego_pos[2])
            
            is_ego_auv = "auv" in ego_id
            if is_ego_auv:
                # AUV has pitch/roll/yaw in obs? Actually cleanrl_sac_pz might just have yaw in ego_dim
                sin_yaw, cos_yaw = float(obs[0]), float(obs[1])
                ego_yaw = math.atan2(sin_yaw, cos_yaw)
            else:
                sin_yaw, cos_yaw = float(obs[0]), float(obs[1])
                ego_yaw = math.atan2(sin_yaw, cos_yaw)

            # 1. Grid
            grid_sp = 20.0
            off_x = ex % grid_sp
            off_y = ey % grid_sp
            for i in np.arange(-self.local_view_range, self.local_view_range, grid_sp):
                p1 = self._to_panel(ex - off_x + i, ey + self.local_view_range, ex, ey)
                p2 = self._to_panel(ex - off_x + i, ey - self.local_view_range, ex, ey)
                pygame.draw.line(panel, self.grid_color, p1, p2)
                p3 = self._to_panel(ex + self.local_view_range, ey - off_y + i, ex, ey)
                p4 = self._to_panel(ex - self.local_view_range, ey - off_y + i, ex, ey)
                pygame.draw.line(panel, self.grid_color, p3, p4)

            # 2. Neighbors
            for nbor_id, nbor_info in info_dict.items():
                if nbor_id == ego_id or "pos" not in nbor_info:
                    continue
                nx, ny, nz = nbor_info["pos"]
                nsx, nsy = self._to_panel(nx, ny, ex, ey)
                is_mob = nbor_info.get("type") == "moving_obstacle"
                is_auv = "auv" in nbor_id
                
                if is_mob:
                    c = self.moving_obstacle_color
                elif is_auv:
                    c = self._get_auv_color(nz)
                else:
                    c = self.nbor_color
                    
                radius = max(4, int(2.2 * self.pov_scale))
                pygame.draw.circle(panel, c, (nsx, nsy), radius)
                if is_auv:
                    pygame.draw.circle(panel, (255, 255, 255), (nsx, nsy), radius, 1)

            # 3. Goal
            if ego_id in goals:
                gx, gy = float(goals[ego_id][0]), float(goals[ego_id][1])
                gsx, gsy = self._to_panel(gx, gy, ex, ey)
                pygame.draw.circle(panel, self.goal_color, (gsx, gsy), max(3, int(2 * self.pov_scale)))
                pygame.draw.line(panel, self.goal_color, (self.panel_size//2, self.panel_size//2), (gsx, gsy), 1)

            # 4. LiDAR (Only for USVs)
            if not is_ego_auv:
                lidar_flat  = obs[-128:]
                lidar_dists = lidar_flat[0::2] * self.lidar_range
                for bi, dist in enumerate(lidar_dists):
                    angle_deg = bi * (360.0 / 64)
                    ray_angle = ego_yaw + math.radians(angle_deg)
                    end_x = ex + dist * math.cos(ray_angle)
                    end_y = ey + dist * math.sin(ray_angle)
                    lx, ly = self._to_panel(end_x, end_y, ex, ey)
                    hit = dist < (self.lidar_range - 0.5)
                    color = self.lidar_hit_color if hit else self.lidar_free_color
                    pygame.draw.line(panel, color, (self.panel_size//2, self.panel_size//2), (lx, ly), 1)
                    if hit:
                        pygame.draw.circle(panel, (255, 0, 0), (lx, ly), 2)

            # 5. Ego
            length = 5 * self.pov_scale
            width  = 2.5 * self.pov_scale
            cx, cy = self.panel_size // 2, self.panel_size // 2
            p1 = (cx + length * math.cos(ego_yaw),       cy - length * math.sin(ego_yaw))
            p2 = (cx + width  * math.cos(ego_yaw + 2.5), cy - width  * math.sin(ego_yaw + 2.5))
            p3 = (cx + width  * math.cos(ego_yaw - 2.5), cy - width  * math.sin(ego_yaw - 2.5))
            
            ego_col = self._get_auv_color(ez) if is_ego_auv else self.agent_color
            pygame.draw.polygon(panel, ego_col, [p1, p2, p3])

            # Label
            label = self.font.render(f"POV: {ego_id} (Z: {ez:.1f}m)", True, (255, 255, 255))
            panel.blit(label, (10, 10))
            pygame.draw.rect(panel, self.border_color, (0, 0, self.panel_size, self.panel_size), 3)

            row = panel_idx // self.cols
            col = panel_idx % self.cols
            self.screen.blit(panel, (col * self.panel_size, row * self.panel_size))
            panel_idx += 1
