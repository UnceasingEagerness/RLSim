import math
import numpy as np
from numba import njit

# ─────────────────────────────────────────────────────────────────────────────
# Numba JIT Compiled LiDAR Raycasting
# ─────────────────────────────────────────────────────────────────────────────
@njit(fastmath=True)
def numba_synthetic_lidar(ego_pos, ego_yaw, obstacles_array, lidar_range, num_beams):
    """
    Ray-cast against circular obstacles to produce a LiDAR array.
    obstacles_array: (N, 3) float array where columns are [cx, cy, radius]
    """
    distances = np.full(num_beams, lidar_range, dtype=np.float32)
    beam_angles = np.arange(num_beams) * (2.0 * math.pi / num_beams)
    
    ex, ey = ego_pos[0], ego_pos[1]
    
    for bi in range(num_beams):
        world_angle = ego_yaw + beam_angles[bi]
        dx = math.cos(world_angle)
        dy = math.sin(world_angle)
        
        min_dist = lidar_range
        
        for i in range(obstacles_array.shape[0]):
            cx = obstacles_array[i, 0]
            cy = obstacles_array[i, 1]
            r = obstacles_array[i, 2]
            
            fx = cx - ex
            fy = cy - ey
            
            # Ray-circle intersection
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - r * r
            disc = b * b - 4.0 * c
            
            if disc >= 0:
                t = (-b - math.sqrt(disc)) / 2.0
                if 0.01 < t < min_dist:
                    min_dist = t
                    
        distances[bi] = min_dist
        
    return distances

# ─────────────────────────────────────────────────────────────────────────────
# Numba JIT Compiled Fossen RK4 Dynamics
# ─────────────────────────────────────────────────────────────────────────────
@njit(fastmath=True)
def _get_derivatives(eta, nu, tau, u_current, m, Iz, X_u_dot, Y_v_dot, N_r_dot, 
                     X_u, Y_v, N_r, X_u_abs_u, Y_v_abs_v, N_r_abs_r, M_inv):
    u, v, r = nu[0], nu[1], nu[2]
    psi = eta[2]
    
    c_psi = math.cos(psi)
    s_psi = math.sin(psi)
    
    # Rotation matrix R (3x3)
    R = np.array([
        [c_psi, -s_psi, 0.0],
        [s_psi,  c_psi, 0.0],
        [0.0,    0.0,   1.0]
    ])
    
    # Current to body frame
    u_c_body = c_psi * u_current[0] + s_psi * u_current[1]
    v_c_body = -s_psi * u_current[0] + c_psi * u_current[1]
    
    # Relative velocity
    u_r = u - u_c_body
    v_r = v - v_c_body
    
    m11 = m - X_u_dot
    m22 = m - Y_v_dot
    
    # Coriolis
    C = np.array([
        [0.0,          0.0,         -m22 * v_r],
        [0.0,          0.0,          m11 * u_r],
        [m22 * v_r,   -m11 * u_r,   0.0]
    ])
    
    # Damping
    D = np.array([
        [-X_u - X_u_abs_u * abs(u_r), 0.0, 0.0],
        [0.0, -Y_v - Y_v_abs_v * abs(v_r), 0.0],
        [0.0, 0.0, -N_r - N_r_abs_r * abs(r)]
    ])
    
    nu_rel = np.array([u_r, v_r, r])
    
    coriolis = np.dot(C, nu_rel)
    damping = np.dot(D, nu_rel)
    
    forces = tau - coriolis - damping
    nu_dot = np.dot(M_inv, forces)
    eta_dot = np.dot(R, nu)
    
    return eta_dot, nu_dot

@njit(fastmath=True)
def numba_rk4_step(eta, nu, tau, u_current, dt, m, Iz, X_u_dot, Y_v_dot, N_r_dot, 
                   X_u, Y_v, N_r, X_u_abs_u, Y_v_abs_v, N_r_abs_r, M_inv):
    """
    Step the dynamics forward in time using Runge-Kutta 4th Order (RK4).
    """
    args = (tau, u_current, m, Iz, X_u_dot, Y_v_dot, N_r_dot, X_u, Y_v, N_r, X_u_abs_u, Y_v_abs_v, N_r_abs_r, M_inv)
    
    eta_dot1, nu_dot1 = _get_derivatives(eta, nu, *args)
    
    eta2 = eta + 0.5 * dt * eta_dot1
    nu2 = nu + 0.5 * dt * nu_dot1
    eta_dot2, nu_dot2 = _get_derivatives(eta2, nu2, *args)
    
    eta3 = eta + 0.5 * dt * eta_dot2
    nu3 = nu + 0.5 * dt * nu_dot2
    eta_dot3, nu_dot3 = _get_derivatives(eta3, nu3, *args)
    
    eta4 = eta + dt * eta_dot3
    nu4 = nu + dt * nu_dot3
    eta_dot4, nu_dot4 = _get_derivatives(eta4, nu4, *args)
    
    new_eta = eta + (dt / 6.0) * (eta_dot1 + 2.0*eta_dot2 + 2.0*eta_dot3 + eta_dot4)
    new_nu = nu + (dt / 6.0) * (nu_dot1 + 2.0*nu_dot2 + 2.0*nu_dot3 + nu_dot4)
    
    # Normalize heading to [-pi, pi]
    new_eta[2] = (new_eta[2] + math.pi) % (2.0 * math.pi) - math.pi
    
    return new_eta, new_nu


# ─────────────────────────────────────────────────────────────────────────────
# Numba JIT Compiled 6-DOF AUV Fossen RK4 Dynamics
# ─────────────────────────────────────────────────────────────────────────────
@njit(fastmath=True)
def _get_auv_derivatives(eta, nu, tau, W, B, xg, yg, zg, xb, yb, zb, M_inv,
                         X_u, Y_v, Z_w, K_p, M_q, N_r,
                         X_u_abs_u, Y_v_abs_v, Z_w_abs_w,
                         K_p_abs_p, M_q_abs_q, N_r_abs_r):
    u, v, w, p, q, r = nu[0], nu[1], nu[2], nu[3], nu[4], nu[5]
    phi, theta, psi  = eta[3], eta[4], eta[5]
    
    c_phi = math.cos(phi)
    s_phi = math.sin(phi)
    c_theta = math.cos(theta)
    s_theta = math.sin(theta)
    t_theta = math.tan(theta)
    c_psi = math.cos(psi)
    s_psi = math.sin(psi)
    
    # Kinematic Transformation Matrix J (6x6)
    J = np.zeros((6, 6), dtype=np.float64)
    # J11 (Rotation)
    J[0,0] = c_psi * c_theta
    J[0,1] = -s_psi * c_phi + c_psi * s_theta * s_phi
    J[0,2] = s_psi * s_phi + c_psi * c_phi * s_theta
    J[1,0] = s_psi * c_theta
    J[1,1] = c_psi * c_phi + s_phi * s_theta * s_psi
    J[1,2] = -c_psi * s_phi + s_theta * s_psi * c_phi
    J[2,0] = -s_theta
    J[2,1] = c_theta * s_phi
    J[2,2] = c_theta * c_phi
    # J22 (Angular velocity transformation)
    J[3,3] = 1.0
    J[3,4] = s_phi * t_theta
    J[3,5] = c_phi * t_theta
    J[4,3] = 0.0
    J[4,4] = c_phi
    J[4,5] = -s_phi
    J[5,3] = 0.0
    J[5,4] = s_phi / c_theta
    J[5,5] = c_phi / c_theta
    
    # Restoring Forces (Buoyancy and Gravity)
    g_eta = np.zeros(6, dtype=np.float64)
    g_eta[0] = (W - B) * s_theta
    g_eta[1] = -(W - B) * c_theta * s_phi
    g_eta[2] = -(W - B) * c_theta * c_phi
    g_eta[3] = -(yg * W - yb * B) * c_theta * c_phi + (zg * W - zb * B) * c_theta * s_phi
    g_eta[4] = (zg * W - zb * B) * s_theta + (xg * W - xb * B) * c_theta * c_phi
    g_eta[5] = -(xg * W - xb * B) * c_theta * s_phi - (yg * W - yb * B) * s_theta
    
    # Damping (Linear + Quadratic diagonal approximation)
    D = np.zeros((6, 6), dtype=np.float64)
    D[0,0] = -X_u - X_u_abs_u * abs(u)
    D[1,1] = -Y_v - Y_v_abs_v * abs(v)
    D[2,2] = -Z_w - Z_w_abs_w * abs(w)
    D[3,3] = -K_p - K_p_abs_p * abs(p)
    D[4,4] = -M_q - M_q_abs_q * abs(q)
    D[5,5] = -N_r - N_r_abs_r * abs(r)
    damping = np.dot(D, nu)
    
    # Simplified Coriolis (Ignored in high speed surge-dominated AUVs for simplicity in this baseline)
    coriolis = np.zeros(6, dtype=np.float64)
    
    forces = tau - coriolis - damping - g_eta
    nu_dot = np.dot(M_inv, forces)
    eta_dot = np.dot(J, nu)
    
    return eta_dot, nu_dot

@njit(fastmath=True)
def numba_auv_rk4_step(eta, nu, tau, dt, W, B, xg, yg, zg, xb, yb, zb, M_inv,
                       X_u, Y_v, Z_w, K_p, M_q, N_r,
                       X_u_abs_u, Y_v_abs_v, Z_w_abs_w,
                       K_p_abs_p, M_q_abs_q, N_r_abs_r):
    
    args = (tau, W, B, xg, yg, zg, xb, yb, zb, M_inv,
            X_u, Y_v, Z_w, K_p, M_q, N_r,
            X_u_abs_u, Y_v_abs_v, Z_w_abs_w,
            K_p_abs_p, M_q_abs_q, N_r_abs_r)
    
    eta_dot1, nu_dot1 = _get_auv_derivatives(eta, nu, *args)
    
    eta2 = eta + 0.5 * dt * eta_dot1
    nu2 = nu + 0.5 * dt * nu_dot1
    eta_dot2, nu_dot2 = _get_auv_derivatives(eta2, nu2, *args)
    
    eta3 = eta + 0.5 * dt * eta_dot2
    nu3 = nu + 0.5 * dt * nu_dot2
    eta_dot3, nu_dot3 = _get_auv_derivatives(eta3, nu3, *args)
    
    eta4 = eta + dt * eta_dot3
    nu4 = nu + dt * nu_dot3
    eta_dot4, nu_dot4 = _get_auv_derivatives(eta4, nu4, *args)
    
    new_eta = eta + (dt / 6.0) * (eta_dot1 + 2.0*eta_dot2 + 2.0*eta_dot3 + eta_dot4)
    new_nu = nu + (dt / 6.0) * (nu_dot1 + 2.0*nu_dot2 + 2.0*nu_dot3 + nu_dot4)
    
    # Normalize heading, pitch, roll to [-pi, pi]
    new_eta[3] = (new_eta[3] + math.pi) % (2.0 * math.pi) - math.pi
    new_eta[4] = (new_eta[4] + math.pi) % (2.0 * math.pi) - math.pi
    new_eta[5] = (new_eta[5] + math.pi) % (2.0 * math.pi) - math.pi
    
    return new_eta, new_nu
