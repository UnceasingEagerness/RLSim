import numpy as np
import math

class AUVDynamics:
    """
    6-DOF Fossen Dynamics Model for a Torpedo-shaped Autonomous Underwater Vehicle (AUV).
    Based on standard REMUS 100 hydrodynamic coefficients.
    
    State Vector (12D):
        eta = [x, y, z, phi, theta, psi]  (Earth-fixed position and orientation)
        nu  = [u, v, w, p, q, r]          (Body-fixed linear and angular velocities)
    
    Control Vector (4D):
        tau = [X, Y, Z, K, M, N]
        (Typically controlled via Surge Thrust, Heave Thrust, Pitch Fin, Yaw Fin)
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        
        # 1. Rigid Body Parameters (REMUS 100 approx)
        self.W = 30.0 * 9.81  # Weight (N)
        self.B = 30.5 * 9.81  # Buoyancy (N) - Slightly positively buoyant
        self.m = 30.0         # Mass (kg)
        
        self.Ixx = 0.177
        self.Iyy = 3.45
        self.Izz = 3.45
        
        # Center of Gravity & Buoyancy (Body frame coordinates in meters)
        self.xg = 0.0
        self.yg = 0.0
        self.zg = 0.02
        
        self.xb = 0.0
        self.yb = 0.0
        self.zb = 0.0
        
        # 2. Added Mass Matrix (Diagonal approximation)
        self.X_u_dot = -0.93
        self.Y_v_dot = -35.5
        self.Z_w_dot = -35.5
        self.K_p_dot = -0.0704
        self.M_q_dot = -4.88
        self.N_r_dot = -4.88
        
        # Total Mass Matrix (Rigid + Added)
        self.M = np.diag([
            self.m - self.X_u_dot,
            self.m - self.Y_v_dot,
            self.m - self.Z_w_dot,
            self.Ixx - self.K_p_dot,
            self.Iyy - self.M_q_dot,
            self.Izz - self.N_r_dot
        ])
        self.M_inv = np.linalg.inv(self.M)
        
        # 3. Linear Damping Matrix
        self.X_u = -0.074
        self.Y_v = -27.0
        self.Z_w = -27.0
        self.K_p = -0.13
        self.M_q = -17.0
        self.N_r = -17.0
        
        # 4. Quadratic Damping Matrix
        self.X_u_abs_u = -1.62
        self.Y_v_abs_v = -131.0
        self.Z_w_abs_w = -131.0
        self.K_p_abs_p = -0.013
        self.M_q_abs_q = -170.0
        self.N_r_abs_r = -170.0
        
        # State Initialization
        self.eta = np.zeros(6, dtype=np.float64)
        self.nu  = np.zeros(6, dtype=np.float64)
        
        # Ocean currents (Body frame)
        self.u_current = np.zeros(6, dtype=np.float64)
        self.u_current_dot = np.zeros(6, dtype=np.float64)

    def reset(self, initial_eta=None, initial_nu=None):
        if initial_eta is not None:
            self.eta = np.array(initial_eta, dtype=np.float64)
        else:
            self.eta = np.zeros(6, dtype=np.float64)
            
        if initial_nu is not None:
            self.nu = np.array(initial_nu, dtype=np.float64)
        else:
            self.nu = np.zeros(6, dtype=np.float64)

    def _J_matrix(self, eta):
        """Transformation matrix from Body to Earth frame."""
        phi, theta, psi = eta[3], eta[4], eta[5]
        
        c_phi = math.cos(phi)
        s_phi = math.sin(phi)
        c_theta = math.cos(theta)
        s_theta = math.sin(theta)
        t_theta = math.tan(theta)
        c_psi = math.cos(psi)
        s_psi = math.sin(psi)
        
        J11 = np.array([
            [c_psi * c_theta, -s_psi * c_phi + c_psi * s_theta * s_phi,  s_psi * s_phi + c_psi * c_phi * s_theta],
            [s_psi * c_theta,  c_psi * c_phi + s_phi * s_theta * s_psi, -c_psi * s_phi + s_theta * s_psi * c_phi],
            [-s_theta,         c_theta * s_phi,                          c_theta * c_phi]
        ], dtype=np.float64)
        
        J22 = np.array([
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi,          -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta]
        ], dtype=np.float64)
        
        J = np.zeros((6, 6), dtype=np.float64)
        J[0:3, 0:3] = J11
        J[3:6, 3:6] = J22
        return J

    def _get_restoring_forces(self, eta):
        """Hydrostatic restoring forces (Buoyancy and Gravity)."""
        phi, theta, _ = eta[3], eta[4], eta[5]
        
        c_phi = math.cos(phi)
        s_phi = math.sin(phi)
        c_theta = math.cos(theta)
        s_theta = math.sin(theta)
        
        g_eta = np.zeros(6, dtype=np.float64)
        
        g_eta[0] = (self.W - self.B) * s_theta
        g_eta[1] = -(self.W - self.B) * c_theta * s_phi
        g_eta[2] = -(self.W - self.B) * c_theta * c_phi
        g_eta[3] = -(self.yg * self.W - self.yb * self.B) * c_theta * c_phi + (self.zg * self.W - self.zb * self.B) * c_theta * s_phi
        g_eta[4] = (self.zg * self.W - self.zb * self.B) * s_theta + (self.xg * self.W - self.xb * self.B) * c_theta * c_phi
        g_eta[5] = -(self.xg * self.W - self.xb * self.B) * c_theta * s_phi - (self.yg * self.W - self.yb * self.B) * s_theta
        
        return g_eta

    def step(self, tau, u_current=None, u_current_dot=None):
        """
        Step the dynamics forward in time using Runge-Kutta 4th Order (RK4).
        Accelerated using Numba JIT Compilation.
        """
        tau = np.array(tau, dtype=np.float64)
        if u_current is not None:
            self.u_current = np.array(u_current, dtype=np.float64)
        if u_current_dot is not None:
            self.u_current_dot = np.array(u_current_dot, dtype=np.float64)
            
        from envs.numba_utils import numba_auv_rk4_step
        
        self.eta, self.nu = numba_auv_rk4_step(
            eta=np.array(self.eta, dtype=np.float64),
            nu=np.array(self.nu, dtype=np.float64),
            tau=tau,
            dt=self.dt,
            W=self.W, B=self.B,
            xg=self.xg, yg=self.yg, zg=self.zg,
            xb=self.xb, yb=self.yb, zb=self.zb,
            M_inv=self.M_inv,
            X_u=self.X_u, Y_v=self.Y_v, Z_w=self.Z_w,
            K_p=self.K_p, M_q=self.M_q, N_r=self.N_r,
            X_u_abs_u=self.X_u_abs_u, Y_v_abs_v=self.Y_v_abs_v, Z_w_abs_w=self.Z_w_abs_w,
            K_p_abs_p=self.K_p_abs_p, M_q_abs_q=self.M_q_abs_q, N_r_abs_r=self.N_r_abs_r
        )
        
        return self.eta.copy(), self.nu.copy()
