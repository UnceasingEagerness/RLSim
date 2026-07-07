import numpy as np
import math

class USVDynamics:
    """
    3-DOF Kinematic and Dynamic model for an underactuated USV.
    Based on Fossen's equations of motion for marine craft.
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        
        # State: eta = [x, y, psi]^T (position and heading in earth-fixed frame)
        self.eta = np.zeros(3)
        # Velocity: nu = [u, v, r]^T (surge, sway, yaw rates in body-fixed frame)
        self.nu = np.zeros(3)
        
        # Approximate parameters for a small USV (e.g., JT-30)
        self.m = 30.0    # mass (kg)
        self.Iz = 10.0   # moment of inertia about z-axis
        
        # Added mass (approximate)
        self.X_u_dot = -2.0
        self.Y_v_dot = -10.0
        self.N_r_dot = -1.0
        
        # Mass matrix M
        self.M = np.array([
            [self.m - self.X_u_dot, 0, 0],
            [0, self.m - self.Y_v_dot, 0],
            [0, 0, self.Iz - self.N_r_dot]
        ])
        self.M_inv = np.linalg.inv(self.M)
        self.m11 = self.M[0, 0]
        self.m22 = self.M[1, 1]
        
        # Linear damping (Fossen)
        self.X_u = -5.0
        self.Y_v = -10.0
        self.N_r = -2.0
        
        # Quadratic damping (Fossen)
        self.X_u_abs_u = -1.0
        self.Y_v_abs_v = -2.0
        self.N_r_abs_r = -0.5
        
        # Hydrodynamic parameters (from paper)
        self.rho = 1025.0  # fluid density (kg/m^3)
        self.A = 1.0       # frontal area (m^2)
        self.V = 0.1       # displaced volume (m^3)
        self.C_D = 0.5     # drag coefficient
        self.C_L = 0.1     # lift coefficient
        self.C_VM = 0.5    # virtual mass coefficient
        
        # Pre-allocate arrays to minimize garbage collection overhead
        self.C = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.R = np.zeros((3, 3))
        self.R[2, 2] = 1.0 # R[2,2] is always 1
        
        self.u_current = np.zeros(2) # [u_c, v_c] in earth frame
        self.u_current_dot = np.zeros(2) # derivative of current for virtual mass
        
    def reset(self, initial_eta=None, initial_nu=None):
        """Reset the state of the USV."""
        self.eta = np.array(initial_eta, dtype=float) if initial_eta is not None else np.zeros(3)
        self.nu = np.array(initial_nu, dtype=float) if initial_nu is not None else np.zeros(3)
        
    def _get_derivatives(self, eta, nu, tau):
        """Calculate state derivatives eta_dot and nu_dot given current state and inputs."""
        u, v, r = nu
        
        # Update Rotation matrix R(psi)
        psi = eta[2]
        c_psi = math.cos(psi)
        s_psi = math.sin(psi)
        
        self.R[0, 0] = c_psi
        self.R[0, 1] = -s_psi
        self.R[1, 0] = s_psi
        self.R[1, 1] = c_psi
        
        # Transform ocean current to body frame
        u_c_body = c_psi * self.u_current[0] + s_psi * self.u_current[1]
        v_c_body = -s_psi * self.u_current[0] + c_psi * self.u_current[1]
        
        # Relative velocity
        u_r = u - u_c_body
        v_r = v - v_c_body
        U_rel = math.sqrt(u_r**2 + v_r**2)
        
        # Update Coriollis and centripetal matrix C(nu_r) - using relative velocity
        self.C[0, 2] = -self.m22 * v_r
        self.C[1, 2] = self.m11 * u_r
        self.C[2, 0] = self.m22 * v_r
        self.C[2, 1] = -self.m11 * u_r
        
        # Update Damping matrix D(nu_r) - using relative velocity (FOSSEN METHOD)
        self.D[0, 0] = -self.X_u - self.X_u_abs_u * abs(u_r)
        self.D[1, 1] = -self.Y_v - self.Y_v_abs_v * abs(v_r)
        self.D[2, 2] = -self.N_r - self.N_r_abs_r * abs(r)
        
        # --- EXPLICIT HYDRODYNAMIC FORCES (PAPER METHOD) ---
        # uncomment to use instead of Fossen D(nu) damping
        # F_D_x = -0.5 * self.rho * self.C_D * self.A * U_rel * u_r
        # F_D_y = -0.5 * self.rho * self.C_D * self.A * U_rel * v_r
        # F_L_x = -0.5 * self.rho * self.C_L * self.A * U_rel * (-v_r)
        # F_L_y = -0.5 * self.rho * self.C_L * self.A * U_rel * (u_r)
        # u_c_dot_body = c_psi * self.u_current_dot[0] + s_psi * self.u_current_dot[1]
        # v_c_dot_body = -s_psi * self.u_current_dot[0] + c_psi * self.u_current_dot[1]
        # F_VM_x = self.rho * self.C_VM * self.V * u_c_dot_body
        # F_VM_y = self.rho * self.C_VM * self.V * v_c_dot_body
        # N_drag = self.N_r * r + self.N_r_abs_r * abs(r) * r
        # F_hydro = np.array([F_D_x + F_L_x + F_VM_x, F_D_y + F_L_y + F_VM_y, N_drag])
        
        # Calculate nu_dot using FOSSEN METHOD
        coriolis = self.C.dot(np.array([u_r, v_r, r]))
        damping = self.D.dot(np.array([u_r, v_r, r]))
        nu_dot = self.M_inv.dot(tau - coriolis - damping)
        
        # If using PAPER METHOD, calculate nu_dot like this:
        # nu_dot = self.M_inv.dot(tau + F_hydro - coriolis)
        
        # Calculate eta_dot
        eta_dot = self.R.dot(nu)
        
        return eta_dot, nu_dot

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
            
        from envs.numba_utils import numba_rk4_step
        
        self.eta, self.nu = numba_rk4_step(
            eta=np.array(self.eta, dtype=np.float64),
            nu=np.array(self.nu, dtype=np.float64),
            tau=tau,
            u_current=self.u_current,
            dt=self.dt,
            m=self.m, Iz=self.Iz,
            X_u_dot=self.X_u_dot, Y_v_dot=self.Y_v_dot, N_r_dot=self.N_r_dot,
            X_u=self.X_u, Y_v=self.Y_v, N_r=self.N_r,
            X_u_abs_u=self.X_u_abs_u, Y_v_abs_v=self.Y_v_abs_v, N_r_abs_r=self.N_r_abs_r,
            M_inv=self.M_inv
        )
        
        return self.eta.copy(), self.nu.copy()
    
    def get_earth_velocity(self):
        """Returns [v_x, v_y] in earth coordinates."""
        psi = self.eta[2]
        c_psi = math.cos(psi)
        s_psi = math.sin(psi)
        # We can optimize this by avoiding new array allocation
        v_x = c_psi * self.nu[0] - s_psi * self.nu[1]
        v_y = s_psi * self.nu[0] + c_psi * self.nu[1]
        return np.array([v_x, v_y])
