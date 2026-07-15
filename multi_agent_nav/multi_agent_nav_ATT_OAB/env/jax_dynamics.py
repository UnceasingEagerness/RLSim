import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class USVParams:
    """Hydrodynamic coefficients and parameters for a USV."""
    m: float = 23.8
    Iz: float = 1.76
    X_u_dot: float = -2.0
    Y_v_dot: float = -10.0
    N_r_dot: float = -1.0
    X_u: float = -0.72253
    Y_v: float = -0.88965
    N_r: float = -1.9
    X_u_abs_u: float = -1.32742
    Y_v_abs_v: float = -36.47287
    N_r_abs_r: float = -0.75
    dt: float = 0.1

@struct.dataclass
class USVState:
    """State vector for the USV."""
    eta: jnp.ndarray  # [x, y, psi] (earth frame)
    nu: jnp.ndarray   # [u, v, r]   (body frame)

@jax.jit
def get_mass_matrix(params: USVParams) -> jnp.ndarray:
    m11 = params.m - params.X_u_dot
    m22 = params.m - params.Y_v_dot
    m33 = params.Iz - params.N_r_dot
    M = jnp.array([
        [m11, 0.0, 0.0],
        [0.0, m22, 0.0],
        [0.0, 0.0, m33]
    ])
    return M

@jax.jit
def get_derivatives(state: USVState, tau: jnp.ndarray, params: USVParams, M_inv: jnp.ndarray, u_current: jnp.ndarray = jnp.zeros(3)) -> tuple:
    u, v, r = state.nu[0], state.nu[1], state.nu[2]
    psi = state.eta[2]
    
    c_psi = jnp.cos(psi)
    s_psi = jnp.sin(psi)
    
    # Rotation matrix R (3x3)
    R = jnp.array([
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
    
    m11 = params.m - params.X_u_dot
    m22 = params.m - params.Y_v_dot
    
    # Coriolis
    C = jnp.array([
        [0.0,          0.0,         -m22 * v_r],
        [0.0,          0.0,          m11 * u_r],
        [m22 * v_r,   -m11 * u_r,   0.0]
    ])
    
    # Damping
    D = jnp.array([
        [-params.X_u - params.X_u_abs_u * jnp.abs(u_r), 0.0, 0.0],
        [0.0, -params.Y_v - params.Y_v_abs_v * jnp.abs(v_r), 0.0],
        [0.0, 0.0, -params.N_r - params.N_r_abs_r * jnp.abs(r)]
    ])
    
    nu_rel = jnp.array([u_r, v_r, r])
    
    coriolis = jnp.dot(C, nu_rel)
    damping = jnp.dot(D, nu_rel)
    
    forces = tau - coriolis - damping
    nu_dot = jnp.dot(M_inv, forces)
    eta_dot = jnp.dot(R, state.nu)
    
    return eta_dot, nu_dot

@jax.jit
def rk4_step(state: USVState, tau: jnp.ndarray, params: USVParams) -> USVState:
    """
    Step the dynamics forward in time using Runge-Kutta 4th Order (RK4).
    Fully compiled by XLA for execution on the GPU/TPU.
    """
    M_inv = jnp.linalg.inv(get_mass_matrix(params))
    
    eta_dot1, nu_dot1 = get_derivatives(state, tau, params, M_inv)
    
    state2 = USVState(eta=state.eta + 0.5 * params.dt * eta_dot1, nu=state.nu + 0.5 * params.dt * nu_dot1)
    eta_dot2, nu_dot2 = get_derivatives(state2, tau, params, M_inv)
    
    state3 = USVState(eta=state.eta + 0.5 * params.dt * eta_dot2, nu=state.nu + 0.5 * params.dt * nu_dot2)
    eta_dot3, nu_dot3 = get_derivatives(state3, tau, params, M_inv)
    
    state4 = USVState(eta=state.eta + params.dt * eta_dot3, nu=state.nu + params.dt * nu_dot3)
    eta_dot4, nu_dot4 = get_derivatives(state4, tau, params, M_inv)
    
    new_eta = state.eta + (params.dt / 6.0) * (eta_dot1 + 2.0*eta_dot2 + 2.0*eta_dot3 + eta_dot4)
    new_nu = state.nu + (params.dt / 6.0) * (nu_dot1 + 2.0*nu_dot2 + 2.0*nu_dot3 + nu_dot4)
    
    # Normalize heading to [-pi, pi]
    new_heading = (new_eta[2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    new_eta = new_eta.at[2].set(new_heading)
    
    return USVState(eta=new_eta, nu=new_nu)
