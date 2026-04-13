# AER1513 - Assignment 2 (Q4 + Q5)
# Extended Kalman Filter for "Lost in the Woods" dataset
# Single-file implementation: numpy arrays only; matplotlib for plots & animation.
# DELIVERABLES:
#   - Q4(a): 9 PNG plots (r_max = 5, 3, 1 for x/y/theta)
#   - Q4(b): 9 PNG plots with poor init (suffix "_badinit")
#   - Q4(c): 9 PNG plots with CRLB-style linearization (suffix "_crlb")
#   - Q5   : MP4 animation for r_max=1 (ffmpeg only)
#
# IMPORTANT FIX for Q4(c):
#   Jacobians F, G, H are evaluated at the ground-truth state,
#   BUT the predicted measurement y_hat is ALWAYS evaluated at the current predicted state.
#   This preserves informative innovations in the "CRLB" reference run.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import scipy.io

# =============================================================================
# Configuration
# =============================================================================
# <-- CHANGE THIS PATH IF NEEDED -->
DATASET_PATH = r"C:\Users\89286\Downloads\dataset2.mat"

SAVE_PNG_DPI = 120
ANIM_INTERVAL_MS = 50
R_MAX_LIST = [5.0, 3.0, 1.0]

# Robust y-axis settings for error plots (no filename changes)
T_CLIP_SEC_FOR_YLIM = 60.0   # set 0.0 to disable time clipping when estimating y-limits
YLIM_PERCENTILE = 100       # 95~99.5 typical; higher => 更聚焦（可能截掉极少数尖峰）

# Toggles
RUN_Q4A = True          # good init
RUN_Q4B_POOR = True     # poor init: x0 = (1,1,0.1)
RUN_Q4C_CRLB = True     # CRLB-style: Jacobians at truth, y_hat at predicted state
RUN_Q5_ANIM = True      # animation for r_max=1

# =============================================================================
# Utilities
# =============================================================================
def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]. Works with scalars or numpy arrays."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

# =============================================================================
# Models & Jacobians (use G_k for process/input noise Jacobian, per your notation)
# =============================================================================
def motion_model(x_prev: np.ndarray, u: np.ndarray, T: float) -> np.ndarray:
    """Unicycle kinematics (Euler discretization)."""
    x, y, th = x_prev
    v_k, om_k = u
    x_n = x + T * v_k * np.cos(th)
    y_n = y + T * v_k * np.sin(th)
    th_n = wrap_angle(th + T * om_k)
    return np.array([x_n, y_n, th_n], dtype=float)

def motion_jacobian_F(x_prev: np.ndarray, u: np.ndarray, T: float) -> np.ndarray:
    """F_k-1 = ∂f/∂x at (x_prev, u)."""
    th = x_prev[2]
    v_k = u[0]
    F = np.array([
        [1.0, 0.0, -T * v_k * np.sin(th)],
        [0.0, 1.0,  T * v_k * np.cos(th)],
        [0.0, 0.0,  1.0]
    ], dtype=float)
    return F

def process_noise_jacobian_G(x_prev: np.ndarray, T: float) -> np.ndarray:
    """G_k-1 = ∂f/∂w mapping control noise [v, om] into state space."""
    th = x_prev[2]
    G = np.array([
        [T * np.cos(th), 0.0],
        [T * np.sin(th), 0.0],
        [0.0,            T  ]
    ], dtype=float)
    return G

def observation_model(x: np.ndarray, landmark_xy: np.ndarray, d: float) -> np.ndarray:
    """
    Range-bearing sensor with forward offset d (sensor ahead of robot center).
    y = [range; bearing(relative to heading)].
    """
    dx = landmark_xy[0] - (x[0] + d * np.cos(x[2]))
    dy = landmark_xy[1] - (x[1] + d * np.sin(x[2]))
    q = dx * dx + dy * dy
    r_pred = np.sqrt(max(q, 1e-12))
    b_pred = wrap_angle(np.arctan2(dy, dx) - x[2])
    return np.array([r_pred, b_pred], dtype=float)

def observation_jacobian_H(x: np.ndarray, landmark_xy: np.ndarray, d: float) -> np.ndarray:
    """H_k = ∂h/∂x (2x3) for range-bearing with forward offset d."""
    dx = landmark_xy[0] - (x[0] + d * np.cos(x[2]))
    dy = landmark_xy[1] - (x[1] + d * np.sin(x[2]))
    q = dx * dx + dy * dy
    r = np.sqrt(max(q, 1e-12))
    inv_q = 1.0 / max(q, 1e-12)

    H = np.zeros((2, 3), dtype=float)
    H[0, 0] = -dx / r
    H[0, 1] = -dy / r
    H[1, 0] =  dy * inv_q
    H[1, 1] = -dx * inv_q
    # d-dependent terms from sensor offset
    H[0, 2] = (dx * d * np.sin(x[2]) - dy * d * np.cos(x[2])) / r
    H[1, 2] = -1.0 - ((dx * d * np.cos(x[2]) + dy * d * np.sin(x[2])) * inv_q)
    return H

# =============================================================================
# Load dataset
# =============================================================================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"dataset2.mat not found at: {DATASET_PATH}")

data = scipy.io.loadmat(DATASET_PATH)

# Extract arrays safely
t = data["t"].ravel().astype(float)
r = np.array(data["r"], dtype=float)         # (N, M) ranges
b = np.array(data["b"], dtype=float)         # (N, M) bearings
l = np.array(data["l"], dtype=float)         # landmarks: (M, 2) or (2, M)
if l.shape[0] == 2 and l.shape[1] != 2:
    l = l.T  # ensure (M, 2)

d = float(np.array(data["d"]).squeeze())
v = data["v"].ravel().astype(float)          # (N,)
om = data["om"].ravel().astype(float)        # (N,)
r_var = float(np.array(data["r_var"]).squeeze())
b_var = float(np.array(data["b_var"]).squeeze())
v_var = float(np.array(data["v_var"]).squeeze())
om_var = float(np.array(data["om_var"]).squeeze())

# Ground truth for evaluation
x_true = data["x_true"].ravel().astype(float)
y_true = data["y_true"].ravel().astype(float)
th_true = data["th_true"].ravel().astype(float)

N = len(t)
M = l.shape[0]
T = 0.1

# =============================================================================
# EKF runner
# =============================================================================
def run_ekf(r_max_list, x0, P0, linearize_at="estimate", suffix="", save_prefix=""):
    """
    Run EKF for all r_max in r_max_list.
    linearize_at: "estimate" (standard EKF) or "truth" (CRLB-style Jacobians).
      - For "truth": Evaluate F, G at x_true[k-1]; H at x_true[k].
        BUT predicted measurement y_hat is ALWAYS evaluated at x_pred.
    suffix: filename suffix for saved plots.
    Returns dict { r_max: (x_est, P_est) }.
    """
    Q_control = np.array([[v_var, 0.0],
                          [0.0,   om_var]], dtype=float)
    R_single = np.array([[r_var, 0.0],
                         [0.0,   b_var]], dtype=float)

    results = {}
    for r_max in r_max_list:
        x_est = np.zeros((N, 3), dtype=float)
        P_est = np.zeros((N, 3, 3), dtype=float)
        x_est[0, :] = x0
        P_est[0, :, :] = P0

        for k in range(1, N):
            # ---------- Prediction ----------
            u_km1 = np.array([v[k - 1], om[k - 1]], dtype=float)
            x_prev = x_est[k - 1, :]
            P_prev = P_est[k - 1, :, :]

            # propagate mean at estimated previous state
            x_pred = motion_model(x_prev, u_km1, T)

            # Jacobians for covariance propagation
            if linearize_at == "truth":
                x_lin_km1 = np.array([x_true[k - 1], y_true[k - 1], th_true[k - 1]])
                F_km1 = motion_jacobian_F(x_lin_km1, u_km1, T)
                G_km1 = process_noise_jacobian_G(x_lin_km1, T)
            else:
                F_km1 = motion_jacobian_F(x_prev, u_km1, T)
                G_km1 = process_noise_jacobian_G(x_prev, T)

            Q_k = G_km1 @ Q_control @ G_km1.T
            P_pred = F_km1 @ P_prev @ F_km1.T + Q_k

            # ---------- Measurement update ----------
            y_blocks = []
            y_hat_blocks = []
            H_blocks = []

            r_k = r[k, :]
            b_k = b[k, :]

            for j in range(M):
                r_lk = r_k[j]
                b_lk = b_k[j]
                if r_lk > 0.0 and r_lk < r_max:
                    lm = l[j, :]
                    y_meas = np.array([r_lk, b_lk], dtype=float)

                    # predicted measurement ALWAYS at x_pred
                    y_hat = observation_model(x_pred, lm, d)

                    if linearize_at == "truth":
                        x_lin_k = np.array([x_true[k], y_true[k], th_true[k]])
                        H_k = observation_jacobian_H(x_lin_k, lm, d)
                    else:
                        H_k = observation_jacobian_H(x_pred, lm, d)

                    y_blocks.append(y_meas)
                    y_hat_blocks.append(y_hat)
                    H_blocks.append(H_k)

            if len(y_blocks) > 0:
                Y = np.vstack(y_blocks)           # (m, 2)
                Y_hat = np.vstack(y_hat_blocks)   # (m, 2)
                H = np.vstack(H_blocks)           # (2m, 3)
                m = Y.shape[0]
                R_block = np.kron(np.eye(m), R_single)

                residual = Y - Y_hat
                # wrap bearing residuals
                residual[:, 1] = wrap_angle(residual[:, 1])
                res_vec = residual.reshape(-1)

                S = H @ P_pred @ H.T + R_block
                K = P_pred @ H.T @ np.linalg.inv(S)

                x_upd = x_pred + K @ res_vec
                x_upd[2] = wrap_angle(x_upd[2])
                I = np.eye(3)
                P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R_block @ K.T
            else:
                x_upd, P_upd = x_pred, P_pred

            x_est[k, :] = x_upd
            P_est[k, :, :] = P_upd

        # ---------- Save plots for this r_max ----------
        results[r_max] = (x_est, P_est)

        err_x = x_est[:, 0] - x_true
        err_y = x_est[:, 1] - y_true
        err_th = wrap_angle(x_est[:, 2] - th_true)

        sig_x = 3.0 * np.sqrt(np.maximum(P_est[:, 0, 0], 0.0))
        sig_y = 3.0 * np.sqrt(np.maximum(P_est[:, 1, 1], 0.0))
        sig_th = 3.0 * np.sqrt(np.maximum(P_est[:, 2, 2], 0.0))

        # -------- Robust error plotting helper (PERCENTILE-BASED Y-LIMITS) --------
        def save_err(name, e, s, ylab):
            """
            Robust y-limit plotting WITHOUT changing filenames:
              - Use |e| and s percentiles (YLIM_PERCENTILE) to avoid rare spikes
              - Optionally ignore the first T_CLIP_SEC_FOR_YLIM seconds when estimating y-limits
              - Saved filename remains: {save_prefix}{name}_error_rmax{int(r_max)}{suffix}.png
            """
            # indices considered for estimating y-limits
            if T_CLIP_SEC_FOR_YLIM > 0.0:
                idx = t >= T_CLIP_SEC_FOR_YLIM
            else:
                idx = np.ones_like(t, dtype=bool)

            e_use = e[idx] if np.any(idx) else e
            s_use = s[idx] if np.any(idx) else s

            p_e = float(np.percentile(np.abs(e_use), YLIM_PERCENTILE)) if e_use.size else 0.0
            p_s = float(np.percentile(s_use,          YLIM_PERCENTILE)) if s_use.size else 0.0
            y_lim = 1.10 * max(p_e, p_s, 1e-6)  # small margin

            plt.figure(figsize=(9, 3.6))
            plt.plot(t, e, label='Error (estimate - truth)')
            plt.plot(t,  s, 'r--', label='±3σ')
            plt.plot(t, -s, 'r--')
            plt.ylim(-y_lim, y_lim)

            plt.xlabel('Time [s]')
            plt.ylabel(ylab)
            plt.title(f'{name} error vs time (r_max={r_max})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            fn = f'{save_prefix}{name}_error_rmax{int(r_max)}{suffix}.png'
            plt.savefig(fn, dpi=SAVE_PNG_DPI)
            plt.close()

        save_err('x',     err_x,  sig_x,  'x error [m]')
        save_err('y',     err_y,  sig_y,  'y error [m]')
        save_err('theta', err_th, sig_th, 'heading error [rad]')

    return results

# =============================================================================
# Q4(a): good initial condition
# =============================================================================
x0_good = np.array([x_true[0], y_true[0], th_true[0]], dtype=float)
P0_good = np.diag([1.0, 1.0, 0.1])

results_good = {}
if RUN_Q4A:
    results_good = run_ekf(R_MAX_LIST, x0_good, P0_good,
                           linearize_at="estimate", suffix="", save_prefix="")

# =============================================================================
# Q4(b): poor initial condition
# =============================================================================
if RUN_Q4B_POOR:
    x0_bad = np.array([1.0, 1.0, 0.1], dtype=float)  # per spec
    _ = run_ekf(R_MAX_LIST, x0_bad, P0_good,
                linearize_at="estimate", suffix="_badinit", save_prefix="")

# =============================================================================
# Q4(c): CRLB-style (Jacobians at truth; y_hat at predicted state)
# =============================================================================
if RUN_Q4C_CRLB:
    _ = run_ekf(R_MAX_LIST, x0_good, P0_good,
                linearize_at="truth", suffix="_crlb", save_prefix="")

# =============================================================================
# Q5: Animation for r_max = 1 (good init), colors per spec (MP4 via ffmpeg)
# =============================================================================
if RUN_Q5_ANIM:
    # Ensure we have the Q4(a) result for r_max=1
    if 1.0 not in (results_good or {}):
        results_good = run_ekf([1.0], x0_good, P0_good,
                               linearize_at="estimate", suffix="", save_prefix="")
    x_est_anim, P_est_anim = results_good[1.0]

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    # Landmarks: black, static
    ax.scatter(l[:, 0], l[:, 1], c='k', marker='.', label='Landmarks')

    # Optional paths
    line_true, = ax.plot([], [], '-', color='tab:blue', lw=1.2, alpha=0.6, label='GT path')
    line_est,  = ax.plot([], [], '-', color='tab:red',  lw=1.2, alpha=0.6, label='EKF path')

    # Moving dots (blue: truth, red: EKF)
    point_true, = ax.plot([], [], 'o', color='tab:blue', ms=5, label='GT pos')
    point_est,  = ax.plot([], [], 'o', color='tab:red',  ms=5, label='EKF pos')

    # 3-sigma covariance ellipse (red)
    ellipse = Ellipse(xy=(0, 0), width=0.0, height=0.0, angle=0.0,
                      edgecolor='tab:red', linestyle='--', fill=False, lw=1.8)
    ax.add_patch(ellipse)

    # Axes limits
    all_x = np.concatenate([x_true, x_est_anim[:, 0], l[:, 0]])
    all_y = np.concatenate([y_true, x_est_anim[:, 1], l[:, 1]])
    margin = 1.0
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('EKF animation (r_max=1; blue: GT, red: estimate)')
    ax.legend(loc='upper right')

    def update(frame: int):
        i = frame
        # Paths
        line_true.set_data(x_true[:i+1], y_true[:i+1])
        line_est.set_data(x_est_anim[:i+1, 0], x_est_anim[:i+1, 1])
        # Points
        point_true.set_data([x_true[i]], [y_true[i]])
        point_est.set_data([x_est_anim[i, 0]], [x_est_anim[i, 1]])
        # 2x2 covariance -> 3-sigma ellipse
        P_xy = P_est_anim[i, 0:2, 0:2]
        evals, evecs = np.linalg.eigh(P_xy)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        evecs = evecs[:, order]
        width = 2.0 * 3.0 * np.sqrt(evals[0])
        height = 2.0 * 3.0 * np.sqrt(evals[1])
        angle_deg = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        ellipse.set_center((x_est_anim[i, 0], x_est_anim[i, 1]))
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle_deg
        return line_true, line_est, point_true, point_est, ellipse

    anim = FuncAnimation(fig, update, frames=N, interval=ANIM_INTERVAL_MS, blit=True)

    # Save MP4 via ffmpeg only
    fps_val = max(1, int(round(1.0 / T))) if T > 0 else 10
    anim.save('ekf_path_animation.mp4', writer='ffmpeg', fps=fps_val)
    print('Saved MP4: ekf_path_animation.mp4')
    plt.close(fig)

# ============================== End of file ==================================
