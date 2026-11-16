
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from baseline_model import (
    competition_model,
    calculate_equilibria,
    calculate_stability,
    symbolic_fixed_point_analysis,
)

# ==========================================
# 1. MODEL PARAMETERS
# ==========================================

# Growth rates (per hour)
r_s = 0.4
r_p = 0.3

# Carrying capacity ratios
K_ratios = [1, 10, 100, 1000, 10000]

# Competition coefficients
alpha_sp = 5.893
alpha_ps = 1.860

os.makedirs("figures", exist_ok=True)

# ==========================================
# 2. SYMBOLIC FIXED-POINT ANALYSIS
# ==========================================

# Run at base parameters (K_s=1, K_p=1)
symbolic_fixed_point_analysis({
    "r_s":      r_s,
    "r_p":      r_p,
    "K_s":      1.0,
    "K_p":      1.0,
    "alpha_sp": alpha_sp,
    "alpha_ps": alpha_ps,
})



# ==========================================
# 3. CALCULATE EQUILIBRIA
# ==========================================

for K_ratio in K_ratios:

    K_s = 1.0
    K_p = K_ratio

    equilibria = calculate_equilibria(
        K_s, K_p, alpha_sp, alpha_ps
    )

    print("\nCARRYING CAPACITY RATIO:", K_ratio)
    print("--------------------------------")

    for name, values in equilibria.items():

        if values is None:
            print(f"{name}: Not biologically feasible")

        else:
            S, P = values

            print(
                f"{name}: SA = {S:.4f}, "
                f"PA = {P:.4f}"
            )


# ==========================================
# 4. EQUILIBRIUM STABILITY
# ==========================================

    print("\nEQUILIBRIUM STABILITY")
    print("--------------------------------")

    for name, values in equilibria.items():

        if values is None:
            continue

        S, P = values

        eigenvalues, stability = calculate_stability(
            S, P, r_s, r_p, K_s, K_p,
            alpha_sp, alpha_ps
        )

        print(f"\n{name}")
        print("Eigenvalues:", eigenvalues)
        print("Stability:", stability)

# ==========================================
# 6. PHASE PORTRAIT (K_s=1, K_p=1)
# ==========================================

r_s_n = r_s
r_p_n = r_p
K_s_n = 1.0
K_p_n = 1.0
alpha_sp_n = alpha_sp
alpha_ps_n = alpha_ps

S_star_n = (
    (K_s_n - alpha_sp_n * K_p_n)
    / (1 - alpha_sp_n * alpha_ps_n)
)
P_star_n = (
    (K_p_n - alpha_ps_n * K_s_n)
    / (1 - alpha_sp_n * alpha_ps_n)
)

fixed_pts = {
    "(0,0)":   (0,        0),
    "(Ks,0)":  (K_s_n,    0),
    "(0,Kp)":  (0,        K_p_n),
    "(S*,P*)": (S_star_n, P_star_n),
}

S_vals = np.linspace(-0.2, 1.5, 300)
P_vals = np.linspace(-0.2, 1.5, 300)
S_grid, P_grid = np.meshgrid(S_vals, P_vals)

dS = r_s_n * S_grid * (
    1 - (S_grid + alpha_sp_n * P_grid) / K_s_n
)
dP = r_p_n * P_grid * (
    1 - (P_grid + alpha_ps_n * S_grid) / K_p_n
)

magnitude = np.sqrt(dS**2 + dP**2)
magnitude[magnitude == 0] = 1

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Nullclines
P_null_S = (K_s_n - S_vals) / alpha_sp_n
P_null_P = K_p_n - alpha_ps_n * S_vals

fp_colors = {
    "(0,0)":   "white",
    "(Ks,0)":  "lime",
    "(0,Kp)":  "cyan",
    "(S*,P*)": "yellow",
}

# --- Plot 1: Streamplot ---
ax1 = axes[0]
ax1.streamplot(
    S_vals, P_vals, dS, dP,
    color=magnitude, cmap="plasma",
    linewidth=0.8, density=1.5
)

ax1.plot(S_vals, P_null_S, "b--",
         label="S nullcline", linewidth=1.5)
ax1.plot(S_vals, P_null_P, "r--",
         label="P nullcline", linewidth=1.5)

for label, (sx, px) in fixed_pts.items():
    ax1.scatter(
        sx, px, s=80, zorder=5,
        color=fp_colors[label],
        edgecolors="black", linewidths=0.8
    )
    ax1.annotate(
        label, (sx, px),
        textcoords="offset points",
        xytext=(6, 6), fontsize=8, color="white"
    )

ax1.set_xlim(-0.2, 1.5)
ax1.set_ylim(-0.2, 1.5)
ax1.set_xlabel("S")
ax1.set_ylabel("P")
ax1.set_title("Phase Portrait (Streamplot)")
ax1.legend(fontsize=8)
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.axvline(0, color="gray", linewidth=0.5)
ax1.set_facecolor("#1a1a2e")

# --- Plot 2: Quiver ---
ax2 = axes[1]
skip = 15
ax2.quiver(
    S_grid[::skip, ::skip],
    P_grid[::skip, ::skip],
    dS[::skip, ::skip] / magnitude[::skip, ::skip],
    dP[::skip, ::skip] / magnitude[::skip, ::skip],
    magnitude[::skip, ::skip],
    cmap="plasma", alpha=0.8
)

ax2.plot(S_vals, P_null_S, "b--",
         label="S nullcline", linewidth=1.5)
ax2.plot(S_vals, P_null_P, "r--",
         label="P nullcline", linewidth=1.5)

for label, (sx, px) in fixed_pts.items():
    ax2.scatter(
        sx, px, s=80, zorder=5,
        color=fp_colors[label],
        edgecolors="black", linewidths=0.8
    )
    ax2.annotate(
        label, (sx, px),
        textcoords="offset points",
        xytext=(6, 6), fontsize=8
    )

ax2.set_xlim(-0.2, 1.5)
ax2.set_ylim(-0.2, 1.5)
ax2.set_xlabel("S")
ax2.set_ylabel("P")
ax2.set_title("Phase Portrait (Quiver)")
ax2.legend(fontsize=8)
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.axvline(0, color="gray", linewidth=0.5)

plt.tight_layout()
plt.savefig("figures/phase_portrait.png", dpi=300)
plt.close()

print("\nAll figures saved to figures/")
