
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from baseline_model import (
    competition_model,
    calculate_equilibria,
    calculate_stability
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


# ==========================================
# 2. CALCULATE EQUILIBRIA
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
# 3. EQUILIBRIUM STABILITY
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
# 4. NUMERICAL SIMULATIONS
# ==========================================

initial_conditions = [
    (0.1, 0.1),    # Equal initial populations
    (0.9, 0.1),    # SA initially dominant
    (0.1, 0.9),    # PA initially dominant
    (0.5, 0.5)     # Intermediate initial populations
]

t_span = (0, 100)

t_eval = np.linspace(0, 100, 500)


for ratio in K_ratios:

    K_s = 1.0
    K_p = ratio

    plt.figure(figsize=(10, 6))

    for S0, P0 in initial_conditions:

        solution = solve_ivp(
            competition_model,
            t_span,
            [S0, P0],
            args=(
                r_s, r_p, K_s, K_p,
                alpha_sp, alpha_ps
            ),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10
        )

        if not solution.success:
            print("Simulation failed:", solution.message)
            continue

        plt.plot(
            solution.t,
            solution.y[0],
            label=f"SA (S0={S0}, P0={P0})"
        )

        plt.plot(
            solution.t,
            solution.y[1],
            linestyle="--",
            label=f"PA (S0={S0}, P0={P0})"
        )

    plt.xlabel("Time (hours)")
    plt.ylabel("Normalized bacterial population")

    plt.title(
        f"Competition Model: Kp/Ks = {ratio}"
    )

    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"figures/competition_Kratio_{ratio}.png",
        dpi=300
    )

    plt.show()
