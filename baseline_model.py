
import numpy as np


# ==========================================
# 1. COMPETITIVE LOTKA-VOLTERRA MODEL
# ==========================================

def competition_model(t, y, r_s, r_p, K_s, K_p,
                      alpha_sp, alpha_ps):

    S, P = y

    # Staphylococcus growth
    dS_dt = r_s * S * (
        1 - (S + alpha_sp * P) / K_s
    )

    # Pseudomonas growth
    dP_dt = r_p * P * (
        1 - (P + alpha_ps * S) / K_p
    )

    return [dS_dt, dP_dt]


# ==========================================
# 2. CALCULATE EQUILIBRIA
# ==========================================

def calculate_equilibria(K_s, K_p, alpha_sp, alpha_ps):

    equilibria = {}

    # Equilibrium 1: Extinction
    equilibria["Extinction"] = (0, 0)

    # Equilibrium 2: SA-only
    equilibria["SA-only"] = (K_s, 0)

    # Equilibrium 3: PA-only
    equilibria["PA-only"] = (0, K_p)

    # Equilibrium 4: Coexistence
    denominator = 1 - alpha_sp * alpha_ps

    if np.isclose(denominator, 0):

        equilibria["Coexistence"] = None

    else:

        S_star = (
            K_s - alpha_sp * K_p
        ) / denominator

        P_star = (
            K_p - alpha_ps * K_s
        ) / denominator

        if S_star > 0 and P_star > 0:

            equilibria["Coexistence"] = (
                S_star, P_star
            )

        else:

            equilibria["Coexistence"] = None

    return equilibria


# ==========================================
# 3. CALCULATE EQUILIBRIUM STABILITY
# ==========================================

def calculate_stability(S, P, r_s, r_p, K_s, K_p,
                        alpha_sp, alpha_ps):

    J = np.array([
        [
            r_s * (1 - (2*S + alpha_sp*P)/K_s),
            -r_s * alpha_sp * S / K_s
        ],
        [
            -r_p * alpha_ps * P / K_p,
            r_p * (1 - (2*P + alpha_ps*S)/K_p)
        ]
    ])

    eigenvalues = np.linalg.eigvals(J)

    real_parts = np.real(eigenvalues)

    if np.all(real_parts < -1e-10):
        stability = "Stable"

    elif np.all(real_parts > 1e-10):
        stability = "Unstable"

    elif (
        np.any(real_parts > 1e-10)
        and np.any(real_parts < -1e-10)
    ):
        stability = "Saddle"

    else:
        stability = "Inconclusive / Non-hyperbolic"

    return eigenvalues, stability
