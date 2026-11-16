
import numpy as np
from sympy import (
    symbols, Matrix, simplify, matrix2numpy
)

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

# ==========================================
# 4. SYMBOLIC FIXED-POINT ANALYSIS
# ==========================================

def symbolic_fixed_point_analysis(params):
    """
    Compute eigenvalues, trace, determinant, and
    stability classification for all four fixed points
    using the symbolic Jacobian.

    params: dict with keys r_s, r_p, K_s, K_p,
            alpha_sp, alpha_ps (all numeric)
    """

    r_s, r_p, K_s, K_p, alpha_sp, alpha_ps = symbols(
        'r_s r_p K_s K_p alpha_sp alpha_ps', positive=True
    )

    S, P = symbols('S P')

    sym_params = {
        r_s:      params["r_s"],
        r_p:      params["r_p"],
        K_s:      params["K_s"],
        K_p:      params["K_p"],
        alpha_sp: params["alpha_sp"],
        alpha_ps: params["alpha_ps"],
    }

    # Symbolic Jacobian
    J11 = r_s - (r_s / K_s) * (2*S + alpha_sp*P)
    J12 = -r_s * alpha_sp * S / K_s
    J21 = -r_p * alpha_ps * P / K_p
    J22 = r_p - (r_p / K_p) * (2*P + alpha_ps*S)

    J = Matrix([[J11, J12], [J21, J22]])

    # Coexistence fixed point (symbolic)
    S_star = (K_s - alpha_sp*K_p) / (1 - alpha_sp*alpha_ps)
    P_star = (K_p - alpha_ps*K_s) / (1 - alpha_sp*alpha_ps)

    fixed_points = {
        "(0, 0)":   {S: 0,      P: 0},
        "(K_s, 0)": {S: K_s,    P: 0},
        "(0, K_p)": {S: 0,      P: K_p},
        "(S*, P*)": {S: S_star,  P: P_star},
    }

    print("\nSYMBOLIC FIXED-POINT ANALYSIS")

    for name, point in fixed_points.items():

        print(f"\n--- Fixed point {name} ---")

        # Substitute fixed point then numerical params
        J_eval = J.subs(point).subs(sym_params)
        J_num = matrix2numpy(J_eval, dtype=float)

        eigenvalues = np.linalg.eigvals(J_num)
        print("Eigenvalues:", eigenvalues)

        det = eigenvalues[0] * eigenvalues[1]
        print("Determinant:", det)

        tr = eigenvalues[0] + eigenvalues[1]
        print("Trace:      ", tr)

        discriminant = tr**2 - 4*det
        print("Discriminant:", discriminant)

        if det < 0:
            stability = "Saddle"
        elif det > 0:
            if discriminant < 0:
                stability = (
                    "Stable Spiral" if tr < 0
                    else "Unstable Spiral"
                )
            elif discriminant > 0:
                stability = (
                    "Stable Node" if tr < 0
                    else "Unstable Node"
                )
            elif discriminant == 0:
                stability = "Degenerate Node"
            elif tr == 0:
                stability = "Center"
            else:
                stability = "Non-isolated fixed points"
        else:
            stability = "Inconclusive"

        print("Stability:  ", stability)

    print("\nSYMBOLIC JACOBIAN ANALYSIS (simplified)")

    for name, point in fixed_points.items():

        print(f"\nFixed point: {name}")

        J_sym = simplify(J.subs(point))
        print("Jacobian:")
        print(J_sym)

        eigenvalues_sym = J_sym.eigenvals()
        for eigenvalue, multiplicity in eigenvalues_sym.items():
            print(
                f"Eigenvalue: {eigenvalue}  "
                f"Multiplicity: {multiplicity}"
            )

        trace = simplify(J_sym.trace())
        determinant = simplify(J_sym.det())
        print("Trace:      ", trace)
        print("Determinant:", determinant)