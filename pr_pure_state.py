import numpy as np
R = 8.31446261815324  # J/mol/K
def PR_Z(T, P, Tc, Pc, omega, phase="vapor"):
    Tr = T / Tc
    Pr = P / Pc

    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(Tr))) ** 2

    A = 0.45724 * alpha * Pr / Tr**2
    B = 0.07780 * Pr / Tr

    coeffs = [1.0, -(1 - B), A - 3 * B**2 - 2 * B, -(A * B - B**2 - B**3)]
    roots = np.roots(coeffs)
    roots = np.real(roots[np.isclose(roots.imag, 0.0)])

    Z = roots.max() if phase == "vapor" else roots.min()
    return Z, roots.tolist()

def PR_departures(T, P, Tc, Pc, omega, phase="vapor"):
    Z, roots = PR_Z(T, P, Tc, Pc, omega, phase)

    Tr = T / Tc
    Pr = P / Pc

    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(Tr))) ** 2

    A = 0.45724 * alpha * Pr / Tr**2
    B = 0.07780 * Pr / Tr

    sqrt2 = np.sqrt(2.0)
    log_term = np.log((Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B))

    ln_phi = Z - 1 - np.log(Z - B) - A / (2 * sqrt2 * B) * log_term
    phi = np.exp(ln_phi)
    fugacity = phi * P  # same unit as P (here: bar)

    h_dep = R * Tc * (Tr * (Z - 1) - 2.078 * (1 + kappa) * np.sqrt(alpha) * log_term)
    s_dep = R * (
        np.log(Z - B)
        - 2.078 * kappa * ((1 + kappa) / np.sqrt(Tr) - kappa) * log_term
    )
    u_dep = h_dep - R * T * (Z - 1)
    g_dep = R * T * ln_phi

    return {
        "Z": Z,
        "roots": roots,
        "h_dep": h_dep,
        "s_dep": s_dep,
        "u_dep": u_dep,
        "g_dep": g_dep,
        "ln_phi": ln_phi,
        "phi": phi,
        "fugacity": fugacity,
    }

def h_ig_DIPPR(T, A, B, C, D, E, Tref):
    def H(T_):
        return A * T_ + 0.5 * B * T_**2 + C * T_**3 / 3 + D * T_**4 / 4 - E / T_

    return H(T) - H(Tref)

def s_ig_DIPPR(T, P, A, B, C, D, E, Tref, Pref):
    def S(T_):
        return (
            A * np.log(T_)
            + B * T_
            + C * T_**2 / 2
            + D * T_**3 / 3
            - E / (2 * T_**2)
        )

    return (S(T) - S(Tref)) - R * np.log(P / Pref)

def real_state_PR(T, P, Tc, Pc, omega, Cp, Tref, Pref, phase="vapor"):
    A, B, C, D, E = Cp
    pr = PR_departures(T, P, Tc, Pc, omega, phase)

    h_ig = h_ig_DIPPR(T, A, B, C, D, E, Tref)
    s_ig = s_ig_DIPPR(T, P, A, B, C, D, E, Tref, Pref)
    u_ig = h_ig - R * T
    g_ig = h_ig - T * s_ig

    return {
        **pr,
        "h_ig": h_ig,
        "s_ig": s_ig,
        "u_ig": u_ig,
        "g_ig": g_ig,
        "h_real": h_ig + pr["h_dep"],
        "s_real": s_ig + pr["s_dep"],
        "u_real": u_ig + pr["u_dep"],
        "g_real": g_ig + pr["g_dep"],
    }

def process_PR(T1, P1, T2, P2, Tc, Pc, omega, Cp, Tref, Pref, phase="vapor"):
    s1 = real_state_PR(T1, P1, Tc, Pc, omega, Cp, Tref, Pref, phase)
    s2 = real_state_PR(T2, P2, Tc, Pc, omega, Cp, Tref, Pref, phase)
    return {
        "Delta_h": s2["h_real"] - s1["h_real"],
        "Delta_s": s2["s_real"] - s1["s_real"],
    }

def to_mass_basis(props, MW_g_per_mol):
    MW_kg_per_mol = MW_g_per_mol / 1000.0  # kg/mol

    def Jmol_to_kJkg(x):
        return (x / MW_kg_per_mol) / 1000.0

    def JmolK_to_kJkgK(x):
        return (x / MW_kg_per_mol) / 1000.0

    out = {}
    for k in ["h_dep", "u_dep", "g_dep", "h_ig", "u_ig", "g_ig", "h_real", "u_real", "g_real"]:
        out[k] = Jmol_to_kJkg(props[k])

    for k in ["s_dep", "s_ig", "s_real"]:
        out[k] = JmolK_to_kJkgK(props[k])

    return out

def print_state(props, MW_g_per_mol=None):
    print(f"Z        = {props['Z']:.6f}")
    print(f"h_dep    = {props['h_dep']:.3f} J/mol")
    print(f"s_dep    = {props['s_dep']:.5f} J/mol/K")
    print(f"u_dep    = {props['u_dep']:.3f} J/mol")
    print(f"g_dep    = {props['g_dep']:.3f} J/mol")
    print(f"h_ig     = {props['h_ig']:.3f} J/mol (relative to Tref)")
    print(f"s_ig     = {props['s_ig']:.5f} J/mol/K (relative to Tref,Pref)")
    print(f"g_ig     = {props['g_ig']:.3f} J/mol")
    print(f"h_real   = {props['h_real']:.3f} J/mol")
    print(f"s_real   = {props['s_real']:.5f} J/mol/K")
    print(f"g_real   = {props['g_real']:.3f} J/mol")
    print(f"u_real   = {props['u_real']:.3f} J/mol")
    print(f"phi      = {props['phi']:.6f}")
    print(f"fugacity = {props['fugacity']:.3f} bar")

    if MW_g_per_mol is not None:
        mb = to_mass_basis(props, MW_g_per_mol)
        print("\n\n--- mass basis ---")
        print(f"h_dep   = {mb['h_dep']:.3f} kJ/kg")
        print(f"s_dep   = {mb['s_dep']:.5f} kJ/kg.K")
        print(f"u_dep   = {mb['u_dep']:.3f} kJ/kg")
        print(f"g_dep   = {mb['g_dep']:.3f} kJ/kg")
        print(f"h_ig    = {mb['h_ig']:.3f} kJ/kg")
        print(f"s_ig    = {mb['s_ig']:.5f} kJ/kg.K")
        print(f"g_ig    = {mb['g_ig']:.3f} kJ/kg")
        print(f"u_ig    = {mb['u_ig']:.3f} kJ/kg")
        print(f"h_real  = {mb['h_real']:.3f} kJ/kg")
        print(f"s_real  = {mb['s_real']:.5f} kJ/kg.K")
        print(f"g_real  = {mb['g_real']:.3f} kJ/kg")
        print(f"u_real  = {mb['u_real']:.3f} kJ/kg")

if __name__ == "__main__":

    Tc = 126.2      # K (N2)
    Pc = 33.5       # bar
    omega = 0.04
    MW = 28.0134    # g/mol (N2) 

    Cp = (28.9, -0.1571e-2, 0.8081e-5, -2.873e-9, 0.0)

    T1, P1 = 180, 80
    T2, P2 = 125, 5
    Tref, Pref = 298.15, 1.0

    state = real_state_PR(T2, P2, Tc, Pc, omega, Cp, Tref, Pref)
    proc = process_PR(T1, P1, T2, P2, Tc, Pc, omega, Cp, Tref, Pref)

    print_state(state, MW_g_per_mol=MW)

    print("\n\n--- process ---")
    print(f"Δh = {proc['Delta_h']:.3f} J/mol")
    print(f"Δs = {proc['Delta_s']:.5f} J/mol.K")
