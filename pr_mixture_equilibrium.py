import numpy as np
R = 8.31446261815324  # J/mol/K
BAR_TO_PA = 1e5
PA_TO_BAR = 1e-5

EPS = 1e-14  

def _as_molefrac(x, name="x"):
    x = np.asarray(x, dtype=float).copy()
    if x.ndim != 1 or len(x) == 0:
        raise ValueError(f"{name} must be a 1D array with length > 0.")
    if np.any(~np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(x < 0):
        raise ValueError(f"{name} contains negative values.")
    s = float(np.sum(x))
    if s <= 0:
        raise ValueError(f"sum({name}) must be > 0.")
    x /= s
    # protect exact zeros for logs
    x = np.clip(x, EPS, None)
    x /= float(np.sum(x))
    return x

def _check_state(T, P):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and > 0 K.")
    if not np.isfinite(P) or P <= 0:
        raise ValueError("P must be finite and > 0 Pa.")

def _check_components(components):
    if not isinstance(components, (list, tuple)) or len(components) == 0:
        raise ValueError("components must be a non-empty list of dicts.")
    for i, c in enumerate(components):
        for k in ("Tc", "Pc", "omega"):
            if k not in c:
                raise ValueError(f"components[{i}] missing key '{k}'.")
        Tc, Pc, w = c["Tc"], c["Pc"], c["omega"]
        if not (np.isfinite(Tc) and Tc > 0):
            raise ValueError(f"components[{i}].Tc must be > 0.")
        if not (np.isfinite(Pc) and Pc > 0):
            raise ValueError(f"components[{i}].Pc must be > 0 Pa.")
        if not np.isfinite(w):
            raise ValueError(f"components[{i}].omega must be finite.")
        if "Cp" in c:
            Cp = c["Cp"]
            if (not isinstance(Cp, (list, tuple))) or len(Cp) != 5:
                raise ValueError(f"components[{i}].Cp must be a 5-tuple if provided.")

def _check_kij(kij, n):
    kij = np.asarray(kij, dtype=float)
    if kij.shape != (n, n):
        raise ValueError(f"kij must have shape ({n},{n}).")
    if np.any(~np.isfinite(kij)):
        raise ValueError("kij contains non-finite values.")
    return kij

def h_ig_DIPPR(T, A, B, C, D, E, Tref):
    def H(T_):
        return A*T_ + 0.5*B*T_**2 + (C/3.0)*T_**3 + (D/4.0)*T_**4 - E/T_
    return H(T) - H(Tref)

def s_ig_DIPPR_at_Pref(T, A, B, C, D, E, Tref):
    def S(T_):
        return A*np.log(T_) + B*T_ + (C/2.0)*T_**2 + (D/3.0)*T_**3 - E/(2.0*T_**2)
    return S(T) - S(Tref)

def pr_kappa(omega):
    return 0.37464 + 1.54226*omega - 0.26992*omega**2

def pr_alpha(T, Tc, omega):
    Tr_sqrt = np.sqrt(T/Tc)
    k = pr_kappa(omega)
    f = 1.0 + k*(1.0 - Tr_sqrt)
    return f*f

def pr_dalpha_dT(T, Tc, omega):
    k = pr_kappa(omega)
    Tr_sqrt = np.sqrt(T/Tc)
    f = 1.0 + k*(1.0 - Tr_sqrt)
    df_dT = -k * (1.0/(2.0*Tr_sqrt*Tc))
    return 2.0*f*df_dT

def pr_ai_bi_and_dai_dT(T, Tc, Pc, omega):
    a0 = 0.45724 * (R**2) * (Tc**2) / Pc
    b  = 0.07780 * R * Tc / Pc
    alpha = pr_alpha(T, Tc, omega)
    dai_dT = a0 * pr_dalpha_dT(T, Tc, omega)
    a = a0 * alpha
    return a, b, dai_dT

def pr_mix_a_b_da(T, y, components, kij=None):
    y = _as_molefrac(y, "y")
    n = len(components)
    if kij is None:
        kij = np.zeros((n, n), dtype=float)
    kij = _check_kij(kij, n)

    ai = np.zeros(n)
    bi = np.zeros(n)
    dai_dT = np.zeros(n)

    for i, c in enumerate(components):
        ai[i], bi[i], dai_dT[i] = pr_ai_bi_and_dai_dT(T, c["Tc"], c["Pc"], c["omega"])

    aij = np.zeros((n, n))
    daij_dT = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if ai[i] <= 0 or ai[j] <= 0:
                aij[i, j] = 0.0
                daij_dT[i, j] = 0.0
            else:
                sqrt_ai_aj = np.sqrt(ai[i] * ai[j])
                aij[i, j] = sqrt_ai_aj * (1.0 - kij[i, j])
                
                d_product = ai[j]*dai_dT[i] + ai[i]*dai_dT[j]
                daij_dT[i, j] = 0.5 * (d_product / sqrt_ai_aj) * (1.0 - kij[i, j])

    b_mix = float(np.dot(y, bi))
    a_mix = float(y @ aij @ y)
    da_mix_dT = float(y @ daij_dT @ y)

    return a_mix, b_mix, da_mix_dT, aij, bi

def pr_mix_Z(T, P, y, components, kij=None, phase="vapor"):
    a_mix, b_mix, da_mix_dT, aij, bi = pr_mix_a_b_da(T, y, components, kij)

    A = a_mix * P / (R**2 * T**2)
    B = b_mix * P / (R * T)

    coeffs = [1.0,
              -(1.0 - B),
              (A - 3.0*B**2 - 2.0*B),
              -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    roots = np.real(roots[np.isclose(roots.imag, 0.0)])
    roots = np.sort(roots)

    if len(roots) == 0:
        raise RuntimeError("No real roots for PR cubic (check inputs).")

    if len(roots) == 1:
        Z = float(roots[0])
    else:
        Z = float(roots[-1] if phase == "vapor" else roots[0])

    return Z, roots.tolist(), A, B, a_mix, b_mix, da_mix_dT, aij, bi

def pr_mix_departures_and_fugacity(T, P, y, components, kij=None, phase="vapor"):

    _check_state(T, P)
    _check_components(components)

    y = _as_molefrac(y, "y")
    Z, roots, A, B, a_mix, b_mix, da_mix_dT, aij, bi = pr_mix_Z(T, P, y, components, kij, phase)

    Z_safe = np.sign(Z) * max(abs(Z), EPS)
    ZmB = max(Z - B, EPS)

    sqrt2 = np.sqrt(2.0)
    den1 = Z + (1.0 - sqrt2)*B
    den2 = Z + (1.0 + sqrt2)*B
    den1 = np.sign(den1) * max(abs(den1), EPS)
    den2 = np.sign(den2) * max(abs(den2), EPS)

    ln_arg = den2 / den1
    ln_arg = max(ln_arg, EPS)
    log_term = np.log(ln_arg)

    n = len(y)
    ln_phi = np.zeros(n)

    B_safe = np.sign(B) * max(abs(B), EPS)
    bmix_safe = max(b_mix, EPS)
    amix_safe = max(a_mix, EPS)

    for i in range(n):
        sum_aij = float(np.dot(y, aij[i, :]))
        term1 = (bi[i]/bmix_safe) * (Z - 1.0) - np.log(ZmB)
        term2_factor = (2.0*sum_aij/amix_safe) - (bi[i]/bmix_safe)
        term2 = (A/(2.0*sqrt2*B_safe)) * term2_factor * log_term
        ln_phi[i] = term1 - term2

    phi_i = np.exp(np.clip(ln_phi, -700, 700))
    f_i = phi_i * y * P  # Pa
    h_dep = R*T*(Z - 1.0) + (T*da_mix_dT - a_mix)/(2.0*sqrt2*bmix_safe) * log_term
    s_dep = R*np.log(ZmB) - (da_mix_dT)/(2.0*sqrt2*bmix_safe) * log_term
    u_dep = h_dep - R*T*(Z - 1.0)
    g_dep = float(np.dot(y, R*T*ln_phi))

    return {
        "Z": Z,
        "roots": roots,
        "A": A, "B": B,
        "ln_phi_i": ln_phi,
        "phi_i": phi_i,
        "fugacity_i": f_i,  # Pa
        "h_dep": h_dep,     # J/mol
        "s_dep": s_dep,     # J/mol/K
        "u_dep": u_dep,     # J/mol
        "g_dep": g_dep,     # J/mol
        "y": y,
        "a_mix": a_mix,
        "b_mix": b_mix
    }

def ideal_mixture_props(T, P, y, components, Tref, Pref):

    y = _as_molefrac(y, "y")
    n = len(y)

    h_ig_i = np.zeros(n)
    s_ig_i_at_Pref = np.zeros(n)

    for i, c in enumerate(components):
        if "Cp" not in c:
            raise ValueError("Cp is required for do_ideal=True.")
        A, B, Cc, D, E = c["Cp"]
        h_ig_i[i] = h_ig_DIPPR(T, A, B, Cc, D, E, Tref)
        s_ig_i_at_Pref[i] = s_ig_DIPPR_at_Pref(T, A, B, Cc, D, E, Tref)

    s_ig_pure_avg = float(np.dot(y, s_ig_i_at_Pref))
    s_mix_ideal = -R * float(np.dot(y, np.log(y)))  
    s_pressure = -R * np.log(P / Pref)               
    
    s_ig = s_ig_pure_avg + s_mix_ideal + s_pressure

    u_ig = h_ig - R*T
    g_ig = h_ig - T*s_ig

    return {
        "h_ig": h_ig,
        "s_ig": s_ig,
        "u_ig": u_ig,
        "g_ig": g_ig,
        "s_ig_mix": s_mix_ideal,
    }

def real_mixture_PR(T, P, y, components, kij, Tref, Pref, phase="vapor", do_ideal=True):

    _check_state(T, P)
    _check_components(components)
    kij = _check_kij(kij, len(components))

    pr = pr_mix_departures_and_fugacity(T, P, y, components, kij, phase)
    out = {**pr}

    if phase == "liquid":
        out["x"] = out["y"].copy()
    else:
        out["y"] = out["y"].copy()

    if do_ideal:
        ig = ideal_mixture_props(T, P, y, components, Tref, Pref)
        out.update(ig)
        out["h_real"] = ig["h_ig"] + pr["h_dep"]
        out["s_real"] = ig["s_ig"] + pr["s_dep"]
        out["u_real"] = ig["u_ig"] + pr["u_dep"]
        out["g_real"] = ig["g_ig"] + pr["g_dep"]
    
    return out

def stable_phase_mixture_PR(T, P, y, components, kij, Tref, Pref, do_ideal=True):

    st_v = real_mixture_PR(T, P, y, components, kij, Tref, Pref, phase="vapor",  do_ideal=do_ideal)
    st_l = real_mixture_PR(T, P, y, components, kij, Tref, Pref, phase="liquid", do_ideal=do_ideal)

    if do_ideal and ("g_real" in st_v) and ("g_real" in st_l):
        gv, gl = st_v["g_real"], st_l["g_real"]
        metric = "g_real"
    else:
        gv, gl = st_v["g_dep"], st_l["g_dep"]
        metric = "g_dep"

    if gl < gv:
        st_best = st_l
        selected = "liquid"
    else:
        st_best = st_v
        selected = "vapor"

    roots = st_v.get("roots", [])
    three_real_roots = (len(roots) >= 3)

    info = {
        "selected_phase": selected,
        "metric": metric,
        "metric_vapor": gv,
        "metric_liquid": gl,
        "three_real_roots": three_real_roots,
        "vapor_candidate": st_v,
        "liquid_candidate": st_l,
    }
    return st_best, info

def print_mix_state(st, names, title=None):
    """Print formatted thermodynamic state."""
    if title:
        print(title)

    print(f"Z        = {st['Z']:.6f}")
    print(f"h_dep    = {st['h_dep']:.3f} J/mol")
    print(f"s_dep    = {st['s_dep']:.5f} J/mol/K")
    print(f"u_dep    = {st['u_dep']:.3f} J/mol")
    print(f"g_dep    = {st['g_dep']:.3f} J/mol")

    if "h_ig" in st:
        print(f"\nh_ig     = {st['h_ig']:.3f} J/mol")
        print(f"s_ig     = {st['s_ig']:.5f} J/mol/K")
        print(f"g_ig     = {st['g_ig']:.3f} J/mol")
        print(f"\nh_real   = {st['h_real']:.3f} J/mol")
        print(f"s_real   = {st['s_real']:.5f} J/mol/K")
        print(f"g_real   = {st['g_real']:.3f} J/mol")
        print(f"u_real   = {st['u_real']:.3f} J/mol")

    comp = st.get("x", st.get("y", None))
    label = "x" if "x" in st else "y"

    print("\n--- Component Fugacity ---")
    for i, nm in enumerate(names):
        f_bar = st["fugacity_i"][i] * PA_TO_BAR
        xi = comp[i] if comp is not None else st["y"][i]
        print(f"{nm:>10s}: {label}={xi:.4f}  phi={st['phi_i'][i]:.6f}  f={f_bar:.6f} bar")

def wilson_K(T, P, components):
    K = []
    P_bar = P * PA_TO_BAR  
    
    for c in components:
        Tc, Pc, w = c["Tc"], c["Pc"], c["omega"]
        Pc_bar = Pc * PA_TO_BAR
        lnK = np.log(Pc_bar / P_bar) + 5.373*(1.0 + w)*(1.0 - Tc/T)
        K.append(np.exp(lnK))
    
    return np.array(K, dtype=float)

def rachford_rice(beta, z, K):
    return np.sum(z*(K - 1.0) / (1.0 + beta*(K - 1.0)))

def solve_beta_rr(z, K, tol=1e-12, max_iter=200):
    z = _as_molefrac(z, "z")
    K = np.asarray(K, float)

    f0 = rachford_rice(0.0, z, K)
    f1 = rachford_rice(1.0, z, K)

    if f0 < 0.0 and f1 < 0.0:
        return 0.0, "liquid"
    if f0 > 0.0 and f1 > 0.0:
        return 1.0, "vapor"

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        fm = rachford_rice(mid, z, K)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid, "two-phase"
        if fm > 0.0:
            lo = mid
        else:
            hi = mid
    return mid, "two-phase"

def _lnphi_PR(T, P, comp, components, kij, phase):
    st = pr_mix_departures_and_fugacity(T, P, comp, components, kij, phase=phase)
    return np.asarray(st["ln_phi_i"], float), st

def tpd_stability_test_PR(T, P, z, components, kij, Tref, Pref, max_iter=100, tol=1e-10):
    _check_state(T, P)
    _check_components(components)
    z = _as_molefrac(z, "z")
    kij = _check_kij(kij, len(components))

    st_ref, info_ref = stable_phase_mixture_PR(T, P, z, components, kij, Tref, Pref, do_ideal=False)
    ref_phase = info_ref["selected_phase"]
    lnphi_ref = np.asarray(st_ref["ln_phi_i"], float)

    K0 = np.clip(wilson_K(T, P, components), 1e-12, 1e12)

    tests = [
        {"name": "vapor-like",  "trial_phase": "vapor",  "w0": _as_molefrac(z * K0, "w0")},
        {"name": "liquid-like", "trial_phase": "liquid", "w0": _as_molefrac(z / K0, "w0")},
    ]

    best_tpd = np.inf
    details = []

    for t in tests:
        w = t["w0"].copy()
        trial_phase = t["trial_phase"]
        converged = False

        for _ in range(max_iter):
            lnphi_w, _ = _lnphi_PR(T, P, w, components, kij, phase=trial_phase)

            w_new = z * np.exp(np.clip(lnphi_ref - lnphi_w, -200, 200))
            w_new = _as_molefrac(w_new, "w_new")

            if np.max(np.abs(w_new - w)) < tol:
                w = w_new
                converged = True
                break

            w = 0.5*w + 0.5*w_new

        lnphi_w, _ = _lnphi_PR(T, P, w, components, kij, phase=trial_phase)
        tpd = float(np.sum(w * ((np.log(w) + lnphi_w) - (np.log(z) + lnphi_ref))))

        best_tpd = min(best_tpd, tpd)
        details.append({
            "test": t["name"],
            "trial_phase": trial_phase,
            "converged": converged,
            "tpd": tpd,
            "w": w,
        })

    TPD_TOL = 1e-10
    stable = (best_tpd >= -TPD_TOL)

    return {"stable": stable, "min_tpd": best_tpd, "details": details, "ref_phase": ref_phase}

def tp_flash_PR(T, P, z, components, kij, Tref, Pref, tol=1e-10, max_iter=80):
    _check_state(T, P)
    _check_components(components)
    kij = _check_kij(kij, len(components))

    z = _as_molefrac(z, "z")
    K = np.clip(wilson_K(T, P, components), 1e-12, 1e12)

    for _ in range(max_iter):
        beta_rr, region = solve_beta_rr(z, K)

        if region == "two-phase":
            beta = beta_rr
        elif region == "liquid":
            beta = 1e-8
        else:
            beta = 1.0 - 1e-8

        den = 1.0 + beta*(K - 1.0)
        den = np.clip(den, 1e-14, None)
        x = _as_molefrac(z / den, "x")
        y = _as_molefrac(K * x, "y")

        stL = real_mixture_PR(T, P, x, components, kij, Tref, Pref, phase="liquid", do_ideal=False)
        stV = real_mixture_PR(T, P, y, components, kij, Tref, Pref, phase="vapor",  do_ideal=False)

        phiL = np.clip(stL["phi_i"], EPS, None)
        phiV = np.clip(stV["phi_i"], EPS, None)
        K_new = np.clip(phiL / phiV, 1e-12, 1e12)

        if np.max(np.abs(np.log(K_new / K))) < tol:
            beta_final, region_final = solve_beta_rr(z, K_new)

            if region_final == "liquid":
                stL_full = real_mixture_PR(T, P, z, components, kij, Tref, Pref, phase="liquid", do_ideal=True)
                return {"region": "liquid", "beta": 0.0, "x": z, "y": None, "K": K_new, "stL": stL_full}

            if region_final == "vapor":
                stV_full = real_mixture_PR(T, P, z, components, kij, Tref, Pref, phase="vapor", do_ideal=True)
                return {"region": "vapor", "beta": 1.0, "x": None, "y": z, "K": K_new, "stV": stV_full}

            beta = beta_final
            den = 1.0 + beta*(K_new - 1.0)
            den = np.clip(den, 1e-14, None)
            x = _as_molefrac(z / den, "x")
            y = _as_molefrac(K_new * x, "y")

            stL_full = real_mixture_PR(T, P, x, components, kij, Tref, Pref, phase="liquid", do_ideal=True)
            stV_full = real_mixture_PR(T, P, y, components, kij, Tref, Pref, phase="vapor",  do_ideal=True)
            return {"region": "two-phase", "beta": beta, "x": x, "y": y, "K": K_new, "stL": stL_full, "stV": stV_full}

        K = 0.5*K + 0.5*K_new

    raise RuntimeError(f"Flash calculation did not converge after {max_iter} iterations")

def industrial_TP_equilibrium_PR(T, P, z, components, kij, Tref, Pref, do_ideal=True, verbose=True):
    _check_state(T, P)
    _check_components(components)
    kij = _check_kij(kij, len(components))

    stab = tpd_stability_test_PR(T, P, z, components, kij, Tref, Pref)

    if verbose:
        print("\n" + "="*60)
        print("TPD STABILITY TEST")
        print("="*60)
        print(f"Reference phase: {stab['ref_phase'].upper()}")
        print(f"Minimum TPD: {stab['min_tpd']:.6e}")
        print("\nTrial phase results:")
        for d in stab["details"]:
            print(f"  {d['test']:12s} ({d['trial_phase']:6s}): "
                  f"TPD = {d['tpd']:+.6e}  |  Converged: {d['converged']}")

    if stab["stable"]:
        if verbose:
            print(f"\n>>> Feed is STABLE as {stab['ref_phase'].upper()} phase")
        
        state, info = stable_phase_mixture_PR(T, P, z, components, kij, Tref, Pref, do_ideal=do_ideal)
        return {
            "type": "single-phase",
            "phase": info["selected_phase"],
            "state": state,
            "stability": stab
        }

    if verbose:
        print("\n>>> Feed is UNSTABLE - running flash calculation...")

    try:
        flash = tp_flash_PR(T, P, z, components, kij, Tref, Pref)
    except RuntimeError as e:
        if verbose:
            print(f"\nWARNING: {e}")
            print("Returning single-phase approximation")
        state, info = stable_phase_mixture_PR(T, P, z, components, kij, Tref, Pref, do_ideal=do_ideal)
        return {
            "type": "single-phase",
            "phase": info["selected_phase"],
            "state": state,
            "stability": stab,
            "warning": str(e)
        }

    if flash["region"] == "two-phase":
        if verbose:
            print(f"\n>>> TWO-PHASE equilibrium confirmed")
            print(f"    Vapor fraction (beta) = {flash['beta']:.6f}")
        return {
            "type": "two-phase",
            "flash": flash,
            "stability": stab
        }

    if flash["region"] == "liquid":
        return {
            "type": "single-phase",
            "phase": "liquid",
            "state": flash["stL"],
            "stability": stab
        }
    else:
        return {
            "type": "single-phase",
            "phase": "vapor",
            "state": flash["stV"],
            "stability": stab
        }
    
if __name__ == "__main__":
    import numpy as np

    print("="*60)
    print("PENG-ROBINSON EOS - PURE COMPONENT (CO2 example)")
    print("="*60)

    T = 320.0  # K
    P_MPa = 10.0
    P = P_MPa * 1e6  # Pa

   
    z = np.array([1.0])
    names = ["CO2"]
    a = 22.26
    b = 5.981e-2
    c = -3.501e-5
    d = 7.469e-9

    components = [
        {
            "name": "CO2",
            "Tc": 304.1282,      # K
            "Pc": 7.3773e6,      # Pa
            "omega": 0.225,      # -
            "MW": 44.01,         # g/mol 
            "Cp": (a, b, c, d, 0.0),
        }
    ]

    kij = np.zeros((1, 1))

    Tref = 304.1282   # K
    Pref = 7.3773e6    # Pa 

    result = industrial_TP_equilibrium_PR(
        T, P, z, components, kij, Tref, Pref,
        do_ideal=True,  
        verbose=True
    )

    print("\n" + "="*60)
    print("EQUILIBRIUM RESULT")
    print("="*60)

    if result["type"] == "single-phase":
        phase = result["phase"]
        print(f"\nResult: SINGLE-PHASE ({phase.upper()})")
        print_mix_state(result["state"], names, title=f"\n--- {phase.upper()} Phase Properties ---")
    else:
        flash = result["flash"]
        print(f"\nResult: TWO-PHASE EQUILIBRIUM")
        print(f"\nVapor fraction (beta) = {flash['beta']:.6f}")
        print(f"Liquid composition (x): {flash['x']}")
        print(f"Vapor composition  (y): {flash['y']}")
        print(f"\nK-values: {flash['K']}")
        print_mix_state(flash["stL"], names, title="\n--- LIQUID Phase Properties ---")
        print_mix_state(flash["stV"], names, title="\n--- VAPOR Phase Properties ---")
