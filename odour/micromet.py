"""
Surface-layer micrometeorology shared by the Lagrangian (ls_model.py) and
Gaussian (gaussian_plume.py) models.

Everything here is Monin-Obukhov similarity theory (MOST), following the
same physics as Kevin's STLS model (data/_model2, Lagrangian Model_b.xlsm,
"Particle Settling II.doc") and WindTrax (Flesch, Wilson & Yee 1995;
Flesch et al. 2004), with the convective/stable boundary-layer extensions
GRAL uses above the surface layer (GralDispersionModel/GRAL, Zeitschleife.cs).

All functions are numpy-vectorized: pass scalars or equal-shape arrays.

HRDPS gives us wind at 10/40/80/120 m AGL, temperature at 2/40/80/120 m
AGL and boundary-layer height, but no surface fluxes - so u* and L are
derived from the profile (u* from the 10 m wind + roughness, theta* from
the 2-40 m potential-temperature difference), iterated to consistency.
That's the same profile method AERMET-style preprocessors use when only
tower data is available.
"""

import numpy as np

KAPPA = 0.4          # von Karman constant
G = 9.81             # m/s2
CP = 1004.0          # J/kg/K
RHO_AIR = 1.2        # kg/m3

# Kolmogorov constant. 3.1 is the value consistent with sigma_w = 1.25 u*
# and the classic neutral-surface-layer TL = 0.5 z / sigma_w (the WindTrax
# setup); GRAL uses 4 (vertical) / 3 (horizontal). Kevin's STLS VBA used
# TL = 0.75 z / sigma_w, which corresponds to C0 ~ 2.1.
C0_DEFAULT = 3.1

# Neutral/stable velocity-variance ratios (sigma / u*). WindTrax defaults;
# Kevin's thesis (Angle & Sakiyama 1991) used 2.4 / 1.9 / 1.3 - within 5%.
SIGU_NEUTRAL = 2.5
SIGV_NEUTRAL = 2.0
SIGW_NEUTRAL = 1.25

L_NEUTRAL = 1.0e5    # |L| above this is treated as neutral
L_MIN_STABLE = 2.0   # floor on stable L (very stable collapse guard)
L_MIN_UNSTABLE = -2.0
SIGMA_FLOOR = 0.05   # m/s, residual turbulence floor (night/residual layer)
EPS_FLOOR = 1.0e-6   # m2/s3


def _arr(x):
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# MOST stability functions (Businger-Dyer, as in Kevin's VBA STLModel)
# ---------------------------------------------------------------------------

def psi_m(zeta):
    """Integrated momentum stability correction psi_m(z/L)."""
    zeta = _arr(zeta)
    out = np.empty_like(zeta)
    st = zeta >= 0
    out[st] = -5.0 * np.minimum(zeta[st], 1.0)
    x = (1.0 - 16.0 * zeta[~st]) ** 0.25
    out[~st] = (2.0 * np.log(0.5 * (1.0 + x)) + np.log(0.5 * (1.0 + x * x))
                - 2.0 * np.arctan(x) + np.pi / 2.0)
    # beyond zeta=1 hold the zeta=1 value plus a linear term (avoids the
    # runaway -5*zeta of very stable nights; Beljaars-Holtslag-like shape)
    vs = zeta > 1.0
    out[vs] = -5.0 - 1.0 * (zeta[vs] - 1.0)
    return out


def psi_h(zeta):
    """Integrated heat stability correction psi_h(z/L)."""
    zeta = _arr(zeta)
    out = np.empty_like(zeta)
    st = zeta >= 0
    out[st] = -5.0 * np.minimum(zeta[st], 1.0)
    vs = zeta > 1.0
    out[vs] = -5.0 - 1.0 * (zeta[vs] - 1.0)
    x = (1.0 - 16.0 * zeta[~st]) ** 0.25
    out[~st] = 2.0 * np.log(0.5 * (1.0 + x * x))
    return out


def phi_m(zeta):
    """Non-dimensional wind shear phi_m(z/L)."""
    zeta = _arr(zeta)
    return np.where(zeta >= 0,
                    1.0 + 5.0 * np.minimum(zeta, 1.0) + np.maximum(zeta - 1.0, 0.0),
                    (1.0 - 16.0 * np.minimum(zeta, 0.0)) ** -0.25)


def potential_temperature(T_k, z_agl):
    """Dry-adiabatic potential temperature relative to the surface (K).
    Only differences matter here, so the 0.0098 K/m lapse correction is
    enough (no pressure needed)."""
    return _arr(T_k) + 0.0098 * _arr(z_agl)


# ---------------------------------------------------------------------------
# u*, L from HRDPS profile
# ---------------------------------------------------------------------------

def surface_layer_from_profile(U10, T2, T40, z0=0.1, zi=None, n_iter=12):
    """
    Derive friction velocity u* (m/s), Obukhov length L (m), temperature
    scale theta* (K), kinematic heat flux (K m/s) and convective velocity
    w* (m/s) from the HRDPS 10 m wind speed and 2/40 m temperatures.

        u*     = kappa U10 / (ln(10/z0) - psi_m(10/L))
        theta* = kappa dtheta / (ln(40/2) - psi_h(40/L) + psi_h(2/L))
        L      = u*^2 theta_mean / (kappa g theta*)

    iterated from a neutral start. Returns a dict of arrays.
    """
    U10 = np.maximum(_arr(U10), 0.3)        # calm floor (AERMOD uses 0.5 at 10 m)
    z0 = np.broadcast_to(_arr(z0), U10.shape).astype(float)
    th2 = potential_temperature(T2, 2.0)
    th40 = potential_temperature(T40, 40.0)
    dth = th40 - th2
    thm = 0.5 * (th2 + th40)

    L = np.full(U10.shape, L_NEUTRAL)
    for _ in range(n_iter):
        ustar = KAPPA * U10 / np.maximum(np.log(10.0 / z0) - psi_m(10.0 / L), 0.5)
        denom = np.log(40.0 / 2.0) - psi_h(40.0 / L) + psi_h(2.0 / L)
        tstar = KAPPA * dth / np.maximum(denom, 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            L_new = ustar ** 2 * thm / (KAPPA * G * tstar)
        L_new = np.where(np.abs(tstar) < 1e-6, L_NEUTRAL, L_new)
        L_new = np.where((L_new > 0) & (L_new < L_MIN_STABLE), L_MIN_STABLE, L_new)
        L_new = np.where((L_new < 0) & (L_new > L_MIN_UNSTABLE), L_MIN_UNSTABLE, L_new)
        L_new = np.clip(L_new, -L_NEUTRAL, L_NEUTRAL)
        # under-relax in 1/L (stable cases oscillate otherwise). Averaging L
        # itself breaks near neutral: a sign flip of equal size (e.g. +1e5 ->
        # -1e5) averages to L = 0 and everything downstream goes NaN.
        inv = 0.5 / L + 0.5 / L_new
        L = np.where(np.abs(inv) < 1.0 / L_NEUTRAL, L_NEUTRAL, 1.0 / np.where(inv == 0, 1.0, inv))

    ustar = KAPPA * U10 / np.maximum(np.log(10.0 / z0) - psi_m(10.0 / L), 0.5)
    kin_heat = -ustar * tstar                # w'theta' (K m/s), >0 daytime
    out = {"ustar": ustar, "L": L, "tstar": tstar, "kin_heat": kin_heat, "z0": z0}
    if zi is not None:
        zi = np.maximum(_arr(zi), 50.0)
        out["zi"] = zi
        out["wstar"] = convective_velocity(kin_heat, zi, thm)
    return out


def convective_velocity(kin_heat, zi, theta_mean=288.0):
    """Deardorff convective velocity scale w* (m/s); 0 when stable."""
    kh = np.maximum(_arr(kin_heat), 0.0)
    return (G / _arr(theta_mean) * kh * _arr(zi)) ** (1.0 / 3.0)


def wind_profile_ratio(z, zref, z0, L):
    """U(z)/U(zref) from the MOST log profile - used below the 10 m HRDPS
    level, where the gridded wind can't resolve the surface shear."""
    z = np.maximum(_arr(z), _arr(z0) * 1.01)
    num = np.log(z / z0) - psi_m(z / L)
    den = np.log(zref / z0) - psi_m(zref / L)
    return np.maximum(num, 0.01) / np.maximum(den, 0.01)


# ---------------------------------------------------------------------------
# Turbulence profiles (surface layer + boundary layer)
# ---------------------------------------------------------------------------

def turbulence(z, ustar, L, zi, wstar=0.0, C0=C0_DEFAULT):
    """
    Velocity standard deviations, dissipation rate and Lagrangian
    timescales at height z (m AGL).

    Surface layer (WindTrax / Flesch et al. 2004, Panofsky et al. 1977):
        unstable: sigma_u = sigma_v = u* (12 - 0.5 zi/L)^(1/3)
                  sigma_w = 1.25 u* (1 - 3 z/L)^(1/3)
        neutral/stable: sigma_u,v,w = (2.5, 2.0, 1.25) u*
        eps = u*^3 / (kappa z) * (phi_m(z/L) - z/L)       (TKE budget)

    Mixed layer (convective; Hanna 1982):
        sigma_w^2 = max(SL value, 1.8 w*^2 (z/zi)^(2/3) (1 - 0.8 z/zi)^2)
        eps       >= 0.4 w*^3 / zi

    Stable/neutral BL decay (after Nieuwstadt 1984): sigma *= (1 - 0.9 z/zi)^(3/4),
    eps *= (1 - 0.9 z/zi)^(3/2). The 0.9 keeps a small residual sigma at zi
    instead of Nieuwstadt's exact zero, whose infinite gradient breaks the
    well-mixed condition at the BL top; above zi a weak floor applies.

    Lagrangian timescales: T_Li = 2 sigma_i^2 / (C0 eps).
    """
    z = np.maximum(_arr(z), 0.05)
    ustar = _arr(ustar)
    L = _arr(L)
    zi = np.maximum(_arr(zi), 50.0)
    wstar = _arr(wstar)

    zeta = z / L
    unstable = L < 0
    above = z >= zi
    frac = np.clip(z / zi, 0.0, 1.0)

    # --- horizontal ---
    sig_h_unst = ustar * np.maximum(12.0 - 0.5 * zi / L, 1.0) ** (1.0 / 3.0)
    sigu = np.where(unstable, sig_h_unst, SIGU_NEUTRAL * ustar)
    sigv = np.where(unstable, sig_h_unst, SIGV_NEUTRAL * ustar)
    decay_h = np.where(unstable, 1.0, (1.0 - 0.9 * frac) ** 0.75)
    sigu = sigu * decay_h
    sigv = sigv * decay_h

    # --- vertical (shared with dsigw2_dz, which needs only this) ---
    sigw = _sigw(z, ustar, zeta, unstable, frac, wstar, above)

    # --- dissipation ---
    zeta_c = np.where(unstable, zeta, np.minimum(zeta, 5.0))
    eps_sl = ustar ** 3 / (KAPPA * z) * np.maximum(phi_m(zeta_c) - zeta_c, 0.2)
    eps = np.where(unstable,
                   np.maximum(eps_sl, 0.4 * wstar ** 3 / zi),
                   eps_sl * (1.0 - 0.9 * frac) ** 1.5)

    # residual layer above zi
    sigu = np.where(above, SIGMA_FLOOR * 2, sigu)
    sigv = np.where(above, SIGMA_FLOOR * 2, sigv)
    eps = np.where(above, EPS_FLOOR * 10, eps)

    sigu = np.maximum(sigu, 2 * SIGMA_FLOOR)
    sigv = np.maximum(sigv, 2 * SIGMA_FLOOR)
    eps = np.maximum(eps, EPS_FLOOR)

    TLu = 2.0 * sigu ** 2 / (C0 * eps)
    TLv = 2.0 * sigv ** 2 / (C0 * eps)
    TLw = 2.0 * sigw ** 2 / (C0 * eps)
    return {"sigu": sigu, "sigv": sigv, "sigw": sigw, "eps": eps,
            "TLu": TLu, "TLv": TLv, "TLw": TLw}


def _sigw(z, ustar, zeta, unstable, frac, wstar, above):
    """sigma_w as turbulence() returns it (inputs already prepared there).
    Split out 2026-10-21 so dsigw2_dz no longer computes the full turbulence
    set twice per sub-step (~half the LS run time); same arithmetic, same result."""
    sigw_sl = SIGW_NEUTRAL * ustar * np.where(unstable, (1.0 - 3.0 * np.minimum(zeta, 0.0)) ** (1.0 / 3.0), 1.0)
    # unstable: the larger of the surface-layer (free-convection) form and
    # the mixed-layer form - NOT their sum, which double-counts convective
    # variance near the ground (K ~ sigma_w^4/eps, so it overstates mixing
    # ~2x; caught by the Prairie Grass check)
    sigw2 = np.where(unstable,
                     np.maximum(sigw_sl ** 2,
                                1.8 * wstar ** 2 * frac ** (2.0 / 3.0) * (1.0 - 0.8 * frac) ** 2),
                     (sigw_sl * (1.0 - 0.9 * frac) ** 0.75) ** 2)
    sigw = np.sqrt(sigw2)
    sigw = np.where(above, SIGMA_FLOOR, sigw)          # residual layer above zi
    return np.maximum(sigw, SIGMA_FLOOR)


def sigma_w(z, ustar, L, zi, wstar=0.0):
    """sigma_w alone, identical to turbulence(...)["sigw"]."""
    z = np.maximum(_arr(z), 0.05)
    ustar = _arr(ustar)
    L = _arr(L)
    zi = np.maximum(_arr(zi), 50.0)
    wstar = _arr(wstar)
    return _sigw(z, ustar, z / L, L < 0, np.clip(z / zi, 0.0, 1.0), wstar, z >= zi)


def dsigw2_dz(z, ustar, L, zi, wstar=0.0, dz=0.5):
    """Vertical gradient of sigma_w^2 (centred difference) - needed for the
    Thomson (1987) well-mixed drift term. Done numerically so it stays
    consistent with whatever turbulence() returns. z+dz and z-dz are
    evaluated in one call (2026-10-21)."""
    z = _arr(z)
    n = z.size
    zp = z + dz
    zm = np.maximum(z - dz, 0.05)
    two = lambda a: np.concatenate([np.broadcast_to(_arr(a), z.shape).ravel()] * 2)
    s2 = sigma_w(np.concatenate([zp.ravel(), zm.ravel()]), two(ustar), two(L), two(zi), two(wstar)) ** 2
    sp, sm = s2[:n].reshape(z.shape), s2[n:].reshape(z.shape)
    return (sp - sm) / (zp - zm)


# ---------------------------------------------------------------------------
# Pasquill-Gifford class from L (for the Gaussian model)
# ---------------------------------------------------------------------------

# Golder (1972) as fitted in Seinfeld & Pandis: 1/L = a + b log10(z0)
_GOLDER = {"A": (-0.096, 0.029), "B": (-0.037, 0.029), "C": (-0.002, 0.018),
           "D": (0.0, 0.0), "E": (0.004, -0.018), "F": (0.035, -0.036)}
PG_CLASSES = "ABCDEF"


def pg_class_from_L(L, z0=0.1):
    """Nearest Pasquill-Gifford class to (1/L, z0) on Golder's curves."""
    L = np.atleast_1d(_arr(L))
    z0 = np.broadcast_to(_arr(z0), L.shape)
    invL = 1.0 / L
    centres = np.stack([_GOLDER[c][0] + _GOLDER[c][1] * np.log10(z0) for c in PG_CLASSES])
    idx = np.argmin(np.abs(centres - invL[None, :]), axis=0)
    return np.array([PG_CLASSES[i] for i in idx])


# ---------------------------------------------------------------------------
# Particle physics (Kevin's "Particle Settling.doc", Heumann 1997)
# ---------------------------------------------------------------------------

def settling_velocity(d_um, rho_p=1500.0, T_k=288.0, shape_factor=1.0):
    """Stokes terminal settling velocity with Cunningham slip correction
    (m/s). d_um = aerodynamic diameter in micrometres.
        lambda = mu / (0.499 rho_g c_bar),  c_bar = sqrt(8RT/(pi M))
        Cc = 1 + (2.52 lambda/d)(1.23 + 0.41 exp(-0.44 d/lambda))
        vg = rho_p d^2 g Cc / (18 mu chi)
    (Kevin's STLS VBA used the same Cc form.)"""
    d = _arr(d_um) * 1e-6
    mu = 1.458e-6 * T_k ** 1.5 / (T_k + 110.4)       # Sutherland, kg/m/s
    cbar = np.sqrt(8 * 8.314 * T_k / (np.pi * 0.02897))
    lam = mu / (0.499 * RHO_AIR * cbar)
    Cc = 1.0 + (2.52 * lam / d) * (1.23 + 0.41 * np.exp(-0.44 * d / lam))
    return (rho_p - RHO_AIR) * d ** 2 * G * Cc / (18.0 * mu * shape_factor)
