"""Reduced-order acoustic-structure coupling: radiation impedance of a baffled
circular/square radiator, added as a medium-loading term on the plate equation.

Z_rad(ka) = rho_f * c_f * [R1(2ka) + i * X1(2ka)]   (piston approximation)

R1(x) = 1 - 2 J1(x)/x,   X1(x) = 2 H1(x)/x   (J1 Bessel, H1 Struve).

The loading enters the harmonic plate equation as an equivalent added mass
and damping: p_ac = -i * omega * Z_rad * W_avg / A_cell.
"""

from __future__ import annotations

from scipy.special import j1, struve


def radiation_impedance(omega: float, a_eff: float,
                        rho_f: float = 1000.0, c_f: float = 1500.0) -> complex:
    """Piston radiation impedance (per unit area) of an effective radius a_eff."""
    k = omega / c_f
    x = 2.0 * k * a_eff
    if x < 1e-9:
        return 0.0 + 0.0j
    R1 = 1.0 - 2.0 * j1(x) / x
    X1 = 2.0 * struve(1, x) / x
    return rho_f * c_f * (R1 + 1j * X1)


def acoustic_pressure_term(omega: float, w_avg: complex, a_eff: float,
                           rho_f: float = 1000.0, c_f: float = 1500.0) -> complex:
    """Equivalent acoustic back-pressure for average deflection amplitude w_avg."""
    Z = radiation_impedance(omega, a_eff, rho_f, c_f)
    v_avg = 1j * omega * w_avg  # velocity amplitude
    return -Z * v_avg
