import numpy as np

# threshold below which |z| uses the series expansion for z/x(z),
# derived from the agreement experiment (see notebook). Placeholder:
_Z_SMALL = 1e-3  # series is O(z^3)-accurate; agreement with raw ~1e-8 at the threshold, ample for vol fitting


def _z_over_x(z: np.ndarray, rho: float) -> np.ndarray:
    """The z / x(z) factor in Hagan's formula, with a series branch near z=0
    to avoid catastrophic cancellation. Tends to 1 as z -> 0."""
    z = np.asarray(z, dtype=float)

    # compute the raw formula on a SAFE z so the masked-out (small-z) entries
    # don't emit divide/log warnings, same trick as bsm_price's safe_vt
    z_safe = np.where(np.abs(z) > _Z_SMALL, z, 1.0)
    x = np.log((np.sqrt(1.0 - 2.0*rho*z_safe + z_safe**2) + z_safe - rho) / (1.0 - rho))
    raw = z_safe / x

    series = 1.0 + rho*z/2.0 + (2.0 - 3.0*rho**2)*z**2/12.0

    return np.where(np.abs(z) > _Z_SMALL, raw, series)

def sabr_atm_vol(F: float, T: float, alpha: float, beta: float,
                 rho: float, nu: float) -> float:
    """Hagan implied vol exactly at the money (K = F).

    The log(F/K) terms vanish, so this is the clean limit the general formula
    must converge to as K -> F.
    """
    one_minus_beta = 1.0 - beta
    F_pow = F ** one_minus_beta                      # F^(1-beta)

    term = (
        (one_minus_beta**2 / 24.0) * alpha**2 / F ** (2.0 * one_minus_beta)
        + 0.25 * rho * beta * nu * alpha / F_pow
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )
    return alpha / F_pow * (1.0 + term * T)


def sabr_vol(K: np.ndarray, F: float, T: float, alpha: float, beta: float,
             rho: float, nu: float) -> np.ndarray:
    """Black implied vol under SABR via Hagan's asymptotic formula.

    Branches to the ATM expression where K is within _Z_SMALL (in z) of the
    money, since the general formula divides by log(F/K).
    """
    K = np.asarray(K, dtype=float)
    one_minus_beta = 1.0 - beta

    log_FK = np.log(F / K)                            # log-moneyness, Hagan sign
    FK_pow = (F * K) ** (one_minus_beta / 2.0)        # (FK)^((1-beta)/2)

    # z and the z/x(z) factor (series-branched internally)
    z = (nu / alpha) * FK_pow * log_FK
    z_factor = _z_over_x(z, rho)

    # leading level term: alpha / [ (FK)^((1-b)/2) * (1 + (1-b)^2/24 log^2 + (1-b)^4/1920 log^4) ]
    denom_series = (
        1.0
        + (one_minus_beta**2 / 24.0) * log_FK**2
        + (one_minus_beta**4 / 1920.0) * log_FK**4
    )
    leading = alpha / (FK_pow * denom_series)

    # 1 + (...) T  correction
    correction = 1.0 + (
        (one_minus_beta**2 / 24.0) * alpha**2 / (F * K) ** one_minus_beta
        + 0.25 * rho * beta * nu * alpha / FK_pow
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    ) * T

    full = leading * z_factor * correction

    # near the money (small |z|), the general formula's log(F/K) denominators
    # vanish; branch to the exact ATM limit elementwise
    atm = sabr_atm_vol(F, T, alpha, beta, rho, nu)
    return np.where(np.abs(z) < _Z_SMALL, atm, full)