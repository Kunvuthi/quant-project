import numpy as np
from models.heston import heston_char_func


def carr_madan_psi(
    u: np.ndarray,
    S0: float, v0: float,
    kappa: float, theta: float, xi: float, rho: float, r: float,
    tau: float, alpha: float,
) -> np.ndarray:
    """
    The damped, Fourier-transformed call price psi(u), Carr & Madan (1999).
    Evaluates heston_char_func at the SHIFTED complex argument u - (alpha+1)i,
    the whole reason the damping trick works: this shift is what makes the
    integral converge in the first place.
    """
    shifted_u = u - (alpha + 1) * 1j
    phi_shifted = heston_char_func(shifted_u, S0, v0, kappa, theta, xi, rho, r, tau)
    denom = alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u
    return np.exp(-r * tau) * phi_shifted / denom


def carr_madan_call_prices(
    S0: float, v0: float,
    kappa: float, theta: float, xi: float, rho: float, r: float,
    tau: float,
    alpha: float = 1.5,
    N: int = 4096,      # FFT points, power of 2
    du: float = 0.25,   # frequency-grid spacing, controls truncation error
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carr-Madan FFT pricer. Returns (strikes, call_prices), a full strip at once,
    this method is inherently a batch method, there's no "single strike" version
    the way COS has one, the FFT produces the whole strip in one shot.
    """
    # 1. frequency grid: u_j = j*du for j = 0, ..., N-1
    u = np.arange(0,N) * du

    # 2. dk from the sampling identity you just derived: du * dk = 2*pi/N
    dk = (2*np.pi)/(N*du)

    # 3. log-strike grid, centered near ln(S0) so the useful strikes land
    #    in the middle of the FFT output rather than at the noisy edges
    k0 = np.log(S0) - N*dk/2
    k = k0 + np.arange(N)*dk

    # 4. Simpson's rule weights for better integral accuracy than a plain
    #    Riemann sum, standard 1/3, 4/3, 2/3, 4/3, ... pattern, weight[0] gets
    #    an extra 1/2 (trapezoidal correction at the u=0 endpoint)
    simpson_weights = (3 + (-1)**np.arange(N)) / 3
    simpson_weights[0] = simpson_weights[0] / 2  

    # 5. build psi(u) and the FFT input, including the e^{i*u*k0} phase shift
    #    needed because the FFT assumes the output starts at index 0, not at
    #    your chosen k0
    psi_vals = carr_madan_psi(u, S0, v0, kappa, theta, xi, rho, r, tau, alpha)
    fft_input = np.exp(1j * u * (-k0)) * psi_vals * du * simpson_weights

    # 6. FFT, then undo the damping factor e^{-alpha*k} to recover real prices
    fft_output = np.fft.fft(fft_input)
    call_prices = (np.exp(-alpha * k) / np.pi) * fft_output.real

    # 7. convert log-strikes back to strikes
    strikes = np.exp(k)

    return strikes, call_prices