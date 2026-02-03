from __future__ import annotations
from typing import Sequence
import cupy as cp
from cupyx.scipy.interpolate import PchipInterpolator
################################################################
# Functions used for stdNTS distribution
# This is just a python + GPU compatible translation of the code in Dr. YS Kim's TemStaR R package. 
# In the future the cleanliness and speed of this code can be improved by allowing tensor inputs for batch computation. 
################################################################
# Characteristic function & cumulative distribution function
def _derive_gamma(alpha: float, theta: float, beta: float):
    # type: ignore[override]
    """Return gamma such that Var[NTS] = 1 when only (alpha, theta, beta) are given."""
    return cp.sqrt(cp.abs(1 - beta ** 2 * (2 - alpha) / (2 * theta)))
  
def chf_stdNTS(u, param):
    '''
    Parameters
    ----------
    u : Array
        Uniform random numbers.
    param : List
        List of stdNTS parameters in the order alpha, theta, beta, gamma, mu, dt.
        
    Returns
    -------
    Array
        Contains phi(u) of equal dimension to u.
    '''
    u = cp.asarray(u)
    a, th, b = param[:3]
    g = param[3] if len(param) >= 4 else _derive_gamma(a, th, b)
    m = param[4] if len(param) >= 5 else 0.0
    dt = param[5] if len(param) >= 6 else 1.0
    term = th - 1j * (b * u + 1j * g ** 2 * u ** 2 / 2)
    coeff = 1j * (m - b) * u
    frac = 2 * th ** (1 - a / 2) / a
    return cp.exp(dt * (coeff - frac * (term ** (a / 2) - th ** (a / 2))))

def cdf_fft_gil_pelaez(arg, param, chf, *, dz=2**-12, m=2**17):
    '''
    Parameters
    ----------
    arg : Array
        Values of x to evaluate for which F(X <= x).
    param : List
        Parameters of the desired distribution in the order alpha, theta, beta, gamma, mu, dt.
    chf : Callable
        Characteristic function of the desired distribution.
    dz : float, defaults to 2**-12
        Step size of integral approximation. Smaller values lead to a better approximation.
    m : int, defaults to 2**17
        Sample size. Specifies how much of the function to approximate. Larger values lead to a better approximation.
    Returns
    -------
    Array
        Contains an approximation of CDF values corresponding to x within arg.
    Note: Small and large values of dz and m, respectively result in much larger memory requirements and slower computation. 

    '''
    k = cp.arange(m, dtype=cp.float64)
    z = dz * (k + 0.5)
    x = cp.pi * (2 * k - m + 1) / (m * dz)
    phi = chf(z, param)
    seq = (phi / z) * cp.exp(1j * cp.pi * (m - 1) / m * k)
    F_fft = 0.5 - (1 / cp.pi) * cp.imag(
        dz * cp.exp(1j * cp.pi * (m - 1) / (2 * m)) * cp.exp(-1j * cp.pi / m * k) * cp.fft.fft(seq)
    )
    pchip_interp = PchipInterpolator(x, F_fft)
    return pchip_interp(arg)
  
def cleanup_cdf(cdfret, argout):
    '''
    Parameters
    ----------
    cdfret : array
        Result of a CDF approximation.
    argout : array
        Corresponding x values of cdfret.

    Returns
    -------
    argout : array
        Cleaned up argout corresponding to cleaned up cdfret.
    cdfret : array
        Cleaned up CDF approximation.

    '''
    cdfret = cp.asarray(cdfret)
    argout = cp.asarray(argout)
    # Filter cdfret > 0
    mask = cdfret > 0
    cdfret = cdfret[mask]
    argout = argout[mask]
    # Filter cdfret < 1
    mask = cdfret < 1
    cdfret = cdfret[mask]
    argout = argout[mask]
    # Filter out where diff(cdfret) == 0
    if cdfret.size > 1:
        diff_mask = cp.diff(cdfret) != 0
        cdfret = cdfret[cp.append(diff_mask, True)]  # Keep last element
        argout = argout[cp.append(diff_mask, True)]
    # Remove non-increasing differences
    while cdfret.size > 1 and cp.any(cp.diff(cdfret) <= 0):
        diff_mask = cp.diff(cdfret) > 0
        cdfret = cdfret[cp.append(diff_mask, True)]
        argout = argout[cp.append(diff_mask, True)]
    return argout, cdfret
################################################################
# Random number generation
def ipnts(u, ntsparam, *, maxmin=[-10, 10], du=0.01, dz=2**-12, m=2**17):
    '''
    Parameters
    ----------
    u : Array
        Uniform random number(s).
    ntsparam : List
        List of stdNTS parameters in the order alpha, theta, beta, gamma, mu, dt.
    maxmin : list
        Contains two x values expressing the lower and upper bound -1 for which to approximate the CDF. The default is [-10, 10].
    du : float
        The step size corresponding to maxmin of the x values to approximate the CDF. The default is 0.01.
    dz : float, defaults to 2**-12
        Step size of integral approximation. Smaller values lead to a better approximation.
    m : int, defaults to 2**17
        Sample size. Specifies how much of the function to approximate. Larger values lead to a better approximation.

    Returns
    -------
    Array
        Array of stdNTS numbers.

    '''
    # Plug uniform random nums into inverse formula
    arg = cp.arange(maxmin[0], maxmin[1], du, dtype=cp.float64)
    cdf = cdf_fft_gil_pelaez(arg, ntsparam, chf_stdNTS, dz=dz, m=m)
    x, y = cleanup_cdf(cdf, arg)
    # Inverse CDF
    pchip_interp = PchipInterpolator(y, x)
    x_out = pchip_interp(u)
    return (x_out.reshape(u.shape))

def rnts(n, ntsparam: Sequence[float]):
    '''
    Parameters
    ----------
    n : int
        Number of random numbers to generate.
    ntsparam : Sequence[float]
        Contains ntsparameters in the order alpha, theta, beta, gamma, mu, dt.

    Returns
    -------
    Array
        Wrapper generating stdNTS distributed random numbers.

    '''
    u = cp.random.random(n).astype(cp.float64)
    return ipnts(u, ntsparam)
