import numpy as np
import polars as pl
from scipy.stats import qmc
import cupy as cp
from stdnts_dist import rnts, chf_stdNTS
import os
###########################################################
def free_gpu_cache():
    """a helper function to free up memory on the GPU manually."""
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
###########################################################
# Path generation functions #
def gensamplesigmapaths_gpu(sample_errors, kappa, xi, lam, zeta, sigma0):
    """Vectorised sigma‑path generator (Python loop, backend math)."""
    errors = cp.asarray(sample_errors)
    npath, ntimestep = errors.shape
    sigma_sq = cp.empty_like(errors)
    sigma_sq[:, 0] = sigma0 ** 2
    for t in range(1, ntimestep):
        err = errors[:, t - 1] - lam
        sigma_sq[:, t] = kappa + xi * sigma_sq[:, t - 1] * err ** 2 + zeta * sigma_sq[:, t - 1]
    return cp.sqrt(sigma_sq)

def genrfsamplestdNTSprices_gpu(alpha, theta, beta, gamma, kappa, xi, lam, zeta, sigma0, S0, *, y0 = 0.0, npath = 1_000, ntimestep = 250, dt = 1 / 252, r = 1 / 250, d = 0.0, index ):
    d = d * dt
    r = r * dt
    ntsparam = [alpha, theta, beta, gamma, 0.0]
    try:
        errors = rnts(npath * ntimestep, ntsparam).reshape((npath, ntimestep))
    except:
        print(ntsparam)
    sigma = gensamplesigmapaths_gpu(errors, kappa, xi, lam, zeta, sigma0)
    w = cp.log(chf_stdNTS(-1j * sigma, [alpha, theta, beta, gamma, 0.0, dt]))
    rtn = r - d - cp.real(w) + sigma * errors
    rtn[:, 0] = y0
    log_price = cp.cumsum(rtn, axis=1)
    return S0 * cp.exp(log_price)
  
def gensamplerfoptionprices_gpu( sample_prices, *, r, moneyness, dt = 1/252 ) -> tuple[np.ndarray, cp.ndarray]:
    r = r * dt
    t = sample_prices.shape[1]
    disc = cp.exp(-r * cp.arange(t))
    strike = sample_prices[:, 0] * moneyness
    S_T = sample_prices[:, -1]
    call = disc[-1] * cp.mean(cp.maximum(S_T - strike, 0))
    put = disc[-1] * cp.mean(cp.maximum(strike - S_T, 0))
    return call, put
###########################################################
def simulateHaltonVectors(n=9000000, sim_B=False):
    n += 20 # Add 20 extra rows for later discarding (fix for halton dependence issue)
    dim = 12 if sim_B else 11
    # Generate Halton sequence for main parameters
    sampler = qmc.Halton(d=dim, scramble=True)
    halton_points = sampler.random(n)
    # Generate an independent uniform(0,0.03) dividend column
    rng = np.random.default_rng(seed=12345) # Fixed seed for reproducibility
    dividend = rng.uniform(0, 0.03, size=n)
    # 1. alpha: Uniform(0,1) -> (0,2)
    halton_points[:, 0] = 2 * halton_points[:, 0]
    # 2. theta: Exponential with mean 1.2544 (rate = 1/mean)
    halton_points[:, 1] = -np.log(1 - halton_points[:, 1]) * 1.2544
    # 3. a1: Uniform(-1,1), no exact 0s
    halton_points[:, 2] = 2 * halton_points[:, 2] - 1
    halton_points[:, 2][halton_points[:, 2] == 0] = np.finfo(float).eps
    # 4. Moneyness ~ Uniform(0.5,1.5)
    halton_points[:, 3] = halton_points[:, 3] + 0.5
    exact_1_indices = np.where(halton_points[:, 3] == 1.0)[0]
    if exact_1_indices.size > 0:
        halton_points[exact_1_indices, 3] += np.random.uniform(-0.5, 0.5, size=exact_1_indices.size)
    # 5. Tao ~ Uniform(0.4,1.0)
    halton_points[:, 4] = 0.6 * halton_points[:, 4] + 0.4
    # 6. kappa: Exponential with mean 1
    halton_points[:, 5] = -np.log(1 - halton_points[:, 5])
    # 7. xi: Uniform(0,1) (unchanged)
    # 8. zeta = u * (1 - xi)
    halton_points[:, 7] = halton_points[:, 7] * (1 - halton_points[:, 6])
    # 9. sigma_error: Uniform(-0.05405997595, 0.05405997595), no exact 0s
    halton_points[:, 8] = 0.05405997595 * 2 * halton_points[:, 8] - 0.05405997595
    halton_points[:, 8][halton_points[:, 8] == 0] = np.finfo(float).eps
    # 10. lambda: Uniform(0,0.8)
    halton_points[:, 9] = 0.8 * halton_points[:, 9]
    # 11. rf: Uniform(0.0001,0.05)
    halton_points[:, 10] = 0.0001 + (0.05 - 0.0001) * halton_points[:, 10]
    # Append dividend column after rf (index 11), before sim_B columns
    if sim_B:
        # B: Uniform(-1,1) + beta and gamma
        halton_points[:, 11] = 2 * halton_points[:, 11] - 1
        halton_points[:, 11][halton_points[:, 11] == 0] = np.finfo(float).eps
        beta = halton_points[:, 11] * np.sqrt(2 * halton_points[:, 1] / (2 - halton_points[:, 0]))
        gamma = 1 - halton_points[:, 11] ** 2
        halton_points = np.hstack([
            halton_points, beta[:, None], gamma[:, None]
        ])
        # Insert dividend column
        halton_points = np.hstack([halton_points[:, :11], dividend[:, None], halton_points[:, 11:]])
        columns = [
            "alpha", "theta", "a1", "moneyness", "tao", "kappa", "xi", "zeta", "sigma_error", "lambda", "rf", "dividend", "B", "betas", "gammas"
        ]
    else:
        # Insert dividend column after rf (index 11)
        halton_points = np.hstack([halton_points[:, :11], dividend[:, None]])
        columns = [
            "alpha", "theta", "a1", "moneyness", "tao", "kappa", "xi", "zeta", "sigma_error", "lambda", "rf", "dividend"
        ]
    # Drop first 20 rows
    halton_points = halton_points[20:, :]
    # Create polars DataFrame
    df = pl.DataFrame(halton_points, schema=columns)
    # Add index column
    df = df.with_columns(pl.Series("index", np.arange(1, df.height + 1)))
    df = df.select(["index"] + columns)  # Reorder columns
    return df
###############################################################
def worker_chunk_gpu( chunk: pl.DataFrame, *, npath: int, sigma0: float, S0: float, y0: float, ) -> pl.DataFrame:
    records = []
    for row in chunk.iter_rows(named=True):
        prices = genrfsamplestdNTSprices_gpu(
            alpha=row["alpha"],
            theta=row["theta"],
            beta=row["betas"] if "betas" in row else row["a1"],
            gamma=row["gammas"] if "gammas" in row else 1.0 - row["a1"]**2,
            kappa=row["kappa"],
            xi=row["xi"],
            lam=row["lambda"],
            zeta=row["zeta"],
            sigma0=sigma0,
            S0=S0,
            y0=y0,
            npath=npath,
            ntimestep=int(np.ceil(row["tao"] * 250)),
            r=row["rf"],
            d=row["dividend"],
            index=int(row["index"]),
        )
        call, put = gensamplerfoptionprices_gpu(prices, r=row["rf"], moneyness=row["moneyness"])
        out = {**row, "call_price": float(cp.asnumpy(call) / S0), "put_price": float(cp.asnumpy(put) / S0)}
        records.append(out)
    free_gpu_cache()
    return pl.DataFrame(records)
################################################################
def stdNTSoptionmontecarlo( n_sim: int, chunk_size: int, output_dir: str, *, npath: int = 20_000, r: float = 0.02 / 250, S0: float = 1.0, y0: float = 0.0, sigma0: float = 9.6e-3, nstart: int = 0, overwrite: bool = False, ) -> None:
    halton = simulateHaltonVectors(n=n_sim, sim_B=True).slice(nstart, 1000)
    os.makedirs(output_dir, exist_ok=True)
    nchunks = int(np.ceil(len(halton) / chunk_size))
    for idx in range(nchunks):
        i0, i1 = idx * chunk_size, min((idx + 1) * chunk_size, len(halton))
        chunk_df = halton.slice(i0, i1 - i0)
        df_out = worker_chunk_gpu(
            chunk_df, npath=npath, sigma0=sigma0, S0=S0, y0=y0
        )
        fname = os.path.join(
            output_dir, f"stdNTSoptionpricemcs_{i0 + nstart}_{i1 + nstart}.csv"
        )
        if not overwrite and os.path.exists(fname):
            raise FileExistsError(f"{fname} exists – use overwrite=True to replace.")
        df_out.write_csv(fname)
        print(f"[GPU] wrote {fname} ({idx + 1}/{nchunks})")
    print("Monte‑Carlo simulation complete.")
