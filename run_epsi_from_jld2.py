# run_epsi_from_jld2.py
# -----------------------------------------------------------------------------
# Load a JLD2 file with keys ['d_obs_multiple','nsrc','d_obs'], run EPSI on it.
# Requires: numpy, matplotlib, h5py; uses epsi_cli.py and/or epsi_operator.py you copied.
# -----------------------------------------------------------------------------
import argparse
import os
import numpy as np

# --- imports from the files you already copied ---
from epsi_cli import EPSI, fft_t, ifft_t, pad_time_top, chop_time_top, pad_r_right, fft_r, ifft_r, chop_r_left  # type: ignore
from epsi_operator import EpsiOperator  # linear surrogate 𝓔

# JLD2 reading via HDF5
try:
    import h5py
except Exception as e:
    raise SystemExit("Please install h5py:  pip install h5py")

# ------------------------ utilities ------------------------

def ricker(f0, dt, length_factor=6.0, nt=None):
    """Zero-phase Ricker; if nt is given, return causalized to length nt."""
    T = length_factor / f0
    tloc = np.arange(-T/2, T/2, dt)
    pf2  = (np.pi * f0)**2
    w    = (1.0 - 2.0 * pf2 * tloc**2) * np.exp(-pf2 * tloc**2)
    if nt is None:
        return w
    # causalize to put peak at t=0 and pad/crop to nt
    w = np.roll(w, w.size//2)
    out = np.zeros(nt, dtype=w.dtype)
    out[:min(w.size, nt)] = w[:min(w.size, nt)]
    return out

def estimate_wavelet_from_data(d, dt, fmin=5.0, fmax=None):
    """
    Crude wavelet estimate: peak frequency of the mean trace → Ricker.
    Works well enough to run EPSI when q is not provided in the file.
    """
    if fmax is None:
        fmax = 0.45 / dt  # avoid Nyquist/aliasing
    m = d.mean(axis=1)  # average over receivers
    nfft = 1 << (m.size-1).bit_length()
    M = np.fft.rfft(m, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        f0 = max(10.0, 0.25/dt)
    else:
        f0 = float(freqs[band][np.argmax(np.abs(M[band]))])
    return ricker(f0=f0, dt=dt, nt=d.shape[0])

def _deref_scalar_or_ref(f, maybe_ref):
    """Helper to deref JLD2 object-reference scalars."""
    if isinstance(maybe_ref, h5py.h5r.Reference):
        return f[maybe_ref][()]
    return maybe_ref

def _read_compound_objref_array(f, objref):
    """
    JLD2 stores compound fields as 'object references'. This handles:
      - a scalar compound with .dtype fields (…,'data','geometry',…)
      - a 1-element object array that points to the real ndarray
    Returns a numpy array.
    """
    arr0 = f[objref]
    if arr0.shape == ():  # scalar compound -> pull fields
        tup = arr0[()]  # tuple of object refs
        # expecting ('nsrc','geometry','data') order
        data_ref = tup[2]
        return _read_compound_objref_array(f, data_ref)

    if arr0.dtype == np.dtype('O'):
        # array of object refs; often length 1
        first_ref = arr0[0]
        return f[first_ref][()]
    else:
        return arr0[()]

def load_jld2_gather(path, key="d_obs"):
    """
    Load d (nt,nr) and dt (seconds) from a JLD2 file with top-level keys:
      ['d_obs_multiple', 'nsrc', 'd_obs'].
    We follow the nested object-ref layout.
    """
    with h5py.File(path, "r") as f:
        if key not in f.keys():
            raise KeyError(f"Key '{key}' not found. Available: {list(f.keys())}")
        # top-level is a scalar compound with fields ('nsrc','geometry','data')
        top = f[key][()]
        nsrc_ref, geom_ref, data_ref = top
        # nsrc
        nsrc = int(f[nsrc_ref][()])
        # geometry → dt (could be ms or s; infer units)
        geom = f[geom_ref][()]
        dt_field = f[geom[3]]  # ('dt' is 4th field)
        dt_arr = dt_field[()]
        # dt_arr can be shape (1,) float; turn into scalar
        dt_raw = float(np.array(dt_arr).reshape(-1)[0])
        # Heuristic: if dt_raw > 0.5, treat as milliseconds
        dt = dt_raw*1e-3 if dt_raw > 0.5 else dt_raw

        # data payload (might be nested object refs)
        d = _read_compound_objref_array(f, data_ref)
        # Make contiguous float64 for numerical stability
        d = np.ascontiguousarray(d.astype(np.float64))

    # Assume first dimension == time; if not, allow caller to swap
    if d.ndim != 2:
        raise ValueError(f"Expected 2D array (nt,nr); got shape {d.shape}")
    nt, nr = d.shape
    return d, dt, nsrc

# ------------------------ runner ------------------------

def run(args):
    d, dt_file, nsrc = load_jld2_gather(args.input, key=args.key)
    print(f"[loaded] {args.input} key={args.key} shape={d.shape}, nsrc={nsrc}, dt_file={dt_file:.6f}s")

    # If you know the correct dt, override from CLI; otherwise use the file value
    dt = args.dt if args.dt is not None else dt_file

    # Choose which array to use (you can also switch to 'd_obs_multiple')
    d_in = d

    # Wavelet: if you have one, load it; otherwise estimate
    if args.wavelet_npy and os.path.exists(args.wavelet_npy):
        q = np.load(args.wavelet_npy).astype(np.float64)
        if q.ndim != 1 or q.size != d_in.shape[0]:
            # causalize/pad if needed
            qq = np.zeros(d_in.shape[0]); L = min(qq.size, q.size)
            qq[:L] = q[:L]
            q = qq
        print(f"[wavelet] loaded {args.wavelet_npy} (len={q.size})")
    else:
        q = estimate_wavelet_from_data(d_in, dt=dt)
        print(f"[wavelet] estimated Ricker from data (len={q.size})")

    # MODE 1: full iterative EPSI (your CLI implementation)
    if args.mode == "epsi":
        epsi = EPSI(d=d_in, q=q, dt=dt,
                    topmute_ms=args.topmute_ms,
                    time_taper_frac=args.time_taper_frac,
                    n_iter=args.n_iter,
                    sparsity_start=args.sparsity_start,
                    sparsity_end=args.sparsity_end)
        # attach data for run()
        epsi.d = d_in
        print(f"[EPSI] running iterations: n_iter={args.n_iter}")
        X0_hat = epsi.run(n_iter=args.n_iter, verbose=True)

        # primaries = Q-only term (paper’s Gb Qb term)
        def Q_only(X0_t):
            X0m = X0_t * epsi.mute_mask
            X0f = fft_t(pad_time_top(X0m.astype(np.complex128), epsi.nt_conv))
            QX0f = epsi.Qf[:, None] * X0f
            return chop_time_top(ifft_t(QX0f), epsi.nt).real

        p_hat = Q_only(X0_hat)
        np.save(args.out_prefix + "_X0_hat.npy", X0_hat)
        np.save(args.out_prefix + "_p_hat.npy",  p_hat)
        print(f"[saved] {args.out_prefix}_X0_hat.npy  /  {args.out_prefix}_p_hat.npy")

    # MODE 2: linear surrogate primaries (one Landweber step):  p̂ ≈ 𝓔 d
    else:
        E = EpsiOperator(d=d_in, q=q, dt=dt,
                         topmute_ms=args.topmute_ms,
                         time_taper_frac=args.time_taper_frac,
                         alpha=None)  # compute Cauchy step automatically
        p_hat = E.forward(d_in)
        np.save(args.out_prefix + "_p_hat.npy", p_hat)
        print(f"[saved] {args.out_prefix}_p_hat.npy (linear surrogate)")

    # Simple report
    def nmse_db(est, ref):
        num = np.sum((est - ref)**2); den = np.sum(ref**2) + 1e-18
        return 10.0*np.log10(num/den)

    print(f"[report] input-rms={np.sqrt(np.mean(d_in**2)):.3e}, output-rms={np.sqrt(np.mean(p_hat**2)):.3e}")
    if args.plot:
        import matplotlib.pyplot as plt
        clip = 0.98 * np.max(np.abs(d_in))
        T = d_in.shape[0] * dt
        extent = [0, d_in.shape[1]-1, T, 0.0]
        fig, axs = plt.subplots(1, 2, figsize=(10,4), sharey=True, constrained_layout=True)
        axs[0].imshow(d_in, cmap="gray", vmin=-clip, vmax=clip, aspect="auto", extent=extent); axs[0].set_title("Observed d")
        axs[1].imshow(p_hat, cmap="gray", vmin=-clip, vmax=clip, aspect="auto", extent=extent); axs[1].set_title("Predicted primaries p̂")
        for ax in axs: ax.set_xlabel("receiver index")
        axs[0].set_ylabel("time (s)")
        if args.fig:
            plt.savefig(args.fig, dpi=160)
            print(f"[saved] figure → {args.fig}")
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run EPSI on a JLD2 dataset")
    ap.add_argument("--input", type=str, default="/mnt/data/nsrc=1.jld2", help="path to .jld2 file")
    ap.add_argument("--key", type=str, default="d_obs", choices=["d_obs", "d_obs_multiple"], help="which dataset key to read")
    ap.add_argument("--mode", type=str, default="epsi", choices=["epsi","surrogate"], help="full EPSI iterations or linear surrogate 𝓔")
    ap.add_argument("--dt", type=float, default=None, help="override dt [s]; default: use value from file")
    ap.add_argument("--wavelet-npy", type=str, default=None, help="optional path to q.npy (length T); if absent, estimate from data")
    # EPSI controls (taken from your CLI)
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--topmute-ms", type=float, default=40.0)
    ap.add_argument("--time-taper-frac", type=float, default=0.15)
    ap.add_argument("--sparsity-start", type=float, default=0.985)
    ap.add_argument("--sparsity-end", type=float, default=0.95)
    # Output / plot
    ap.add_argument("--out-prefix", type=str, default="epsi_run")
    ap.add_argument("--plot", action="store_true", help="show a quick figure")
    ap.add_argument("--fig", type=str, default=None, help="optional path to save the figure")
    args = ap.parse_args()
    run(args)
