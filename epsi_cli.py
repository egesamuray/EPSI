#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
epsi_cli.py — Tiny EPSI (Estimating Primaries by Sparse Inversion) demo CLI
Author: <your name>

Model (per van Groenestijn & Verschuur, 2009):  d ≈ Q(X0) + P(X0)
- Q-term: time convolution with source wavelet (Ricker); applied via FFT over time
- P-term: spatial convolution over receivers for each temporal frequency; FFT over receivers
- Top-mute: applied to X0(t) to avoid trivial source leakage
- Update: steepest-descent (Cauchy step) with sparse update (soft threshold)

Dependencies: numpy, matplotlib (no SciPy)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Common helpers -----------------------------

def ricker(f0, dt, length_factor=6.0):
    """Zero-phase Ricker; w(t)=(1-2(pi f0 t)^2)exp(-(pi f0 t)^2)."""
    T = length_factor / f0
    tloc = np.arange(-T/2, T/2, dt)
    pf2  = (np.pi * f0)**2
    w    = (1.0 - 2.0 * pf2 * tloc**2) * np.exp(-pf2 * tloc**2)
    return w

def causalize_zero_phase(w, nt):
    """Shift zero-phase wavelet so the peak is at t=0, then pad/crop to nt."""
    w = np.asarray(w)
    L = w.size
    wc = np.roll(w, L//2)  # move center to index 0
    out = np.zeros(nt, dtype=wc.dtype)
    out[:min(L, nt)] = wc[:min(L, nt)]
    return out

def tukey_1d(n, frac):
    if frac <= 0: return np.ones(n)
    if frac >= 1: return np.hanning(n)
    w = np.ones(n)
    L = int(np.floor(frac*(n-1)/2.0))
    if L > 0:
        u = np.arange(L)/L
        w[:L]  = 0.5*(1-np.cos(np.pi*u))
        w[-L:] = w[:L][::-1]
    return w

def shift_right(x, nshift, axis=0):
    """Zero-filled shift along axis; positive nshift → later times/indices."""
    x = np.asarray(x); y = np.zeros_like(x)
    n = x.shape[axis]
    if nshift == 0: return x.copy()
    if nshift > 0:
        if nshift >= n: return y
        src = [slice(None)]*x.ndim; dst = [slice(None)]*x.ndim
        src[axis] = slice(0, n-nshift); dst[axis] = slice(nshift, None)
        y[tuple(dst)] = x[tuple(src)]
    else:
        nshift = -nshift
        if nshift >= n: return y
        src = [slice(None)]*x.ndim; dst = [slice(None)]*x.ndim
        src[axis] = slice(nshift, None); dst[axis] = slice(0, n-nshift)
        y[tuple(dst)] = x[tuple(src)]
    return y

def pinfo(name, x):
    rms = float(np.sqrt(np.mean(np.abs(x)**2)))
    print(f"{name:>10s}: shape={tuple(x.shape)}, rms={rms:.3e}, max={np.max(np.abs(x)):.3e}")

def soft_thresh(x, tau):
    m = np.abs(x)
    return np.sign(x) * np.maximum(m - tau, 0.0)

# ----------------------------- EPSI kernels (FFT stacks) -----------------------------

def pad_time_top(x, nt_out):
    y = np.zeros((nt_out,) + x.shape[1:], dtype=np.complex128)
    y[:x.shape[0], ...] = x
    return y

def chop_time_top(x, nt_out):
    return x[:nt_out, ...]

def fft_t(x):   return np.fft.fft(x, axis=0, norm="ortho")
def ifft_t(X):  return np.fft.ifft(X, axis=0, norm="ortho")
def fft_r(x):   return np.fft.fft(x, axis=1, norm="ortho")
def ifft_r(X):  return np.fft.ifft(X, axis=1, norm="ortho")

def pad_r_right(x, nr_out):
    y = np.zeros((x.shape[0], nr_out), dtype=np.complex128)
    y[:, :x.shape[1]] = x
    return y

def chop_r_left(x, nr_out):
    return x[:, :nr_out]

# ----------------------------- EPSI forward/adjoint -----------------------------

class EPSI:
    def __init__(self, d, q, dt, topmute_ms=40.0, time_taper_frac=0.15,
                 n_iter=60, sparsity_start=0.985, sparsity_end=0.95):
        """
        d: data (nt, nr), real
        q: wavelet time series (nt,), real
        """
        # Shapes & sizes
        assert d.ndim == 2 and q.ndim == 1
        self.nt, self.nr = d.shape
        self.dt = float(dt)
        self.nt_conv = 2*self.nt
        self.nr_conv = 2*self.nr

        # Mute mask (applied to X0 in time)
        topmute = int(round(topmute_ms/1000.0/self.dt))
        self.mute_mask = np.ones((self.nt, 1)); self.mute_mask[:topmute, 0] = 0.0

        # Finite-FFT master window for building P
        wstart = int(np.ceil(0.8*(topmute+1))) - 1
        wstart = max(0, min(wstart, self.nt-1))
        win_t  = np.zeros(self.nt); win_t[wstart:] = tukey_1d(self.nt - wstart, time_taper_frac)
        Dw     = (d * win_t[:, None]).astype(np.complex128)

        # Precompute Q(ω) and D(ω,κ) kernels
        self.Qf = np.fft.fft(np.pad(q.astype(np.complex128), (0, self.nt_conv-self.nt)), norm="ortho")  # (nt_conv,)
        Dtf     = fft_t(pad_time_top(Dw, self.nt_conv))  # (nt_conv, nr)
        Dtf_pad = pad_r_right(Dtf, self.nr_conv)         # (nt_conv, nr_conv)
        self.Dk = fft_r(Dtf_pad)                         # (nt_conv, nr_conv)

        # Iteration controls
        self.n_iter = int(n_iter)
        self.sparsity_start = float(sparsity_start)
        self.sparsity_end   = float(sparsity_end)

    # Forward: A(X0) = Q(X0) + P(X0)
    def forward(self, X0_t):
        X0m = X0_t * self.mute_mask
        X0f = fft_t(pad_time_top(X0m.astype(np.complex128), self.nt_conv))     # (nt_conv, nr)
        # Q-term
        QX0f = self.Qf[:, None] * X0f                                          # (nt_conv, nr)
        # P-term: spatial conv via FFT along receivers
        X0f_pad = pad_r_right(X0f, self.nr_conv)
        X0k     = fft_r(X0f_pad)
        PX0_pad = ifft_r(X0k * self.Dk)
        PX0f    = chop_r_left(PX0_pad, self.nr)
        Yf = QX0f + PX0f
        y_t = chop_time_top(ifft_t(Yf), self.nt).real
        return y_t

    # Adjoint: A^H(r) = Q^H(r) + P^H(r)
    def adjoint(self, res_t):
        Resf = fft_t(pad_time_top(res_t.astype(np.complex128), self.nt_conv))
        # Q^H
        GQf = np.conj(self.Qf)[:, None] * Resf
        # P^H
        Resf_pad = pad_r_right(Resf, self.nr_conv)
        Rk       = fft_r(Resf_pad)
        GP_pad   = ifft_r(Rk * np.conj(self.Dk))
        GPf      = chop_r_left(GP_pad, self.nr)
        Gf = GQf + GPf
        g_t = chop_time_top(ifft_t(Gf), self.nt).real
        return g_t * self.mute_mask

    # Run EPSI iterations (steepest-descent with sparse update)
    def run(self, n_iter=None, verbose=True):
        if n_iter is None:
            n_iter = self.n_iter
        X0 = np.zeros((self.nt, self.nr))
        for it in range(1, n_iter+1):
            d_hat = self.forward(X0)
            r_t   = self.d - d_hat
            g     = self.adjoint(r_t)

            # Cauchy step
            Ag    = self.forward(g)
            num   = np.sum(g*g)
            den   = np.sum(Ag*Ag) + 1e-12
            alpha = float(num/den)
            X0_new = X0 + alpha * g

            # sparse update schedule
            q_lo = self.sparsity_start + (self.sparsity_end - self.sparsity_start) * (it-1)/max(1, n_iter-1)
            tau  = np.quantile(np.abs(X0_new), q_lo)
            X0   = soft_thresh(X0_new, 0.5*tau)

            if verbose and (it % max(1, n_iter//5) == 0):
                print(f"[iter {it:03d}] alpha={alpha:.3e}, thresh~{0.5*tau:.3e}")

        return X0

# ----------------------------- Synthetic generator -----------------------------

def synth(nt=1024, dt=0.002, nr=61, f0=20.0, v=2000.0,
          t0s=(0.45, 0.70), amps=(0.8, -0.7), tau_w=0.06, a=0.7, noise_level=0.05, seed=7):
    rng = np.random.default_rng(seed)
    t   = np.arange(nt)*dt
    offsets = np.linspace(-1000.0, 1000.0, nr)

    # Reflectivity spikes with hyperbolic moveout
    r = np.zeros((nt, nr))
    for ir, x in enumerate(offsets):
        for A, t0 in zip(amps, t0s):
            tx = np.sqrt(t0**2 + (x/v)**2)
            it = int(round(tx/dt))
            if 0 <= it < nt: r[it, ir] += A

    # Wavelet
    q = causalize_zero_phase(ricker(f0, dt), nt)

    # Primaries via time-FFT conv
    nt_conv = 2*nt
    X = np.fft.fft(np.pad(r, ((0, nt_conv-nt),(0,0))), axis=0)
    Q = np.fft.fft(np.pad(q.astype(np.complex128), (0, nt_conv-nt)))
    P = np.fft.ifft(X * Q[:, None], axis=0).real[:nt, :]

    # Multiples (first & second order)
    dshift = int(round((2.0*tau_w)/dt))
    M = a*shift_right(P, dshift, axis=0) + (a**2)*shift_right(P, 2*dshift, axis=0)

    # Data with noise
    sig  = float(np.sqrt(np.mean(P**2)))
    d    = P + M + noise_level*sig*rng.standard_normal((nt, nr))

    return t, offsets, q, r, P, M, d

# ----------------------------- Plotting -----------------------------

def make_figure(d, p_hat, p_true, offsets, dt, outpng=None, show=True, title_suffix=""):
    tmax = d.shape[0]*dt
    clip = 0.98*np.max(np.abs(d))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, constrained_layout=True)
    axs[0].imshow(d,     cmap='gray', aspect='auto', vmin=-clip, vmax=clip, extent=[offsets[0], offsets[-1], tmax, 0.0]); axs[0].set_title(f"a) Data{title_suffix}")
    axs[1].imshow(p_hat, cmap='gray', aspect='auto', vmin=-clip, vmax=clip, extent=[offsets[0], offsets[-1], tmax, 0.0]); axs[1].set_title("b) EPSI primaries")
    axs[2].imshow(p_true,cmap='gray', aspect='auto', vmin=-clip, vmax=clip, extent=[offsets[0], offsets[-1], tmax, 0.0]); axs[2].set_title("c) True primaries")
    for ax in axs: ax.set_xlabel("Offset (m)")
    axs[0].set_ylabel("Time (s)")
    if outpng:
        plt.savefig(outpng, dpi=160)
        print(f"[saved] {outpng}")
    if show:
        plt.show()
    else:
        plt.close(fig)

# ----------------------------- Metrics -----------------------------

def nmse_db(est, ref):
    num = np.sum((est - ref)**2); den = np.sum(ref**2) + 1e-12
    return 10.0*np.log10(num/den)

# ----------------------------- CLI -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="EPSI (Estimating Primaries by Sparse Inversion) demo script")
    # Modes
    ap.add_argument("--mode", choices=["synthetic","from-npy"], default="synthetic",
                    help="synthetic: build toy gather; from-npy: load d.npy (nt×nr) and q.npy (nt,)")
    # Common controls
    ap.add_argument("--dt", type=float, default=0.002, help="sampling interval [s]")
    ap.add_argument("--topmute-ms", type=float, default=40.0, help="top-mute on X0(t) [ms]")
    ap.add_argument("--n-iter", type=int, default=60, help="EPSI iterations")
    ap.add_argument("--sparsity-start", type=float, default=0.985, help="quantile kept at iter=1 (e.g., 0.985 → top 1.5%)")
    ap.add_argument("--sparsity-end", type=float, default=0.95, help="quantile kept at last iter (e.g., 0.95 → top 5%)")
    ap.add_argument("--time-taper-frac", type=float, default=0.15, help="Tukey fraction for P-term master window")
    # Synthetic params
    ap.add_argument("--nt", type=int, default=1024)
    ap.add_argument("--nr", type=int, default=61)
    ap.add_argument("--f0", type=float, default=20.0, help="Ricker peak freq [Hz]")
    ap.add_argument("--tau-w", type=float, default=0.06, help="water-layer two-way time [s]")
    ap.add_argument("--a", type=float, default=0.7, help="free-surface reflection coeff (1st order)")
    ap.add_argument("--noise", type=float, default=0.05, help="white noise level relative to primary RMS")
    ap.add_argument("--seed", type=int, default=7)
    # NPZ/NPY input mode
    ap.add_argument("--input-d", type=str, help="path to d.npy (nt×nr)")
    ap.add_argument("--input-q", type=str, help="path to q.npy (nt,)")
    # Outputs
    ap.add_argument("--outfig", type=str, default="epsi_result.png", help="output PNG figure")
    ap.add_argument("--save-npy-prefix", type=str, default=None, help="prefix to save X0_hat/p_hat/m_hat as .npy")
    ap.add_argument("--no-show", action="store_true", help="do not open matplotlib window")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.mode == "synthetic":
        t, offsets, q, r, p_true, m_true, d = synth(
            nt=args.nt, dt=args.dt, nr=args.nr, f0=args.f0,
            tau_w=args.tau_w, a=args.a, noise_level=args.noise, seed=args.seed
        )
        print("### Synthesis")
        pinfo("p_true", p_true); pinfo("m_true", m_true); pinfo("d", d)
    else:
        assert args.input_d and args.input_q, "--input-d and --input-q are required for from-npy mode"
        d = np.load(args.input_d)  # (nt, nr)
        q = np.load(args.input_q)  # (nt,)
        assert d.ndim==2 and q.ndim==1, "d must be (nt,nr), q must be (nt,)"
        nt, nr = d.shape
        t = np.arange(nt)*args.dt
        offsets = np.arange(nr)    # generic index offsets
        p_true = None; m_true = None
        print("### Loaded")
        pinfo("d", d); pinfo("q", q)

    # Build EPSI object and attach data (store d for internal use)
    epsi = EPSI(d=d, q=q, dt=args.dt, topmute_ms=args.topmute_ms,
                time_taper_frac=args.time_taper_frac, n_iter=args.n_iter,
                sparsity_start=args.sparsity_start, sparsity_end=args.sparsity_end)
    # quick hack: store d for run()
    epsi.d = d

    print("### EPSI running ...")
    X0_hat = epsi.run(n_iter=args.n_iter, verbose=True)

    # Separate Q-only and P-only for outputs
    def Q_only(X0_t):
        X0m = X0_t * epsi.mute_mask
        X0f = fft_t(pad_time_top(X0m.astype(np.complex128), epsi.nt_conv))
        QX0f = epsi.Qf[:, None] * X0f
        return chop_time_top(ifft_t(QX0f), epsi.nt).real

    def P_only(X0_t):
        X0m = X0_t * epsi.mute_mask
        X0f = fft_t(pad_time_top(X0m.astype(np.complex128), epsi.nt_conv))
        X0f_pad = pad_r_right(X0f, epsi.nr_conv)
        X0k = fft_r(X0f_pad)
        PX0_pad = ifft_r(X0k * epsi.Dk)
        PX0f = chop_r_left(PX0_pad, epsi.nr)
        return chop_time_top(ifft_t(PX0f), epsi.nt).real

    p_hat = Q_only(X0_hat)
    m_hat = P_only(X0_hat)

    # Metrics
    if p_true is not None:
        nmse_in  = nmse_db(d, p_true)
        nmse_out = nmse_db(p_hat, p_true)
        snr_in   = 10.0*np.log10(np.sum(p_true**2) / (np.sum((d - p_true)**2) + 1e-12))
        snr_out  = 10.0*np.log10(np.sum(p_true**2) / (np.sum((p_hat - p_true)**2) + 1e-12))
        print("\n### Metrics (synthetic)")
        print(f"NMSE_in  (d vs p_true):     {nmse_in:7.2f} dB")
        print(f"NMSE_out (p_hat vs p_true): {nmse_out:7.2f} dB")
        print(f"SNR_in   (before):          {snr_in:7.2f} dB")
        print(f"SNR_out  (after):           {snr_out:7.2f} dB")

    # Save arrays
    if args.save_npy_prefix:
        np.save(args.save_npy_prefix + "_X0_hat.npy", X0_hat)
        np.save(args.save_npy_prefix + "_p_hat.npy", p_hat)
        np.save(args.save_npy_prefix + "_m_hat.npy", m_hat)
        print(f"[saved] {args.save_npy_prefix}_X0_hat.npy / p_hat.npy / m_hat.npy")

    # Figure
    make_figure(d, p_hat, p_true if p_true is not None else p_hat*0.0, offsets,
                dt=args.dt, outpng=args.outfig, show=(not args.no_show),
                title_suffix=" (multiples+noise)")

if __name__ == "__main__":
    main()
