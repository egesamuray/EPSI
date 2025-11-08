# examples_synthetic.py
# -----------------------------------------------------------------------------
# Tiny synthetic: reflectivity → primaries P0, add water-layer multiples → data P,
# then recover P0 with 𝓔 = I + G (using true G here for verification).
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from epsi_operator import EpsiOperator

def ricker(f0, dt, length_factor=6.0):
    T = int(np.ceil(length_factor / f0 / dt))
    t = np.arange(-T//2, T//2) * dt
    pf2 = (np.pi * f0)**2
    w = (1 - 2 * pf2 * t**2) * np.exp(-pf2 * t**2)
    # causalize ('same' length pad to nt in main)
    return w

def synth(nt=512, dt=0.002, nr=5, f0=20.0, v=2000.0, t0s=(0.4, 0.8), amps=(1.0, -0.7),
          tau_w=0.06, mul_amp=0.7, noise=0.02, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(nt)*dt
    offs = np.linspace(-800.0, 800.0, nr)
    # reflectivity impulses (independent per trace here for simplicity)
    g = np.zeros((nt, nr))
    for j in range(nr):
        for A, t0 in zip(amps, t0s):
            it = int(round(t0/dt))
            if 0 <= it < nt:
                g[it, j] += A
    # wavelet
    w = ricker(f0, dt)
    wfull = np.zeros(nt); wfull[:min(nt, w.size)] = np.roll(w, w.size//2)[:min(nt, w.size)]
    # primaries P0 = g * w (per trace)
    def conv_same(h, x):
        y = np.convolve(x, h, mode='full')[:x.size]
        return y
    P0 = np.zeros_like(g)
    for j in range(nr):
        P0[:, j] = conv_same(wfull, g[:, j])
    # add simple water-layer multiples as delayed/scaled copies of P0
    dshift = int(round((2*tau_w)/dt))
    M = mul_amp*np.pad(P0, ((dshift,0),(0,0)))[:nt,:]
    M += (mul_amp**2)*np.pad(P0, ((2*dshift,0),(0,0)))[:nt,:]
    P = P0 + M + noise*np.std(P0)*rng.standard_normal(P0.shape)
    return t, offs, g, wfull, P0, P

def main():
    nt, dt, nr = 512, 0.002, 7
    t, offs, g, w, P0_true, P = synth(nt=nt, dt=dt, nr=nr)
    # Build operator with true G (for verification): 𝓔 = I + G
    E = EpsiOperator(g)
    P0_hat = E.forward(P)

    nmse = 10*np.log10(np.sum((P0_hat - P0_true)**2)/np.sum(P0_true**2) + 1e-20)
    print(f"NMSE(P0_hat vs P0_true) = {nmse:.2f} dB")

    # Minimal plot (optional)
    clip = 0.98*np.max(np.abs(P))
    fig, axs = plt.subplots(1,3, figsize=(12,3), sharex=True, sharey=True, constrained_layout=True)
    extent=[offs[0], offs[-1], t[-1], t[0]]
    axs[0].imshow(P, cmap='gray', aspect='auto', vmin=-clip, vmax=clip, extent=extent); axs[0].set_title("a) Data P")
    axs[1].imshow(P0_hat, cmap='gray', aspect='auto', vmin=-clip, vmax=clip, extent=extent); axs[1].set_title("b) 𝓔[P]")
    axs[2].imshow(P0_true, cmap='gray', aspect='auto', vmin=-clip, vmax=clip, extent=extent); axs[2].set_title("c) True P0")
    for ax in axs: ax.set_xlabel("Offset (m)")
    axs[0].set_ylabel("Time (s)")
    plt.show()

if __name__ == "__main__":
    main()
