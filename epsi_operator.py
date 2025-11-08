# epsi_operator.py
# -----------------------------------------------------------------------------
# Predicted-primaries operator 𝓔 for EPSI:  P0 = (I - G R) P  with R = -I → 𝓔 = I + G
# Matrix-free (LinearOperator) and explicit sparse matrix (toy) implementations.
# -----------------------------------------------------------------------------
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csc_matrix, block_diag
from scipy.linalg import toeplitz

def _as_2d(x):
    x = np.asarray(x)
    return x if x.ndim == 2 else x.reshape(x.shape[0], 1)

def _build_toeplitz_causal(g):
    """
    Build causal Toeplitz conv matrix T(g) for length-T signals with 'same' padding.
    First column = [g0, g1, ..., g_{T-1}]^T ; first row = [g0, 0, ..., 0].
    """
    g = np.asarray(g).ravel()
    T = g.size
    c = np.zeros(T, dtype=np.complex128); c[:T] = g
    r = np.zeros(T, dtype=np.complex128); r[0] = g[0] if T > 0 else 0.0
    return toeplitz(c, r)  # dense; caller may wrap as sparse

def _conv1d_causal_same(g, x):
    """
    Linear convolution (causal, 'same' length). g, x are 1D arrays length T.
    """
    g = np.asarray(g).ravel().astype(np.float64)
    x = np.asarray(x).ravel().astype(np.float64)
    T = x.size
    y = np.zeros_like(x, dtype=np.float64)
    for t in range(T):
        # y[t] = sum_{k=0}^{t} g[k] * x[t-k]
        kmax = min(t, g.size - 1)
        if kmax >= 0:
            y[t] = np.dot(g[:kmax+1], x[t-kmax:t+1][::-1])
    return y

def _corr1d_causal_same(g, x):
    """
    Adjoint of _conv1d_causal_same: correlation with time-reversed kernel (same length).
    """
    g = np.asarray(g).ravel().astype(np.float64)
    x = np.asarray(x).ravel().astype(np.float64)
    T = x.size
    y = np.zeros_like(x, dtype=np.float64)
    # Equivalent to convolution with reversed kernel
    for t in range(T):
        kmax = min(T-1-t, g.size - 1)
        if kmax >= 0:
            y[t] = np.dot(g[:kmax+1], x[t:t+kmax+1])
    return y

class EpsiOperator:
    """
    EPSI predicted-primaries operator 𝓔 acting on time × receiver data:
        𝓔[P] = (I - G R) P, with R = -I  → 𝓔 = I + G.
    In this toy, G is per-trace causal time convolution with kernel g[:, ir].
    """
    def __init__(self, g, R_sign=-1):
        """
        Parameters
        ----------
        g : array, shape (T,) or (T, NR)
            Estimated surface-free Green’s impulse responses per receiver (time-domain).
        R_sign : int
            Free-surface reflection coefficient on pressure (+1 or -1). For an ideal
            free surface, R = -I ⇒ R_sign = -1.
        """
        G = _as_2d(g.astype(np.float64))
        self.g = G            # (T, NR)
        self.T, self.NR = G.shape
        self.R_sign = float(R_sign)

    def forward(self, P):
        """
        Apply 𝓔 to data P:  P0 = (I - G R) P = P - R_sign * (G P).
        Here G acts per receiver as causal convolution in time.
        Shapes
        ------
        P : array, shape (T, NR)
        Returns : array, shape (T, NR)
        """
        P = _as_2d(P.astype(np.float64))
        assert P.shape == (self.T, self.NR)
        out = np.empty_like(P)
        for j in range(self.NR):
            GjP = _conv1d_causal_same(self.g[:, j], P[:, j])  # G acting on column j
            out[:, j] = P[:, j] - self.R_sign * GjP
        return out

    def adjoint(self, Y):
        """
        Adjoint 𝓔* = I - R_sign * G*  (since R = R_sign * I and self-adjoint).
        Here G* is correlation with g per trace.
        """
        Y = _as_2d(Y.astype(np.float64))
        assert Y.shape == (self.T, self.NR)
        out = np.empty_like(Y)
        for j in range(self.NR):
            GtY = _corr1d_causal_same(self.g[:, j], Y[:, j])  # G* acting on column j
            out[:, j] = Y[:, j] - self.R_sign * GtY
        return out

    def as_linear_operator(self):
        """
        Expose as scipy.sparse.linalg.LinearOperator with matvec/rmatvec on vec-stacked data.
        """
        T, NR = self.T, self.NR
        shape = (T*NR, T*NR)
        def mv(x):
            X = x.reshape(T, NR)
            Y = self.forward(X)
            return Y.ravel()
        def rmv(y):
            Y = y.reshape(T, NR)
            X = self.adjoint(Y)
            return X.ravel()
        return LinearOperator(shape=shape, matvec=mv, rmatvec=rmv, dtype=np.float64)

    @staticmethod
    def build_explicit_matrix(g):
        """
        Build explicit sparse matrix E for small toy sizes.

        For per-trace 'same' causal convolution G = block_diag_j T(g[:,j]),
        𝓔 = I - R_sign * G, but we fix R_sign=-1 (free surface) → 𝓔 = I + G.

        Returns
        -------
        E : scipy.sparse.csc_matrix of shape (T*NR, T*NR)
        """
        G = _as_2d(np.asarray(g))
        T, NR = G.shape
        blocks = []
        for j in range(NR):
            Tgj = _build_toeplitz_causal(G[:, j])
            blocks.append(csc_matrix(Tgj))
        Gblk = block_diag(blocks, format='csc')
        I = csc_matrix(np.eye(T*NR))
        E = I + Gblk  # R = -I
        return E

    # --- sanity & tests -----------------------------------------------------
    def sanity_forward_on_impulse(self):
        """
        Send an impulse per trace and check the output equals (delta - R_sign*g) * delta
        which evaluates to delta - R_sign*g convolved with delta = delta - R_sign*g.
        """
        P = np.zeros((self.T, self.NR), dtype=np.float64)
        P[0, :] = 1.0  # delta at t=0, all traces
        Y = self.forward(P)
        # Expected: delta - R_sign * g (convolved with delta gives kernel itself)
        expect = P - self.R_sign * self.g
        return np.linalg.norm(Y - expect) / (np.linalg.norm(expect) + 1e-16)

    def dot_product_test(self, seed=7):
        """
        Check <𝓔 u, v> = <u, 𝓔* v>.
        """
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((self.T, self.NR))
        v = rng.standard_normal((self.T, self.NR))
        Eu = self.forward(u)
        Etv = self.adjoint(v)
        lhs = float(np.vdot(Eu.ravel(), v.ravel()))
        rhs = float(np.vdot(u.ravel(), Etv.ravel()))
        num = abs(lhs - rhs); den = abs(lhs) + abs(rhs) + 1e-16
        return num / den
