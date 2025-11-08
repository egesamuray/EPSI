# epsi_torch.py
# -----------------------------------------------------------------------------
# PyTorch layer for 𝓔[P] = (I - G R)P with R = -I → 𝓔 = I + G (per-trace 1D conv).
# -----------------------------------------------------------------------------
import torch
from torch import nn

class EpsiLayer(nn.Module):
    """
    Differentiable predicted-primaries operator. Assumes per-trace causal time filters g.
    Shapes:
      input:  (T, NR) or (B, T, NR) [B optional batch of gathers]
      g:      (T, NR) parameter/buffer (not updated by this layer unless registered as nn.Parameter)
    Implementation uses conv1d with groups=NR by reshaping to (B*1, NR, T).
    """
    def __init__(self, g, R_sign=-1):
        super().__init__()
        g = torch.as_tensor(g, dtype=torch.double)  # use double for gradcheck stability
        if g.ndim == 1: g = g[:, None]
        T, NR = g.shape
        self.T, self.NR = T, NR
        self.R_sign = float(R_sign)
        # conv1d in PyTorch is cross-correlation; to realize causal convolution with kernel g,
        # we flip in time when building weights and use padding='zeros'.
        w = torch.zeros((NR, 1, T), dtype=torch.double)
        for j in range(NR):
            w[j, 0, :] = torch.flip(g[:, j], dims=(0,))
        self.weight = nn.Parameter(w, requires_grad=False)  # fixed operator by default
        self.padding = T - 1  # full causal padding, then crop to 'same'

    def forward(self, P):
        """
        P : (T, NR) or (B, T, NR). Returns same shape.
        """
        if P.ndim == 2:
            P = P.unsqueeze(0)
        B, T, NR = P.shape
        assert T == self.T and NR == self.NR
        # reshape to (B, NR, T)
        X = P.permute(0, 2, 1).to(dtype=torch.double)
        # full causal pad on the left
        Xpad = nn.functional.pad(X, (self.padding, 0))
        Ycorr = nn.functional.conv1d(Xpad, self.weight, bias=None, stride=1, padding=0, groups=self.NR)
        # crop to same length (keep last T samples)
        Ycorr = Ycorr[:, :, :T]
        # 𝓔 = I - R_sign * G  (G realized by correlation ≡ adjoint; but since kernel flipped, it equals conv)
        Y = X - self.R_sign * Ycorr
        out = Y.permute(0, 2, 1)
        return out if out.shape[0] > 1 else out[0]

def gradient_check():
    torch.manual_seed(0)
    T, NR = 16, 3
    g = torch.randn(T, NR, dtype=torch.double) * 0.1
    layer = EpsiLayer(g)
    P = torch.randn(T, NR, dtype=torch.double, requires_grad=True)
    def func(x):
        return layer(x).pow(2).sum()  # simple scalar loss
    ok = torch.autograd.gradcheck(func, (P,), eps=1e-6, atol=1e-5, rtol=1e-5)
    return bool(ok)
