"""The network.

1-nt dilated conv stem -> stride-4 conv path (4 nt) -> stride-4 to 16-nt tokens of width d ->
10 BiMamba2 layers -> pool x8 -> 4 global RoPE attention blocks at 128 nt -> unpool + residual ->
10 BiMamba2 layers -> transpose convs 16 -> 4 -> 1 with additive skips -> heads.

Input  [B, 6, W]: one-hot ACGTN + soft-mask bit.  W must be a multiple of 128.
Output logits: joint [B, 2, 23, W], sites [B, 2, 6, W], repeat [B, 1, W], mlm [B, 5, W].
Strand plane 0 = + strand, 1 = - strand, both in genome coordinates.

BiMamba2 is the Caduceus form: a forward and a reversed Mamba2 SSD scan whose in_proj and
out_proj are tied; conv1d, dt_bias, A_log, D and the gated norm are per direction.  Because in_proj is
pointwise, projecting once and flipping the projection equals projecting the flipped input, and
out_proj is linear, so out_proj(y_f) + out_proj(y_r) = out_proj(y_f + y_r): one in_proj and one
out_proj per layer.

The SSD scan uses the vendored fused Triton kernel on CUDA (`ssd_backend="kernel"`) and a pure-torch
chunked reference on CPU (`"ref"`). `"auto"` selects Triton on CUDA and fails if that kernel is
unavailable; both paths compute the same function.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Optional on CPU-only hosts; CUDA dispatch below fails rather than silently using the reference.
    from ancestra._vendor.mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
except Exception:  # pragma: no cover - depends on the environment
    mamba_split_conv1d_scan_combined = None

N_STATES = 23  # joint gene-structure states (see ancestra.model.adapter)
N_SITES = 6    # donor, acceptor, START, STOP, TSS, TES
N_MLM = 5      # ACGTN
IN_CH = 6      # ACGTN one-hot + soft-mask bit


@dataclass
class Config:
    d: int = 1024
    # 1-nt stem
    stem_width: int = 128
    stem_kernel: int = 9
    stem_dilations: tuple = (1, 2, 4, 8, 16, 32, 1, 2)
    # 4-nt path
    down_width: int = 256
    down_blocks: int = 4
    down_kernel: int = 9
    down_dilations: tuple = (1, 2, 4, 8)
    # trunk
    n_mamba_pre: int = 10
    n_mamba_post: int = 10
    n_attn: int = 4
    pool: int = 8
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    d_conv: int = 4
    ngroups: int = 1
    chunk_size: int = 256
    ffn_mult: float = 2.75
    attn_head_dim: int = 64
    rope_base: float = 10000.0
    # heads
    n_states: int = N_STATES
    n_sites: int = N_SITES
    head_kernel: int = 5
    # numerics
    ssd_backend: str = "auto"  # auto | kernel | ref

    def __post_init__(self):
        if self.ssd_backend not in ("auto", "kernel", "ref"):
            raise ValueError(f"ssd_backend must be auto|kernel|ref, got {self.ssd_backend}")
        if (self.d * self.expand) % self.headdim:
            raise ValueError("d * expand must be a multiple of headdim")
        if self.d % self.attn_head_dim:
            raise ValueError("d must be a multiple of attn_head_dim")

    @property
    def token_nt(self) -> int:
        return 16

    @property
    def window_multiple(self) -> int:
        return self.token_nt * self.pool


class RMSNorm(nn.Module):
    """RMSNorm over the last dim (dim=-1) or the channel dim of [B, C, L] (dim=1)."""

    def __init__(self, n, dim=-1, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(self.dim, keepdim=True) + self.eps)
        w = self.weight if self.dim == -1 else self.weight[:, None]
        return (y * w).to(x.dtype)


class ConvBlock(nn.Module):
    """Pre-norm residual dilated conv block on [B, C, L]: x + conv1x1(gelu(conv_k,dil(norm(x))))."""

    def __init__(self, w, k, dilation):
        super().__init__()
        self.norm = RMSNorm(w, dim=1)
        self.conv = nn.Conv1d(w, w, k, padding=dilation * (k - 1) // 2, dilation=dilation)
        self.proj = nn.Conv1d(w, w, 1)

    def forward(self, x):
        return x + self.proj(F.gelu(self.conv(self.norm(x))))


class SwiGLU(nn.Module):
    def __init__(self, d, mult):
        super().__init__()
        h = int(round(d * mult))
        self.norm = RMSNorm(d)
        self.gate_up = nn.Linear(d, 2 * h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        g, u = self.gate_up(self.norm(x)).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


def _segsum(x):
    """Stable segment sum: out[..., i, j] = sum_{j<k<=i} x[..., k] for j <= i, -inf above."""
    t = x.size(-1)
    x = x[..., None].expand(*x.shape, t)
    low = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~low, 0)
    s = torch.cumsum(x, dim=-2)
    keep = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=0)
    return s.masked_fill(~keep, -torch.inf)


def ssd_reference(x, dt, a, b, c, chunk):
    """Chunked SSD (Mamba2 'minimal discrete' form), fp32.

    x [B, L, H, P], dt [B, L, H] (post-softplus), a [H] (negative), b/c [B, L, G, N]. Returns y [B, L, H, P].
    """
    bsz, L, H, P = x.shape
    G = b.shape[2]
    pad = (-L) % chunk
    if pad:
        x, dt = F.pad(x, (0, 0, 0, 0, 0, pad)), F.pad(dt, (0, 0, 0, pad))
        b, c = F.pad(b, (0, 0, 0, 0, 0, pad)), F.pad(c, (0, 0, 0, 0, 0, pad))
    n = x.shape[1] // chunk
    rep = H // G
    b = b.repeat_interleave(rep, dim=2)
    c = c.repeat_interleave(rep, dim=2)
    xd = (x * dt[..., None]).view(bsz, n, chunk, H, P)
    A = (dt * a).view(bsz, n, chunk, H).permute(0, 3, 1, 2)  # [B, H, n, l]
    b = b.view(bsz, n, chunk, H, -1)
    c = c.view(bsz, n, chunk, H, -1)
    a_cum = torch.cumsum(A, dim=-1)
    Lm = torch.exp(_segsum(A))
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", c, b, Lm, xd)
    decay_states = torch.exp(a_cum[..., -1:] - a_cum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", b, decay_states, xd)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(a_cum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]
    y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", c, states, torch.exp(a_cum))
    y = (y_diag + y_off).reshape(bsz, n * chunk, H, P)
    return y[:, :L]


class _Direction(nn.Module):
    """Per-direction Mamba2 parameters (everything except the tied projections)."""

    def __init__(self, cfg: Config):
        super().__init__()
        d_inner = cfg.d * cfg.expand
        nheads = d_inner // cfg.headdim
        conv_dim = d_inner + 2 * cfg.ngroups * cfg.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, cfg.d_conv, groups=conv_dim, padding=cfg.d_conv - 1)
        # Mamba2 defaults: dt in [1e-3, 1e-1] log-uniform, A in [1, 16], D = 1
        dt = torch.exp(torch.rand(nheads) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(nheads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(nheads))
        self.norm_weight = nn.Parameter(torch.ones(d_inner))


class BiMamba2(nn.Module):
    """Forward + reversed Mamba2 SSD scans with tied in/out projections."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.d_inner = cfg.d * cfg.expand
        self.nheads = self.d_inner // cfg.headdim
        self.in_proj = nn.Linear(cfg.d, 2 * self.d_inner + 2 * cfg.ngroups * cfg.d_state + self.nheads, bias=False)
        self.out_proj = nn.Linear(self.d_inner, cfg.d, bias=False)
        self.fwd = _Direction(cfg)
        self.rev = _Direction(cfg)

    def _use_kernel(self, x):
        be = self.cfg.ssd_backend
        if be == "ref":
            return False
        if be == "kernel" and not x.is_cuda:
            raise RuntimeError("ssd_backend='kernel' needs CUDA input")
        if not x.is_cuda:
            return False
        if mamba_split_conv1d_scan_combined is None:
            raise RuntimeError("vendored Triton SSD kernel is unavailable")
        return True

    def _scan_kernel(self, zxbcdt, p: _Direction):
        c = self.cfg
        return mamba_split_conv1d_scan_combined(
            zxbcdt, p.conv1d.weight.squeeze(1), p.conv1d.bias, p.dt_bias, -torch.exp(p.A_log.float()), p.D,
            chunk_size=c.chunk_size, activation="silu", rmsnorm_weight=p.norm_weight, rmsnorm_eps=1e-5,
            outproj_weight=None, outproj_bias=None, headdim=c.headdim, ngroups=c.ngroups, norm_before_gate=False)

    def _scan_ref(self, zxbcdt, p: _Direction):
        c = self.cfg
        dtype = zxbcdt.dtype
        zxbcdt = zxbcdt.float()
        bsz, L, _ = zxbcdt.shape
        gn = c.ngroups * c.d_state
        z, xbc, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * gn, self.nheads], dim=-1)
        xbc = F.conv1d(xbc.transpose(1, 2), p.conv1d.weight.float(), p.conv1d.bias.float(),
                       padding=c.d_conv - 1, groups=xbc.shape[-1])[..., :L]
        xbc = F.silu(xbc).transpose(1, 2)
        x, b, cc = torch.split(xbc, [self.d_inner, gn, gn], dim=-1)
        dt = F.softplus(dt + p.dt_bias.float())
        a = -torch.exp(p.A_log.float())
        xh = x.view(bsz, L, self.nheads, c.headdim)
        y = ssd_reference(xh, dt, a, b.view(bsz, L, c.ngroups, c.d_state), cc.view(bsz, L, c.ngroups, c.d_state),
                          c.chunk_size)
        y = (y + xh * p.D.float()[:, None]).reshape(bsz, L, self.d_inner)
        # gated RMSNorm, norm_before_gate=False, group size d_inner / ngroups
        y = y * F.silu(z)
        yg = y.view(bsz, L, c.ngroups, -1)
        yg = yg * torch.rsqrt(yg.pow(2).mean(-1, keepdim=True) + 1e-5)
        y = yg.view(bsz, L, self.d_inner) * p.norm_weight.float()
        return y.to(dtype)

    def forward(self, x):  # x [B, T, d]
        zxbcdt = self.in_proj(x)
        scan = self._scan_kernel if self._use_kernel(zxbcdt) else self._scan_ref
        y = scan(zxbcdt, self.fwd) + scan(zxbcdt.flip(1), self.rev).flip(1)
        return self.out_proj(y)


class MambaLayer(nn.Module):
    """Pre-norm residual BiMamba2, then pre-norm residual SwiGLU."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm = RMSNorm(cfg.d)
        self.mixer = BiMamba2(cfg)
        self.ffn = SwiGLU(cfg.d, cfg.ffn_mult)

    def forward(self, x):
        x = x + self.mixer(self.norm(x))
        return x + self.ffn(x)


def _rope(t, dh, base, device):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh))
    f = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv)
    return f.cos(), f.sin()


def _apply_rope(x, cos, sin):  # x [B, H, T, dh]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class AttnBlock(nn.Module):
    """Global (non-causal) RoPE attention via SDPA (FlashAttention backend in bf16 on A100), then SwiGLU."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.heads = cfg.d // cfg.attn_head_dim
        self.norm = RMSNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        self.ffn = SwiGLU(cfg.d, cfg.ffn_mult)

    def forward(self, x):
        b, t, d = x.shape
        dh = self.cfg.attn_head_dim
        q, k, v = self.qkv(self.norm(x)).view(b, t, 3, self.heads, dh).permute(2, 0, 3, 1, 4)
        cos, sin = _rope(t, dh, self.cfg.rope_base, x.device)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        return x + self.ffn(x)


class Network(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = c = cfg
        sw, dw, d = c.stem_width, c.down_width, c.d
        self.embed = nn.Conv1d(IN_CH, sw, 1)
        self.stem = nn.ModuleList(ConvBlock(sw, c.stem_kernel, dl) for dl in c.stem_dilations)
        self.down1 = nn.Conv1d(sw, dw, 4, stride=4)
        self.down_blocks = nn.ModuleList(
            ConvBlock(dw, c.down_kernel, c.down_dilations[i % len(c.down_dilations)]) for i in range(c.down_blocks))
        self.down2 = nn.Conv1d(dw, d, 4, stride=4)
        self.pre = nn.ModuleList(MambaLayer(c) for _ in range(c.n_mamba_pre))
        self.pool_norm = RMSNorm(d)
        self.attn = nn.ModuleList(AttnBlock(c) for _ in range(c.n_attn))
        self.unpool_norm = RMSNorm(d)
        self.unpool_proj = nn.Linear(d, d, bias=False)
        self.post = nn.ModuleList(MambaLayer(c) for _ in range(c.n_mamba_post))
        self.out_norm = RMSNorm(d)
        self.up1 = nn.ConvTranspose1d(d, dw, 4, stride=4)
        self.up2 = nn.ConvTranspose1d(dw, sw, 4, stride=4)
        self.head_norm = RMSNorm(sw, dim=1)
        self.n_out = 2 * c.n_states + 2 * c.n_sites + 1 + N_MLM
        self.head = nn.Conv1d(sw, self.n_out, c.head_kernel, padding=c.head_kernel // 2)

    def forward(self, x):
        c = self.cfg
        bsz, ch, w = x.shape
        if ch != IN_CH:
            raise ValueError(f"expected [B, {IN_CH}, W], got {tuple(x.shape)}")
        if w % c.window_multiple:
            raise ValueError(f"W must be a multiple of {c.window_multiple}, got {w}")
        h = self.embed(x)
        for blk in self.stem:
            h = blk(h)
        s1 = h                                            # [B, sw, W]
        h = self.down1(h)
        for blk in self.down_blocks:
            h = blk(h)
        s4 = h                                            # [B, dw, W/4]
        h = self.down2(h).transpose(1, 2)                 # [B, T, d], T = W/16
        for blk in self.pre:
            h = blk(h)
        t = h.shape[1]
        g = self.pool_norm(h.view(bsz, t // c.pool, c.pool, -1).mean(2))  # [B, T/8, d]
        for blk in self.attn:
            g = blk(g)
        h = h + self.unpool_proj(self.unpool_norm(g)).repeat_interleave(c.pool, dim=1)
        for blk in self.post:
            h = blk(h)
        h = self.out_norm(h).transpose(1, 2)              # [B, d, T]
        h = F.gelu(self.up1(h) + s4)
        h = F.gelu(self.up2(h) + s1)
        y = self.head(self.head_norm(h))                  # [B, n_out, W]
        ns, nz = c.n_states, c.n_sites
        joint = y[:, : 2 * ns].view(bsz, 2, ns, w)
        sites = y[:, 2 * ns: 2 * ns + 2 * nz].view(bsz, 2, nz, w)
        rep = y[:, 2 * ns + 2 * nz: 2 * ns + 2 * nz + 1]
        mlm = y[:, 2 * ns + 2 * nz + 1:]
        return {"joint": joint, "sites": sites, "repeat": rep, "mlm": mlm}


def build_model(cfg: dict) -> Network:
    return Network(Config(**cfg))
