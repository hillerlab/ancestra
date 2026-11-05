# Vendored from causal_conv1d==1.7.0; upstream source SHA-256: 80e75c8f761c88ede2ff73d244b0e9c0d2a974a780e2d3d8befeec1bd8aac29e; project added a pure-torch fallback; imports: causal_conv1d.* -> ancestra._vendor.causal_conv1d.*; CUDA extension import removed.
# Copyright (c) 2024, Tri Dao.

import torch

causal_conv1d_cuda = None

import torch.nn.functional as _F


def _ref_causal_conv1d(x, weight, bias, silu_activation):
    # Stateless, seq-less path of causal_conv1d_ref (pure torch, autograd-safe).
    dtype_in = x.dtype
    xw = x.to(weight.dtype)
    seqlen = x.shape[-1]
    width = weight.shape[1]
    dim = weight.shape[0]
    out = _F.conv1d(xw, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    out = out[..., :seqlen]
    if silu_activation:
        out = _F.silu(out)
    return out.to(dtype_in)


LIBRARY_NAME = "DaoAILab"


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_fwd_cpp", mutates_args={"out", "final_states_out"})
def _causal_conv1d_fwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    out: torch.Tensor,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> None:
    causal_conv1d_cuda.causal_conv1d_fwd(
        x,
        weight,
        bias,
        seq_idx,
        initial_states,
        out,
        final_states_out,
        silu_activation,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_bwd_cpp", mutates_args={
    "dfinal_states",
    "dx",
    "dweight",
    "dbias",
    "dinitial_states",
})
def _causal_conv1d_bwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    dx: torch.Tensor,
    dweight: torch.Tensor,
    dbias: torch.Tensor | None,
    dinitial_states: torch.Tensor,
    silu_activation: bool,
) -> None:
    causal_conv1d_cuda.causal_conv1d_bwd(
        x,
        weight,
        bias,
        dout,
        seq_idx,
        initial_states,
        dfinal_states,
        dx,
        dweight,
        dbias,
        dinitial_states,
        silu_activation,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_update_cpp", mutates_args={"out", "conv_state"})
def _causal_conv1d_update_cpp(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    silu_activation: bool,
    cache_seqlens: torch.Tensor | None,
    conv_state_indices: torch.Tensor | None,
) -> None:
    causal_conv1d_cuda.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        out,
        silu_activation,
        cache_seqlens,
        conv_state_indices
    )


def causal_conv1d_fwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> torch.Tensor:
    if causal_conv1d_cuda is None:
        assert seq_idx is None and initial_states is None and final_states_out is None
        return _ref_causal_conv1d(x, weight, bias, silu_activation)
    out = torch.empty_like(x)
    _causal_conv1d_fwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        seq_idx=seq_idx,
        initial_states=initial_states,
        out=out,
        final_states_out=final_states_out,
        silu_activation=silu_activation,
    )
    return out


def causal_conv1d_bwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    dx: torch.Tensor | None,
    return_dinitial_states: torch.Tensor,
    silu_activation: bool,
) -> tuple[torch.Tensor | None]:
    if causal_conv1d_cuda is None:
        assert seq_idx is None and initial_states is None and dfinal_states is None
        assert not return_dinitial_states
        with torch.enable_grad():
            xx = x.detach().to(weight.dtype).requires_grad_(True)
            ww = weight.detach().requires_grad_(True)
            bb = bias.detach().requires_grad_(True) if bias is not None else None
            outs = _ref_causal_conv1d(xx, ww, bb, silu_activation)
            ins = [xx, ww] + ([bb] if bb is not None else [])
            gs = torch.autograd.grad(outs, ins, dout.to(weight.dtype))
        if dx is None:
            dx = torch.empty_like(x)
        dx.copy_(gs[0].to(x.dtype))
        dweight = gs[1].to(weight.dtype)
        dbias = gs[2].to(bias.dtype) if bb is not None else None
        return dx, dweight, dbias, None
    batch_size, dim = x.size()[:2]
    width = weight.size(-1)

    if dx is None:
        dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    dbias = None
    if bias is not None:
        dbias = torch.zeros_like(bias, dtype=torch.float32)
    dinitial_states = None
    if return_dinitial_states:
        dinitial_states = torch.empty(batch_size, width - 1, dim, device=x.device, dtype=x.dtype).transpose(1, 2)

    _causal_conv1d_bwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        dout=dout,
        seq_idx=seq_idx,
        initial_states=initial_states,
        dfinal_states=dfinal_states,
        dx=dx,
        dweight=dweight,
        dbias=dbias,
        dinitial_states=dinitial_states,
        silu_activation=silu_activation,
    )

    dweight = dweight.type_as(weight)
    if dbias is not None:
        dbias = dbias.type_as(bias)
    return dx, dweight, dbias, dinitial_states


def causal_conv1d_update_function(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    silu_activation: bool,
    cache_seqlens: torch.Tensor | None,
    conv_state_indices: torch.Tensor | None,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _causal_conv1d_update_cpp(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        out=out,
        silu_activation=silu_activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
    )
    return out
