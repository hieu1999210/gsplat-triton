import math

import pytest
import roma
import torch
import triton
import triton.language as tl
from test_kernel_utils import (
    load_sym_mat3,
    load_vec3,
    load_vec4,
    store_sym_mat3,
    store_vec3,
    store_vec4,
)
from test_utils import rand_sym_mat

from gsplat.triton_impl.quat_scale_to_covar import (
    _quat_scale_to_covar,
    _quat_scale_to_covar_vjp,
)

BLOCK_SIZE = 256
dtype = torch.float32
device = torch.device("cuda")


@triton.jit
def _quat_scale_to_covar_kernel(
    N,
    q_ptr,
    s_ptr,
    covar_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N

    q0, q1, q2, q3 = load_vec4(offsets, q_ptr, masks)
    s0, s1, s2 = load_vec3(offsets, s_ptr, masks, default=1.0)

    c3xx, c3xy, c3xz, c3yy, c3yz, c3zz = _quat_scale_to_covar(
        q0, q1, q2, q3, s0, s1, s2
    )

    store_sym_mat3(offsets, covar_ptr, c3xx, c3xy, c3xz, c3yy, c3yz, c3zz, masks)


def quat_scale_to_covar(q, s):
    N = q.shape[0]
    q = q.contiguous()
    s = s.contiguous()
    covar = torch.zeros(N, 3, 3, dtype=dtype, device=device)
    n_blocks = int(math.ceil(N / BLOCK_SIZE))
    _quat_scale_to_covar_kernel[
        n_blocks,
    ](N, q, s, covar, BLOCK_SIZE)
    return covar


def quat_scale_to_covar_torch(q, s):
    # roma use xyzw convention while gsplat use wxyz
    q = q[..., [1, 2, 3, 0]]
    q = q / q.norm(dim=-1, keepdim=True)
    R = roma.unitquat_to_rotmat(q)
    RS = R * s[:, None, :]
    covar = torch.einsum("...ij,...kj->...ik", RS, RS)
    return covar


def test_fwd():
    N = 1024
    q = torch.randn(N, 4, dtype=dtype, device=device)
    assert q.norm(dim=-1).min() > 1e-6
    s = torch.rand(N, 3, dtype=dtype, device=device)

    covar = quat_scale_to_covar(q, s)
    _covar = quat_scale_to_covar_torch(q, s)
    torch.testing.assert_close(covar, _covar)


@triton.jit
def _quat_scale_to_covar_vjp_kernel(
    N, q_ptr, s_ptr, d_covar_ptr, d_q_ptr, d_s_ptr, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N

    q0, q1, q2, q3 = load_vec4(offsets, q_ptr, masks)
    s0, s1, s2 = load_vec3(offsets, s_ptr, masks, default=1.0)

    d_c3xx, d_c3xy, d_c3xz, d_c3yy, d_c3yz, d_c3zz = load_sym_mat3(
        offsets, d_covar_ptr, masks
    )
    # fmt: off
    (
        d_q0, d_q1, d_q2, d_q3, 
        d_s0, d_s1, d_s2,
    ) = _quat_scale_to_covar_vjp(
        q0, q1, q2, q3, 
        s0, s1, s2, 
        d_c3xx, d_c3xy, d_c3xz, 
        d_c3yy, d_c3yz, d_c3zz
    )
    # fmt: on
    store_vec4(offsets, d_q_ptr, d_q0, d_q1, d_q2, d_q3, masks)
    store_vec3(offsets, d_s_ptr, d_s0, d_s1, d_s2, masks)


def quat_scale_to_covar_vjp(q, s, d_covar):
    N = q.shape[0]
    q = q.contiguous()
    s = s.contiguous()
    d_covar = d_covar.contiguous()
    d_q = torch.empty_like(q)
    d_s = torch.empty_like(s)
    n_blocks = int(math.ceil(N / BLOCK_SIZE))
    _quat_scale_to_covar_vjp_kernel[
        n_blocks,
    ](N, q, s, d_covar, d_q, d_s, BLOCK_SIZE)
    return d_q, d_s


class QuatScaleToCovar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, s):
        ctx.save_for_backward(q, s)
        covar = quat_scale_to_covar(q, s)
        return covar

    @staticmethod
    def backward(ctx, d_covar):
        q, s = ctx.saved_tensors
        d_q, d_s = quat_scale_to_covar_vjp(q, s, d_covar)
        return d_q, d_s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("N", [256, 1023, 1024, 1025, 4096])
def test_autograd(N):
    q = torch.randn(N, 4, dtype=dtype, device=device, requires_grad=True)
    assert q.norm(dim=-1).min() > 1e-6
    s = torch.rand(N, 3, dtype=dtype, device=device, requires_grad=True)

    q.requires_grad = True
    s.requires_grad = True

    v_covar = rand_sym_mat(N, 3, dtype=dtype, device=device)

    covar = QuatScaleToCovar.apply(q, s)
    (v_q, v_s) = torch.autograd.grad((covar * v_covar).sum(), (q, s))

    _covar = quat_scale_to_covar_torch(q, s)
    (_v_q, _v_s) = torch.autograd.grad((_covar * v_covar).sum(), (q, s))
    torch.testing.assert_close(covar, _covar)
    torch.testing.assert_close(v_s, _v_s)
    torch.testing.assert_close(v_q, _v_q)
