import math

import pytest
import roma
import torch
import triton
import triton.language as tl
from test_kernel_utils import load_mat3x3, load_vec4, store_mat3x3, store_vec4

from gsplat.triton_impl.quat_scale_to_covar import _quat_to_R, _quat_to_R_vjp

BLOCK_SIZE = 256
dtype = torch.float32
device = torch.device("cuda")


@triton.jit
def _quat_to_R_kernel(
    N,
    q_ptr,
    R_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N

    q0, q1, q2, q3 = load_vec4(offsets, q_ptr, masks)
    # fmt: off
    (
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
    ) = _quat_to_R(q0, q1, q2, q3)
    # fmt: on
    store_mat3x3(offsets, R_ptr, Rxx, Rxy, Rxz, Ryx, Ryy, Ryz, Rzx, Rzy, Rzz, masks)


def quat_to_R(q):
    N = q.shape[0]
    q = q.contiguous()
    R = torch.empty(N, 3, 3, dtype=dtype, device=device)
    n_blocks = int(math.ceil(N / BLOCK_SIZE))
    _quat_to_R_kernel[
        n_blocks,
    ](N, q, R, BLOCK_SIZE)
    return R


def quat_to_R_torch(q):
    # roma use xyzw convention while gsplat use wxyz
    q = q[..., [1, 2, 3, 0]]
    q = q / q.norm(dim=-1, keepdim=True)
    R = roma.unitquat_to_rotmat(q)
    return R


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_fwd():
    N = 1024
    q = torch.randn(N, 4, dtype=dtype, device=device)
    assert q.norm(dim=-1).min() > 1e-6

    R = quat_to_R(q)
    _R = quat_to_R_torch(q)
    torch.testing.assert_close(R, _R)


@triton.jit
def _quat_to_R_vjp_kernel(N, q_ptr, d_R_ptr, d_q_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N

    q0, q1, q2, q3 = load_vec4(offsets, q_ptr, masks)
    # fmt: off
    (
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryx, d_Ryy, d_Ryz,
        d_Rzx, d_Rzy, d_Rzz,
    ) = load_mat3x3(offsets, d_R_ptr, masks)

    d_q0, d_q1, d_q2, d_q3 = _quat_to_R_vjp(
        q0, q1, q2, q3,
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryx, d_Ryy, d_Ryz,
        d_Rzx, d_Rzy, d_Rzz,
    )
    # fmt: on
    store_vec4(offsets, d_q_ptr, d_q0, d_q1, d_q2, d_q3, masks)


def quat_to_R_vjp(q, d_R):
    N = q.shape[0]
    q = q.contiguous()
    d_R = d_R.contiguous()
    d_q = torch.empty(N, 4, dtype=dtype, device=device)
    n_blocks = int(math.ceil(N / BLOCK_SIZE))
    _quat_to_R_vjp_kernel[
        n_blocks,
    ](N, q, d_R, d_q, BLOCK_SIZE)
    return d_q


class QuatToR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q):
        ctx.save_for_backward(q)
        return quat_to_R(q)

    @staticmethod
    def backward(ctx, d_R):
        (q,) = ctx.saved_tensors
        return quat_to_R_vjp(q, d_R)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("N", [256, 1023, 1024, 1025, 4096])
def test_autograd(N):
    q = torch.randn(N, 4, dtype=dtype, device=device, requires_grad=True)
    assert q.norm(dim=-1).min() > 1e-3

    v_R = torch.randn(N, 3, 3, dtype=dtype, device=device)

    R = QuatToR.apply(q)
    # R = quat_to_R_torch(q)
    v_q = torch.autograd.grad((R * v_R).sum(), q)

    _R = quat_to_R_torch(q)
    _v_q = torch.autograd.grad((_R * v_R).sum(), q)
    torch.testing.assert_close(R, _R)
    torch.testing.assert_close(v_q, _v_q)
