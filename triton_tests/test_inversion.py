import math

import pytest
import torch
import triton
import triton.language as tl
from einops import repeat
from test_kernel_utils import load_sym_mat2, store_sym_mat2
from test_utils import rand_sym_mat
from torch import Tensor

from gsplat.triton_impl.util_kernels import _inverse_sym_mat2, _inverse_sym_mat2_vjp

BLOCK_SIZE = 256
dtype = torch.float32
device = torch.device("cuda")


@triton.jit
def _inverse_sym_mat2_kernel(
    N,
    M_ptr,  # Float [N 2 2]
    iM_ptr,  # Float [N 2 2]
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N
    Mxx, Mxy, Myy = load_sym_mat2(offsets, M_ptr, masks)
    iMxx, iMxy, iMyy = _inverse_sym_mat2(Mxx, Mxy, Myy)
    store_sym_mat2(offsets, iM_ptr, iMxx, iMxy, iMyy, masks)


def inverse_sym_mat2(M: Tensor) -> Tensor:
    """
    M^T=M
    iM @ M = I
    return iM
    """
    M = M.contiguous()
    N = M.numel() // M.shape[-1] // M.shape[-2]
    iM = torch.empty_like(M)
    n_block = int(math.ceil(N / BLOCK_SIZE))
    _inverse_sym_mat2_kernel[
        n_block,
    ](N, M, iM, BLOCK_SIZE)
    return iM


@triton.jit
def _inverse_sym_mat2_vjp_kernel(
    N,
    iM_ptr,  # Float [N 2 2]
    v_iM_ptr,  # Float [N 2 2]
    v_M_ptr,  # Float [N 2 2]
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N
    iMxx, iMxy, iMyy = load_sym_mat2(offsets, iM_ptr, masks)
    v_iMxx, v_iMxy, v_iMyy = load_sym_mat2(offsets, v_iM_ptr, masks)
    v_Mxx, v_Mxy, v_Myy = _inverse_sym_mat2_vjp(
        v_iMxx, v_iMxy, v_iMyy, iMxx, iMxy, iMyy
    )
    store_sym_mat2(offsets, v_M_ptr, v_Mxx, v_Mxy, v_Myy, masks)


def inverse_sym_mat2_vjp(iM: Tensor, v_iM: Tensor) -> Tensor:
    """
    M^T=M
    iM @ M = I
    return iM
    """
    iM = iM.contiguous()
    v_iM = v_iM.contiguous()
    N = iM.numel() // iM.shape[-1] // iM.shape[-2]
    v_M = torch.empty_like(iM)
    n_block = int(math.ceil(N / BLOCK_SIZE))
    _inverse_sym_mat2_vjp_kernel[
        n_block,
    ](N, iM, v_iM, v_M, BLOCK_SIZE)
    return v_M


class _InverseSymMat2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        iM = inverse_sym_mat2(M)
        ctx.save_for_backward(iM)
        return iM

    @staticmethod
    def backward(ctx, grad_iM):
        (iM,) = ctx.saved_tensors
        v_M = inverse_sym_mat2_vjp(iM, grad_iM)
        return v_M


def test_fwd():
    N = 1024
    eps = 1e-3
    M = rand_sym_mat(N, 2, device, dtype, eps)

    iM = inverse_sym_mat2(M)
    I = repeat(torch.eye(2, device=device, dtype=dtype), "i j -> n i j", n=N)
    torch.testing.assert_close(iM @ M, I, atol=1e-4, rtol=1e-4)
    _iM = torch.inverse(M)
    torch.testing.assert_close(iM, _iM, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("N", [256, 1023, 1024, 1025, 4096])
def test_autograd(N):
    eps = 1e-3
    M = rand_sym_mat(N, 2, device, dtype, eps)
    M.requires_grad = True

    v_iM = rand_sym_mat(N, 2, device, dtype, eps)

    iM = _InverseSymMat2.apply(M)
    (v_M,) = torch.autograd.grad((v_iM * iM).sum(), (M,))

    _iM = torch.inverse(M)
    (_v_M,) = torch.autograd.grad((v_iM * _iM).sum(), (M,))
    torch.testing.assert_close(v_M, _v_M, atol=5e-4, rtol=5e-4)
