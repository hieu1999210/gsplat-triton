import math
import os
from typing import Tuple

import pytest
import torch
import triton
import triton.language as tl
from jaxtyping import Float
from test_inversion import inverse_sym_mat2, inverse_sym_mat2_vjp
from test_utils import rand_sym_mat
from torch import Tensor

from gsplat.triton_impl.util_kernels import _add_blur, _add_blur_vjp

BLOCK_SIZE = 256
dtype = torch.float32
device = torch.device("cuda")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@triton.jit
def _add_blur_kernel(
    N, M_ptr, M_blur_ptr, det_ptr, comp_ptr, eps, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N
    Mxx = tl.load(M_ptr + offsets * 4, mask=masks, other=1)
    Mxy = tl.load(M_ptr + offsets * 4 + 1, mask=masks, other=0)
    Myy = tl.load(M_ptr + offsets * 4 + 3, mask=masks, other=1)

    new_det, compensation, Mxx, Myy = _add_blur(Mxx, Mxy, Myy, eps)

    tl.store(M_blur_ptr + offsets * 4, Mxx, mask=masks)
    tl.store(M_blur_ptr + offsets * 4 + 1, Mxy, mask=masks)
    tl.store(M_blur_ptr + offsets * 4 + 2, Mxy, mask=masks)
    tl.store(M_blur_ptr + offsets * 4 + 3, Myy, mask=masks)
    tl.store(det_ptr + offsets, new_det, mask=masks)
    tl.store(comp_ptr + offsets, compensation, mask=masks)


def add_blur(M: Tensor, eps: Float) -> Tuple[Tensor, Tensor, Tensor]:
    N = M.numel() // M.shape[-1] // M.shape[-2]
    M = M.contiguous()
    M_blur = torch.empty_like(M)
    det = torch.empty(N, device=device, dtype=dtype)
    comp = torch.empty(N, device=device, dtype=dtype)
    n_block = int(math.ceil(N / BLOCK_SIZE))
    _add_blur_kernel[
        n_block,
    ](N, M, M_blur, det, comp, eps, BLOCK_SIZE)
    comp = comp.reshape(*(M.shape[:-2]))
    det = det.reshape(*(M.shape[:-2]))
    return M_blur, det, comp


def add_blur_torch(M, eps):
    M_blur = M.clone()
    M_blur[..., 0, 0] += eps
    M_blur[..., 1, 1] += eps

    old_det = torch.linalg.det(M)
    new_det = torch.linalg.det(M_blur)
    compensation = torch.sqrt(torch.clamp_min_(old_det / new_det, 0.0))
    return M_blur, new_det, compensation


def test_fwd():
    N = 1024
    eps = 1e-3
    M = rand_sym_mat(N, 2, device, dtype, eps)

    M_blur, new_det, compensation = add_blur(M, eps)
    _M_blur, _new_det, _compensation = add_blur_torch(M, eps)
    torch.testing.assert_close(M_blur, _M_blur)
    torch.testing.assert_close(new_det, _new_det)
    torch.testing.assert_close(compensation, _compensation, rtol=1e-4, atol=1e-4)


@triton.jit
def _add_blur_vjp_kernel(
    N, iM_blur_ptr, comp_ptr, v_comp_ptr, v_M_ptr, eps, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N
    iMxx = tl.load(iM_blur_ptr + offsets * 4, mask=masks, other=1)
    iMxy = tl.load(iM_blur_ptr + offsets * 4 + 1, mask=masks, other=0)
    iMyy = tl.load(iM_blur_ptr + offsets * 4 + 3, mask=masks, other=1)
    comp = tl.load(comp_ptr + offsets, mask=masks, other=1)
    v_comp = tl.load(v_comp_ptr + offsets, mask=masks, other=0)

    v_Mxx, v_Mxy, v_Myy = _add_blur_vjp(iMxx, iMxy, iMyy, comp, v_comp, eps)

    tl.store(v_M_ptr + offsets * 4, v_Mxx, mask=masks)
    tl.store(v_M_ptr + offsets * 4 + 1, v_Mxy, mask=masks)
    tl.store(v_M_ptr + offsets * 4 + 2, v_Mxy, mask=masks)
    tl.store(v_M_ptr + offsets * 4 + 3, v_Myy, mask=masks)


def add_blur_vjp(iM_blur, comp, v_comp, eps):
    N = iM_blur.numel() // iM_blur.shape[-1] // iM_blur.shape[-2]
    iM_blur = iM_blur.contiguous()
    comp = comp.contiguous()
    v_comp = v_comp.contiguous()
    v_M = torch.empty_like(iM_blur)
    n_block = int(math.ceil(N / BLOCK_SIZE))
    _add_blur_vjp_kernel[
        n_block,
    ](N, iM_blur, comp, v_comp, v_M, eps, BLOCK_SIZE)
    return v_M


def add_blur_inv_torch(M, eps):
    M_blur, new_det, compensation = add_blur_torch(M, eps)
    iM_blur = torch.inverse(M_blur)
    return iM_blur, compensation


class _AddBlurInv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, eps):
        M_blur, _, comp = add_blur(M, eps)
        iM_blur = inverse_sym_mat2(M_blur)
        ctx.save_for_backward(iM_blur, comp)
        ctx.eps = eps
        return iM_blur, comp

    @staticmethod
    def backward(ctx, grad_iM_blur, grad_comp):
        iM_blur, comp = ctx.saved_tensors
        eps = ctx.eps
        v_M = add_blur_vjp(iM_blur, comp, grad_comp, eps)
        v_M += inverse_sym_mat2_vjp(iM_blur, grad_iM_blur)
        return v_M, None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("N", [256, 1023, 1024, 1025, 4096])
def test_autograd(N):
    eps = 0.3
    M = rand_sym_mat(N, 2, device, dtype, 1e-3)
    M.requires_grad = True

    v_iM_blur = rand_sym_mat(N, 2, device, dtype, 1e-3)
    v_comp = torch.randn(N, device=device, dtype=dtype)

    iM_blur, comp = add_blur_inv_torch(M, eps)
    (v_M,) = torch.autograd.grad(
        (v_iM_blur * iM_blur).sum() + (comp * v_comp).sum(), (M,)
    )

    _iM_blur, _comp = _AddBlurInv.apply(M, eps)
    (_v_M,) = torch.autograd.grad(
        (v_iM_blur * _iM_blur).sum() + (_comp * v_comp).sum(), (M,)
    )

    torch.testing.assert_close(iM_blur, _iM_blur, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(comp, _comp, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_M, _v_M, rtol=2e-4, atol=2e-4)
