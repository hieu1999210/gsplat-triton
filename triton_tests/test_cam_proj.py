import math

import pytest
import torch
import triton
import triton.language as tl
from einops import rearrange
from test_kernel_utils import (
    load_sym_mat2,
    load_sym_mat3,
    load_vec2,
    load_vec3,
    store_sym_mat2,
    store_sym_mat3,
    store_vec2,
    store_vec3,
)
from test_utils import rand_sym_mat

from gsplat.triton_impl.cam_proj import _persp_proj, _persp_proj_vjp

BLOCK_SIZE = 256
dtype = torch.float32
device = torch.device("cuda")
image_width = 640
image_height = 480


@triton.jit
def load_K(offsets, Ks_ptr, masks):
    fx = tl.load(Ks_ptr + offsets * 9, mask=masks, other=1.0)
    fy = tl.load(Ks_ptr + offsets * 9 + 4, mask=masks, other=1.0)
    cx = tl.load(Ks_ptr + offsets * 9 + 2, mask=masks, other=0.0)
    cy = tl.load(Ks_ptr + offsets * 9 + 5, mask=masks, other=0.0)
    return fx, fy, cx, cy


@triton.jit
def _persp_proj_kernel(
    N,
    mean3_ptr,
    cov3_ptr,
    Ks_ptr,
    mean2_ptr,
    cov2_ptr,
    image_width,
    image_height,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N

    mean_x, mean_y, mean_z = load_vec3(offsets, mean3_ptr, masks)
    c3xx, c3xy, c3xz, c3yy, c3yz, c3zz = load_sym_mat3(offsets, cov3_ptr, masks)

    fx, fy, cx, cy = load_K(offsets, Ks_ptr, masks)

    mean2_x, mean2_y, cov2_xx, cov2_xy, cov2_yy = _persp_proj(
        # fmt: off
        mean_x, mean_y, mean_z,
        c3xx, c3xy, c3xz, c3yy, c3yz, c3zz,
        fx, fy, cx, cy,
        image_width, image_height,
        # fmt: on
    )
    store_vec2(offsets, mean2_ptr, mean2_x, mean2_y, masks)
    store_sym_mat2(offsets, cov2_ptr, cov2_xx, cov2_xy, cov2_yy, masks)


def persp_proj(mean3, cov3, Ks, image_width, image_height):
    N = mean3.shape[0]
    mean3 = mean3.contiguous()
    cov3 = cov3.contiguous()
    Ks = Ks.contiguous()
    mean2 = torch.empty(N, 2, dtype=dtype, device=device)
    cov2 = torch.empty(N, 2, 2, dtype=dtype, device=device)
    n_blocks = int(math.ceil(N / BLOCK_SIZE))
    _persp_proj_kernel[
        n_blocks,
    ](N, mean3, cov3, Ks, mean2, cov2, image_width, image_height, BLOCK_SIZE)
    return mean2, cov2


@triton.jit
def _persp_proj_vjp_kernel(
    N,
    mean3_ptr,
    cov3_ptr,
    Ks_ptr,
    v_mean2_ptr,
    v_cov2_ptr,
    v_mean3_ptr,
    v_cov3_ptr,
    image_width,
    image_height,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N

    m3_x, m3_y, m3_z = load_vec3(offsets, mean3_ptr, masks)
    c3xx, c3xy, c3xz, c3yy, c3yz, c3zz = load_sym_mat3(offsets, cov3_ptr, masks)
    v_m2_x, v_m2_y = load_vec2(offsets, v_mean2_ptr, masks)
    v_c2_xx, v_c2_xy, v_c2_yy = load_sym_mat2(offsets, v_cov2_ptr, masks)
    fx, fy, cx, cy = load_K(offsets, Ks_ptr, masks)
    # fmt: off
    (
        v_m3_x, v_m3_y, v_m3_z,
        v_c3_xx, v_c3_xy, v_c3_xz, 
        v_c3_yy, v_c3_yz, v_c3_zz,
    ) = _persp_proj_vjp(
        m3_x, m3_y, m3_z,
        c3xx, c3xy, c3xz, 
        c3yy, c3yz, c3zz,
        fx, fy, cx, cy,
        image_width, image_height,
        v_m2_x, v_m2_y,
        v_c2_xx, v_c2_xy, v_c2_yy,
    )
    # fmt: on
    store_vec3(offsets, v_mean3_ptr, v_m3_x, v_m3_y, v_m3_z, masks)
    store_sym_mat3(
        offsets, v_cov3_ptr, v_c3_xx, v_c3_xy, v_c3_xz, v_c3_yy, v_c3_yz, v_c3_zz, masks
    )


def persp_proj_vjp(mean3, cov3, Ks, v_mean2, v_cov2, image_width, image_height):
    N = mean3.shape[0]
    mean3 = mean3.contiguous()
    cov3 = cov3.contiguous()
    Ks = Ks.contiguous()
    v_mean2 = v_mean2.contiguous()
    v_cov2 = v_cov2.contiguous()
    v_mean3 = torch.empty(N, 3, dtype=dtype, device=device)
    v_cov3 = torch.empty(N, 3, 3, dtype=dtype, device=device)
    n_blocks = int(math.ceil(N / BLOCK_SIZE))
    _persp_proj_vjp_kernel[n_blocks,](
        N,
        mean3,
        cov3,
        Ks,
        v_mean2,
        v_cov2,
        v_mean3,
        v_cov3,
        image_width,
        image_height,
        BLOCK_SIZE,
    )
    return v_mean3, v_cov3


class PerspProj(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean3, cov3, Ks, image_width, image_height):
        mean2, cov2 = persp_proj(mean3, cov3, Ks, image_width, image_height)
        ctx.save_for_backward(mean3, cov3, Ks)
        ctx.image_width = image_width
        ctx.image_height = image_height
        return mean2, cov2

    @staticmethod
    def backward(ctx, v_mean2, v_cov2):
        mean3, cov3, Ks = ctx.saved_tensors
        image_width = ctx.image_width
        image_height = ctx.image_height
        v_mean3, v_cov3 = persp_proj_vjp(
            mean3, cov3, Ks, v_mean2, v_cov2, image_width, image_height
        )
        return v_mean3, v_cov3, None, None, None


def persp_proj_torch(mean3, cov3, Ks, image_width, image_height, margin=0.15):
    fx = Ks[..., 0, 0]
    fy = Ks[..., 1, 1]
    cx = Ks[..., 0, 2]
    cy = Ks[..., 1, 2]

    iz = 1.0 / mean3[..., 2]

    margin_x = margin * image_width / fx
    screen_x_min = -margin_x - cx / fx
    screen_x_max = margin_x + (image_width - cx) / fx
    screen_x = torch.clamp(mean3[..., 0] * iz, screen_x_min, screen_x_max)

    margin_y = margin * image_height / fy
    screen_y_min = -margin_y - cy / fy
    screen_y_max = margin_y + (image_height - cy) / fy
    screen_y = torch.clamp(mean3[..., 1] * iz, screen_y_min, screen_y_max)

    zeros = torch.zeros_like(mean3[..., 0])
    # fmt: off
    J = rearrange([
        fx*iz, zeros, -fx*screen_x*iz,
        zeros, fy*iz, -fy*screen_y*iz,
    ], "(i j) ... -> ... i j", i=2, j=3)
    # fmt: on
    cov2 = torch.einsum("...ij,...jk,...lk->...il", J, cov3, J)
    mean2 = torch.stack(
        [mean3[..., 0] * iz * fx + cx, mean3[..., 1] * iz * fy + cy], dim=-1
    )
    return mean2, cov2


def rand_Ks(N, device, dtype, image_height, image_width, min_fovx, max_fovx):
    fov_x = torch.rand(N, dtype=dtype, device=device) * (max_fovx - min_fovx) + min_fovx
    fovx_radian = fov_x * math.pi / 180
    fx = 0.5 * image_width / torch.tan(0.5 * fovx_radian)
    fy = fx
    Ks = torch.zeros(N, 3, 3, dtype=dtype, device=device)
    Ks[:, 0, 0] = fx
    Ks[:, 1, 1] = fy
    Ks[:, 0, 2] = image_width * 0.5
    Ks[:, 1, 2] = image_height * 0.5
    Ks[:, 2, 2] = 1
    return Ks


def test_fwd():
    N = 1024
    mean3 = torch.randn(N, 3, dtype=dtype, device=device)
    mean3[:, 2] += 1
    mean3[:, 2].clamp_(min=5e-2)
    cov3 = rand_sym_mat(N, 3, device, dtype, 1e-3)
    Ks = rand_Ks(N, device, dtype, image_height, image_width, 30, 90)

    mean2, cov2 = persp_proj(mean3, cov3, Ks, image_width, image_height)
    _mean2, _cov2 = persp_proj_torch(mean3, cov3, Ks, image_width, image_height)
    torch.testing.assert_close(mean2, _mean2, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(cov2, _cov2, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("N", [256, 1023, 1024, 1025])
def test_autograd(N):
    mean3 = torch.randn(N, 3, dtype=dtype, device=device)
    mean3[:, 2] += 2
    mean3[:, 2].clamp_(min=1e-1)
    cov3 = rand_sym_mat(N, 3, device, dtype, 1e-3)
    Ks = rand_Ks(N, device, dtype, image_height, image_width, 30, 90)

    mean3.requires_grad = True
    cov3.requires_grad = True

    v_mean2 = torch.randn(N, 2, dtype=dtype, device=device)
    v_cov2 = rand_sym_mat(N, 2, device, dtype, 1e-3)

    mean2, cov2 = PerspProj.apply(mean3, cov3, Ks, image_width, image_height)
    (v_mean3, v_cov3) = torch.autograd.grad(
        (mean2 * v_mean2).sum() + (cov2 * v_cov2).sum(), (mean3, cov3)
    )

    _mean2, _cov2 = persp_proj_torch(mean3, cov3, Ks, image_width, image_height)
    (_v_mean3, _v_cov3) = torch.autograd.grad(
        (_mean2 * v_mean2).sum() + (_cov2 * v_cov2).sum(), (mean3, cov3)
    )
    torch.testing.assert_close(mean2, _mean2, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(cov2, _cov2, rtol=5e-3, atol=5e-4)
    torch.testing.assert_close(v_mean3, _v_mean3, rtol=1e-1, atol=3e-2)
    torch.testing.assert_close(v_cov3, _v_cov3, rtol=1e-2, atol=1e-2)
