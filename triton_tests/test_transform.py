import math

import pytest
import quaternion as qt
import torch
import triton
import triton.language as tl
from test_kernel_utils import (
    load_mat3x3,
    load_sym_mat3,
    load_vec3,
    store_mat3x3,
    store_sym_mat3,
    store_vec3,
)
from test_utils import rand_sym_mat

from gsplat.triton_impl.transform import (
    _transform_cov3D,
    _transform_cov3D_vjp,
    _transform_v3D,
    _transform_v3D_vjp,
)

BLOCK_SIZE = 256
dtype = torch.float32
device = torch.device("cuda")


def rand_Rmat(B, device, dtype):
    q = torch.rand(B, 4, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    assert not torch.any(torch.isnan(q))
    q = qt.as_quat_array(q.cpu().numpy())

    R = qt.as_rotation_matrix(q)
    R = torch.tensor(R, device=device, dtype=dtype)
    return R


@triton.jit
def _transform_v3D_kernel(N, V_ptr, R_ptr, T_ptr, U_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N
    Vx, Vy, Vz = load_vec3(offsets, V_ptr, masks)
    # fmt: off
    (
        Rxx, Rxy, Rxz, 
        Ryx, Ryy, Ryz, 
        Rzx, Rzy, Rzz,
    ) = load_mat3x3(offsets, R_ptr, masks)

    Tx, Ty, Tz = load_vec3(offsets, T_ptr, masks)

    Ux, Uy, Uz = _transform_v3D(
        Vx, Vy, Vz, 
        Rxx, Rxy, Rxz, 
        Ryx, Ryy, Ryz, 
        Rzx, Rzy, Rzz, 
        Tx, Ty, Tz
    )
    # fmt: on

    store_vec3(offsets, U_ptr, Ux, Uy, Uz, masks)


def transform_v3D(V, R, T):
    N = V.shape[0]
    V = V.contiguous()
    R = R.contiguous()
    T = T.contiguous()
    U = torch.empty_like(V)
    n_block = int(math.ceil(N / BLOCK_SIZE))
    _transform_v3D_kernel[
        n_block,
    ](N, V, R, T, U, BLOCK_SIZE)
    return U


@triton.jit
def _transform_v3D_vjp_kernel(
    N, V_ptr, R_ptr, d_U_ptr, d_V_ptr, d_R_ptr, d_T_ptr, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N
    Vx, Vy, Vz = load_vec3(offsets, V_ptr, masks)

    # fmt: off
    (
        Rxx, Rxy, Rxz, 
        Ryx, Ryy, Ryz, 
        Rzx, Rzy, Rzz,
    ) = load_mat3x3(offsets, R_ptr, masks)

    # fmt: on

    d_Ux, d_Uy, d_Uz = load_vec3(offsets, d_U_ptr, masks)

    # fmt: off
    (
        d_Vx, d_Vy, d_Vz,
        d_Tx, d_Ty, d_Tz,
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryx, d_Ryy, d_Ryz,
        d_Rzx, d_Rzy, d_Rzz,
    ) = _transform_v3D_vjp(
        Rxx, Rxy, Rxz, 
        Ryx, Ryy, Ryz, 
        Rzx, Rzy, Rzz, 
        Vx, Vy, Vz, 
        d_Ux, d_Uy, d_Uz
    )
    store_vec3(offsets, d_V_ptr, d_Vx, d_Vy, d_Vz, masks)

    store_mat3x3(
        offsets,
        d_R_ptr,
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryx, d_Ryy, d_Ryz,
        d_Rzx, d_Rzy, d_Rzz,
        masks,
    )

    # fmt: on
    store_vec3(offsets, d_T_ptr, d_Tx, d_Ty, d_Tz, masks)


def transform_v3D_vjp(R, V, d_U):
    N = V.shape[0]
    V = V.contiguous()
    R = R.contiguous()
    d_U = d_U.contiguous()
    d_V = torch.empty_like(V)
    d_R = torch.empty_like(R)
    d_T = torch.empty_like(V)
    n_block = int(math.ceil(N / BLOCK_SIZE))
    _transform_v3D_vjp_kernel[
        n_block,
    ](N, V, R, d_U, d_V, d_R, d_T, BLOCK_SIZE)
    return d_V, d_R, d_T


class TransformV3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V, R, T):
        U = transform_v3D(V, R, T)
        ctx.save_for_backward(V, R)
        return U

    @staticmethod
    def backward(ctx, d_U):
        V, R = ctx.saved_tensors
        return transform_v3D_vjp(R, V, d_U)


def transform_v3D_torch(V, R, T):
    """
    V: [... 3]
    R: [... 3 3]
    T: [... 3]
    """
    return torch.einsum("...ij,...j->...i", R, V) + T


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_fwd_vec():
    N = 1024
    V = torch.rand(N, 3, device=device, dtype=dtype)
    R = rand_Rmat(N, device, dtype)
    T = torch.rand(N, 3, device=device, dtype=dtype)

    U = transform_v3D(V, R, T)
    _U = transform_v3D_torch(V, R, T)
    torch.testing.assert_close(U, _U)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_autograd_vec():
    N = 1024
    V = torch.rand(N, 3, device=device, dtype=dtype)
    R = rand_Rmat(N, device, dtype)
    T = torch.rand(N, 3, device=device, dtype=dtype)

    V.requires_grad = True
    R.requires_grad = True
    T.requires_grad = True

    v_U = torch.randn_like(V)

    U = TransformV3D.apply(V, R, T)
    (v_V, v_R, v_T) = torch.autograd.grad((U * v_U).sum(), (V, R, T))

    _U = transform_v3D_torch(V, R, T)
    (_v_V, _v_R, _v_T) = torch.autograd.grad((_U * v_U).sum(), (V, R, T))

    torch.testing.assert_close(_U, U)
    torch.testing.assert_close(v_V, _v_V)
    torch.testing.assert_close(v_R, _v_R)
    torch.testing.assert_close(v_T, _v_T)


@triton.jit
def _transform_cov3D_kernel(N, C_ptr, R_ptr, S_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N
    Cxx, Cxy, Cxz, Cyy, Cyz, Czz = load_sym_mat3(offsets, C_ptr, masks)

    # fmt: off
    (
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
    ) = load_mat3x3(offsets, R_ptr, masks)

    (
        Sxx, Sxy, Sxz,
        Syy, Syz, Szz,
    ) = _transform_cov3D(
        Cxx, Cxy, Cxz,
        Cyy, Cyz, Czz,
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
    )

    store_sym_mat3(
        offsets,
        S_ptr,
        Sxx, Sxy, Sxz,
        Syy, Syz, Szz,
        masks,
    )
    # fmt: on


def transform_cov3D(C, R):
    N = C.shape[0]
    C = C.contiguous()
    R = R.contiguous()
    S = torch.empty_like(C)
    n_block = int(math.ceil(N / BLOCK_SIZE))
    _transform_cov3D_kernel[
        n_block,
    ](N, C, R, S, BLOCK_SIZE)
    return S


def transform_cov3D_torch(C, R):
    """
    R @ C @ R.T
    R: [... 3 3]
    C: [... 3 3]
    """
    return torch.einsum("...ij,...jk,...lk->...il", R, C, R)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_fwd_cov():
    N = 1024
    C = rand_sym_mat(N, 3, device, dtype)
    R = rand_Rmat(N, device, dtype)

    S = transform_cov3D(C, R)
    _S = transform_cov3D_torch(C, R)
    torch.testing.assert_close(S, _S)


@triton.jit
def _transform_cov3D_vjp_kernel(
    N, C_ptr, R_ptr, d_S_ptr, d_C_ptr, d_R_ptr, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = offsets < N
    # fmt: off
    (
        Cxx, Cxy, Cxz,
        Cyy, Cyz, Czz,
    ) = load_sym_mat3(offsets, C_ptr, masks)

    (
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
    ) = load_mat3x3(offsets, R_ptr, masks)

    (
        d_Sxx, d_Sxy, d_Sxz,
        d_Syy, d_Syz, d_Szz,
    ) = load_sym_mat3(offsets, d_S_ptr, masks)

    (
        d_Cxx, d_Cxy, d_Cxz,
        d_Cyy, d_Cyz, d_Czz,
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryx, d_Ryy, d_Ryz,
        d_Rzx, d_Rzy, d_Rzz,
    ) = _transform_cov3D_vjp(
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
        Cxx, Cxy, Cxz,
        Cyy, Cyz, Czz,
        d_Sxx, d_Sxy, d_Sxz,
        d_Syy, d_Syz, d_Szz,
    )

    store_sym_mat3(
        offsets,
        d_C_ptr,
        d_Cxx, d_Cxy, d_Cxz,
        d_Cyy, d_Cyz, d_Czz,
        masks,
    )

    store_mat3x3(
        offsets,
        d_R_ptr,
        d_Rxx, d_Rxy, d_Rxz,
        d_Ryx, d_Ryy, d_Ryz,
        d_Rzx, d_Rzy, d_Rzz,
        masks,
    )
    # fmt: on


def transform_cov3D_vjp(C, R, d_S):
    N = C.shape[0]
    C = C.contiguous()
    R = R.contiguous()
    d_S = d_S.contiguous()
    d_C = torch.empty_like(C)
    d_R = torch.empty_like(R)
    n_block = int(math.ceil(N / BLOCK_SIZE))
    _transform_cov3D_vjp_kernel[
        n_block,
    ](N, C, R, d_S, d_C, d_R, BLOCK_SIZE)
    return d_C, d_R


class TransformCov3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, R):
        S = transform_cov3D(C, R)
        ctx.save_for_backward(C, R)
        return S

    @staticmethod
    def backward(ctx, d_S):
        C, R = ctx.saved_tensors
        return transform_cov3D_vjp(C, R, d_S)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_autograd_cov():
    N = 1024
    C = rand_sym_mat(N, 3, device, dtype)
    R = rand_Rmat(N, device, dtype)

    C.requires_grad = True
    R.requires_grad = True

    v_S = torch.randn_like(C)
    v_S = v_S @ v_S.transpose(-1, -2)

    S = TransformCov3D.apply(C, R)
    (v_C, v_R) = torch.autograd.grad((v_S * S).sum(), (C, R))

    _S = transform_cov3D_torch(C, R)
    (_v_C, _v_R) = torch.autograd.grad((v_S * _S).sum(), (C, R))

    torch.testing.assert_close(S, _S)
    torch.testing.assert_close(v_R, _v_R)
    torch.testing.assert_close(v_C, _v_C)


def transform_torch(C, V, R, T):
    U = transform_v3D_torch(V, R, T)
    S = transform_cov3D_torch(C, R)
    return U, S


class Transform3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V, C, R, T):
        U = transform_v3D(V, R, T)
        S = transform_cov3D(C, R)
        ctx.save_for_backward(C, V, R)
        return U, S

    @staticmethod
    def backward(ctx, d_U, d_S):
        C, V, R = ctx.saved_tensors
        d_V, d_R, d_T = transform_v3D_vjp(R, V, d_U)
        d_C, _d_R = transform_cov3D_vjp(C, R, d_S)
        d_R += _d_R
        return d_V, d_C, d_R, d_T


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("N", [256, 1023, 1024, 1025, 4096])
def test_autograd_mean_cov(N):
    C = rand_sym_mat(N, 3, device, dtype)
    V = torch.rand(N, 3, device=device, dtype=dtype)
    R = rand_Rmat(N, device, dtype)
    T = torch.rand(N, 3, device=device, dtype=dtype)

    C.requires_grad = True
    V.requires_grad = True
    R.requires_grad = True
    T.requires_grad = True

    v_S = rand_sym_mat(N, 3, device, dtype)
    v_U = torch.randn_like(V)

    U, S = Transform3D.apply(V, C, R, T)
    (v_V, v_C, v_R, v_T) = torch.autograd.grad(
        (v_U * U).sum() + (v_S * S).sum(), (V, C, R, T)
    )

    _U, _S = transform_torch(C, V, R, T)
    (_v_V, _v_C, _v_R, _v_T) = torch.autograd.grad(
        (_U * v_U).sum() + (_S * v_S).sum(), (V, C, R, T)
    )

    torch.testing.assert_close(U, _U)
    torch.testing.assert_close(S, _S)
    torch.testing.assert_close(v_V, _v_V)
    torch.testing.assert_close(v_C, _v_C)
    torch.testing.assert_close(v_R, _v_R)
    torch.testing.assert_close(v_T, _v_T)
