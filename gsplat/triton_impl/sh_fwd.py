"""
the spherical harmonic basis are as follows:

Y_{l,m} =
    sqrt(2) * K_{l,-m} * g_{m} * q_{l,-m}   (if m < 0)
    K_{l,0} * q_{l,0}                       (if m = 0)
    sqrt(2) * K_{l,m} * h_{m} * q_{l,m}     (if m > 0)

where
    i)   K_{l,m} = sqrt((2l+1)/4pi * (l-m)!/(l+m)!)
    ii)  g_{m}(x,y) = sin(m * phi) * sin(theta)^m
    iii) h_{m}(x,y) = cos(m * phi) * sin(theta)^m
    iv)  q_{l,m}(z) = P_{l,m}(cos(theta))

we have
    x = cos(phi) * sin(theta)
    y = sin(phi) * sin(theta)
    z = cos(theta)
so g,h,q are polynomials of x,y,z

Recurrence relations of g_m and h_m are based on trigonometric identities:
    g_{m+1} = x * g_{m} + y * h_{m}
    h_{m+1} = x * h_{m} - y * g_{m}
    g_0 = 0, h_0 = 1

Formula of P_{l,m} is based on Rodrigues' formula:
    P_{l,m}(z) = (1 / (2^l * l!)) * (d^(m+l) / dz^(m+l))[(z^2-1)^l]
    Recurrence relation of P_{l,m}:
        (l-m+1)P_{l+1,m}(z) = (2l+1)z * P_{l,m}(z) - (l+m)P_{l-1,m}(z)

Here we precompute the coefficients and use the recurrent relations to compute q,h,q
"""

import math
from typing import Optional, Union
from triton.language.extra import libdevice
import torch
import triton
import triton.language as tl
from jaxtyping import Bool, Float, Int
from torch import Tensor

# its not necessary to use the negative sign
# but we follow the convention of the orginal paper for compatibility
# let denote N_l = 1/(l! * 2^l)
# fmt: off
C0_0 = 0.28209479177387814  # K_{0,0} * N_0

C1_0 = 0.4886025119029199   # K_{1,0} * N_1 
C1_1 = -C1_0                # K_{1,1} * N_1 * sqrt(2)  

C2_0 = 0.31539156525252005  # K_{2,0} * N_2 * 4
C2_1 = -1.0925484305920792  # K_{2,1} * N_2 * 24 * sqrt(2)
C2_2 = 0.5462742152960396   # K_{2,2} * N_2 * 24 * sqrt(2)

C3_0 = 0.3731763325901154   # K_{3,0} * N_3 * 24
C3_1 = -0.4570457994644658  # K_{3,1} * N_3 * 72 * sqrt(2)
C3_2 = 1.445305721320277    # K_{3,2} * N_3 * 720 * sqrt(2)
C3_3 = -0.5900435899266435  # K_{3,3} * N_3 * 720 * sqrt(2)

C4_0 = 0.10578554691520431  # K_{4,0} * N_4 * 48
C4_1 = -0.6690465435572892  # K_{4,1} * N_4 * 960 * sqrt(2)
C4_2 = 0.47308734787878004  # K_{4,2} * N_4 * 2880 * sqrt(2)
C4_3 = -1.7701307697799304  # K_{4,3} * N_4 * 40320 * sqrt(2)
C4_4 = 0.6258357354491761   # K_{4,4} * N_4 * 40320 * sqrt(2)
# fmt: on


@triton.heuristics({"num_warps": lambda args: max(1, args["BLOCK_SIZE"] // 32)})
@triton.jit
def _sh_to_color_kernel(
    N,
    COLOR_DIM: tl.constexpr,
    K: tl.constexpr,
    deg: tl.constexpr,  # l
    sh_ptr,  # float32 [N, K, COLOR_DIM]
    dir_ptr,  # float32 [N, 3]
    color_ptr,  # float32 [N, COLOR_DIM]
    BLOCK_SIZE: tl.constexpr,
):
    KC = K * COLOR_DIM
    ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = ids < N * COLOR_DIM
    c = ids % COLOR_DIM
    offsets = ids // COLOR_DIM

    # add offset of color channel index
    c_sh_ptr = sh_ptr + c + offsets * KC

    # sh[:,0] for l=0, m=0
    # C0_0 = 0.28209479177387814
    res = 0.28209479177387814 * tl.load(c_sh_ptr, masks, 0.0)

    if deg >= 1:
        x = tl.load(dir_ptr + offsets * 3, masks, 0.0)
        y = tl.load(dir_ptr + offsets * 3 + 1, masks, 0.0)
        z = tl.load(dir_ptr + offsets * 3 + 2, masks, 0.0)
        inorm = libdevice.rsqrt(x * x + y * y + z * z)
        x *= inorm
        y *= inorm
        z *= inorm

        # sh[:,1:4] for l=1, m=-1,0,1
        # C1_0 = 0.4886025119029199
        # fmt: off
        res += 0.4886025119029199 * (
            - y * tl.load(c_sh_ptr + 1*COLOR_DIM, masks, 0.0)
            + z * tl.load(c_sh_ptr + 2*COLOR_DIM, masks, 0.0)
            - x * tl.load(c_sh_ptr + 3*COLOR_DIM, masks, 0.0)
        )
        # fmt: on

        if deg >= 2:
            zz = z * z

            # g1 = y, h1 = x
            g2 = 2.0 * x * y  # g_2
            h2 = x * x - y * y  # h_2

            # sh[:,4:9] for l=2, m=-2,-1,0,1,2
            # C2_0 = 0.31539156525252005
            # C2_1 = -1.0925484305920792
            # C2_2 = 0.5462742152960396
            # P2_0 = 3z2-1
            # C2_0_P2_0 = 0.9461746957575601 * zz - 0.3153915652525201
            # P2_1 = z
            C2_1_P2_1 = -1.0925484305920792 * z
            # fmt: off
            res += (
                    0.5462742152960396  * g2 * tl.load(c_sh_ptr + 4*COLOR_DIM, masks, 0.0)
                + C2_1_P2_1           *  y * tl.load(c_sh_ptr + 5*COLOR_DIM, masks, 0.0)
                + (0.9461746957575601 * zz - 0.3153915652525201) * tl.load(c_sh_ptr + 6*COLOR_DIM, masks, 0.0)
                + C2_1_P2_1           *  x * tl.load(c_sh_ptr + 7*COLOR_DIM, masks, 0.0)
                + 0.5462742152960396  * h2 * tl.load(c_sh_ptr + 8*COLOR_DIM, masks, 0.0)
            )
            # fmt: on

            if deg >= 3:
                # sh[:,9:16] for l=3, m=-3,-2,-1,0,1,2,3
                g3 = x * g2 + y * h2
                h3 = x * h2 - y * g2
                # C3_0 = 0.3731763325901154
                # C3_1 = -0.4570457994644658
                # C3_2 = 1.445305721320277
                # C3_3 = -0.5900435899266435
                # P3_0 = 5z3-3z
                # C3_0_P3_0 = z * (1.865881662950577 * zz - 1.119528997770346)
                # P3_1 = 5z2-1
                C3_1_P3_1 = -2.285228997322329 * zz + 0.4570457994644658
                # fmt: off
                res += (
                        -0.5900435899266435 * g3     * tl.load(c_sh_ptr +  9*COLOR_DIM, masks, 0.0)
                    + 1.445305721320277  * g2 * z * tl.load(c_sh_ptr + 10*COLOR_DIM, masks, 0.0)
                    + C3_1_P3_1               * y * tl.load(c_sh_ptr + 11*COLOR_DIM, masks, 0.0)
                    + (z * (1.865881662950577 * zz - 1.119528997770346)) * tl.load(c_sh_ptr + 12*COLOR_DIM, masks, 0.0)
                    + C3_1_P3_1               * x * tl.load(c_sh_ptr + 13*COLOR_DIM, masks, 0.0)
                    + 1.445305721320277  * h2 * z * tl.load(c_sh_ptr + 14*COLOR_DIM, masks, 0.0)
                    - 0.5900435899266435 * h3     * tl.load(c_sh_ptr + 15*COLOR_DIM, masks, 0.0)
                )
                # fmt: on

                if deg >= 4:
                    # sh[:,16:25] for l=4, m=-4,-3,-2,-1,0,1,2,3,4
                    g4 = x * g3 + y * h3
                    h4 = x * h3 - y * g3
                    # C4_0 = 0.10578554691520431
                    # C4_1 = -0.6690465435572892
                    # C4_2 = 0.47308734787878004
                    # C4_3 = -1.7701307697799304
                    # C4_4 = 0.6258357354491761
                    # fmt: off
                    # P4_0 = 35z4-30z2+3
                    # C4_0_P4_0 = zz * (3.7024941420321507 * zz - 3.1735664074561294) + 0.31735664074561293
                    # P4_1 = 7z3-3z
                    C4_1_P4_1 = z * (-4.683325804901024 * zz + 2.0071396306718676)
                    # P4_2 = 7z2-1
                    C4_2_P4_2 = 3.31161143515146 * zz - 0.47308734787878
                    res += (
                            0.6258357354491761 * g4     * tl.load(c_sh_ptr + 16*COLOR_DIM, masks, 0.0)
                        - 1.7701307697799304 * g3 * z * tl.load(c_sh_ptr + 17*COLOR_DIM, masks, 0.0)
                        + C4_2_P4_2          * g2     * tl.load(c_sh_ptr + 18*COLOR_DIM, masks, 0.0)
                        + C4_1_P4_1               * y * tl.load(c_sh_ptr + 19*COLOR_DIM, masks, 0.0)
                        + (zz * (3.7024941420321507 * zz - 3.1735664074561294) + 0.31735664074561293) * tl.load(c_sh_ptr + 20*COLOR_DIM, masks, 0.0)
                        + C4_1_P4_1               * x * tl.load(c_sh_ptr + 21*COLOR_DIM, masks, 0.0)
                        + C4_2_P4_2          * h2 *     tl.load(c_sh_ptr + 22*COLOR_DIM, masks, 0.0)
                        - 1.7701307697799304 * h3 * z * tl.load(c_sh_ptr + 23*COLOR_DIM, masks, 0.0)
                        + 0.6258357354491761 * h4     * tl.load(c_sh_ptr + 24*COLOR_DIM, masks, 0.0)
                    )
                    # fmt: on

    tl.store(color_ptr + ids, res, masks)


def sh_to_color_fwd(
    degree: int,
    dirs: Float[Tensor, "... 3"],
    coeffs: Float[Tensor, "... K C"],
    block_size: int = None,
) -> Float[Tensor, "... C"]:

    assert 0 <= degree <= 4

    *_, K, C = coeffs.shape
    N = coeffs.numel() // (K * C)
    assert 25 >= K >= (degree + 1) ** 2

    dirs = dirs.contiguous()
    coeffs = coeffs.contiguous()

    colors = torch.empty(
        *coeffs.shape[:-2], C, device=coeffs.device, dtype=coeffs.dtype
    )
    # hardcode autotune
    if block_size is None:
        block_size = {
            0: 32,
            1: 32,
            2: 32,
            3: 32,
            4: 32,
        }[degree]
    n_blocks = int(math.ceil(N * C / block_size))
    _sh_to_color_kernel[n_blocks,](
        N,
        C,
        K,
        degree,
        coeffs,
        dirs,
        colors,
        block_size,
    )
    return colors
