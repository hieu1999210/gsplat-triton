import math
from typing import Optional, Union, Tuple

import torch
import triton
import triton.language as tl
from jaxtyping import Bool, Float, Int
from torch import Tensor
from triton.language.extra import libdevice

"""
Recurrence relations of g_m and h_m are based on trigonometric identities:
    g_{m+1} = x * g_{m} + y * h_{m}
    h_{m+1} = x * h_{m} - y * g_{m}
    g_0 = 0, h_0 = 1

Recurrence relations for partial derivatives of g_m and h_m:
    g_1_x = 0, g_1_y = 1
    h_1_x = 1, h_1_y = 0

    g_{m+1}_x = g_m + x * g_m_x + y * h_m_x
    g_{m+1}_y = h_m + x * g_m_y + y * h_m_y

    h_{m+1}_x =  h_m + x * h_m_x - y * g_m_x
    h_{m+1}_y = -g_m + x * h_m_y - y * g_m_y

    then we have (induction by m)
    g_{m+1}_x = (m+1) * g_m
    g_{m+1}_y = (m+1) * h_m

    h_{m+1}_x = (m+1) * h_m
    h_{m+1}_y = -(m+1) * g_m
"""


@triton.heuristics({"num_warps": lambda args: max(1, args["BLOCK_SIZE"] // 32)})
@triton.jit
def _sh_to_color_vjp_kernel(
    N,
    COLOR_DIM: tl.constexpr,
    K: tl.constexpr,
    deg: tl.constexpr,  # l
    sh_ptr,  # float32 [N, K, COLOR_DIM]
    dir_ptr,  # float32 [N, 3]
    # fwd output grad
    v_color_ptr,  # float32 [N, COLOR_DIM]
    # input grad
    v_sh_ptr,  # float32 [N, K, COLOR_DIM]
    has_v_dir: tl.constexpr,
    v_dir_ptr,  # float32 [N, 3]
    BLOCK_SIZE: tl.constexpr,
):
    KC = K * COLOR_DIM
    ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    masks = ids < N * COLOR_DIM
    c = ids % COLOR_DIM
    offsets = ids // COLOR_DIM

    # add offset of color channel index
    c_sh_ptr = sh_ptr + c + offsets * KC
    v_c_sh_ptr = v_sh_ptr + c + offsets * KC

    v_color = tl.load(v_color_ptr + ids, masks, 0.0)

    # sh[:,0] for l=0, m=0
    # C0_0 = 0.28209479177387814
    tl.store(v_c_sh_ptr, 0.28209479177387814 * v_color, masks)

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
        tl.store(v_c_sh_ptr + 1*COLOR_DIM, -0.4886025119029199 * y * v_color, masks)
        tl.store(v_c_sh_ptr + 2*COLOR_DIM,  0.4886025119029199 * z * v_color, masks)
        tl.store(v_c_sh_ptr + 3*COLOR_DIM, -0.4886025119029199 * x * v_color, masks)
        # fmt: on

        if has_v_dir:
            # fmt: off
            v_y = -0.4886025119029199 * tl.load(c_sh_ptr + 1 * COLOR_DIM, masks, 0.0)
            v_z =  0.4886025119029199 * tl.load(c_sh_ptr + 2 * COLOR_DIM, masks, 0.0)
            v_x = -0.4886025119029199 * tl.load(c_sh_ptr + 3 * COLOR_DIM, masks, 0.0)
            # fmt: on

        if deg >= 2:
            zz = z * z
            xz = x * z
            yz = y * z

            # g1 = y, h1 = x
            g2 = 2.0 * x * y  # g_2
            h2 = x * x - y * y  # h_2

            # sh[:,4:9] for l=2, m=-2,-1,0,1,2
            # C2_0 = 0.31539156525252005
            # C2_1 = -1.0925484305920792
            # C2_2 = 0.5462742152960396
            # fmt: off
            C2_1_P2_1 = -1.0925484305920792 * z
            tl.store(v_c_sh_ptr + 4*COLOR_DIM, 0.5462742152960396  * g2 * v_color, masks)
            tl.store(v_c_sh_ptr + 5*COLOR_DIM, C2_1_P2_1           *  y * v_color, masks)
            tl.store(v_c_sh_ptr + 6*COLOR_DIM, (0.9461746957575601 * zz - 0.3153915652525201) * v_color, masks)
            tl.store(v_c_sh_ptr + 7*COLOR_DIM, C2_1_P2_1           *  x * v_color, masks)
            tl.store(v_c_sh_ptr + 8*COLOR_DIM, 0.5462742152960396  * h2 * v_color, masks)
            # fmt: on

            if has_v_dir:
                # g2_x = 2g1 = 2*y
                # g2_y = 2h1 = 2*x
                # h2_x = 2h1 = 2*x
                # h2_y = -2g1 = -2*y

                # preload sh2p2 sh2m2 sh2p1 sh2m1
                sh2m2 = tl.load(c_sh_ptr + 4 * COLOR_DIM, masks, 0.0)
                sh2m1 = tl.load(c_sh_ptr + 5 * COLOR_DIM, masks, 0.0)
                sh2p1 = tl.load(c_sh_ptr + 7 * COLOR_DIM, masks, 0.0)
                sh2p2 = tl.load(c_sh_ptr + 8 * COLOR_DIM, masks, 0.0)
                # fmt: off
                # v_x = (C2_2 * g2_x * sh2m2) + (C2_1 * z * sh2p1) + (C2_2 * h2_x * sh2p2)
                # 
                C2_2_2g1 = 1.0925484305920792 * y # = C2_2 * gx_x = - C2_2 * g2_y
                C2_2_2h1 = 1.0925484305920792 * x # = C2_2 * g2_y = C2_2 * h2_x
                v_x += (
                        C2_2_2g1  * sh2m2
                    + C2_1_P2_1 * sh2p1
                    + C2_2_2h1  * sh2p2
                )

                # v_y = (C2_2 * g2_y * sh2m2) + (C2_1 * P2_1 * sh2m1) + (C2_2 * h2_y * sh2p2)
                v_y += (
                        C2_2_2h1  * sh2m2
                    + C2_1_P2_1 * sh2m1
                    - C2_2_2g1  * sh2p2
                )

                # v_z = (C2_1 * g1 * sh2m1) + (C2_0 * P2_0_z * sh2m0) + (C2_1 * h1 * sh2p1)
                # g1 = y, h1 = x, P2_0_z = 6z
                # C2_1 * g1 = -C2_2 * 2g1
                # C2_1 * h1 = -C2_2 * 2h1 since C2_1 = -2 * C2_2
                # C2_0 * P2_0_z = 0.31539156525252005 * 6.0 * z
                v_z += (
                    - C2_2_2g1 * sh2m1
                    + 1.8923493915151202 * z * tl.load(c_sh_ptr + 6*COLOR_DIM, masks, 0.0)
                    - C2_2_2h1 * sh2p1
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
                C3_1_P3_1 = -2.2852289973223288 * zz + 0.4570457994644658
                # fmt: off
                tl.store(v_c_sh_ptr +  9*COLOR_DIM, -0.5900435899266435 * g3 * v_color, masks)
                tl.store(v_c_sh_ptr + 10*COLOR_DIM,  1.445305721320277  * g2 * z * v_color, masks)
                tl.store(v_c_sh_ptr + 11*COLOR_DIM,  C3_1_P3_1               * y * v_color, masks)
                tl.store(v_c_sh_ptr + 12*COLOR_DIM,  (z * (1.865881662950577 * zz - 1.119528997770346)) * v_color, masks)
                tl.store(v_c_sh_ptr + 13*COLOR_DIM,  C3_1_P3_1               * x * v_color, masks)
                tl.store(v_c_sh_ptr + 14*COLOR_DIM,  1.445305721320277  * h2 * z * v_color, masks)
                tl.store(v_c_sh_ptr + 15*COLOR_DIM, -0.5900435899266435 * h3 * v_color, masks)
                # fmt: on

                if has_v_dir:
                    # g3_x = 3*g2
                    # g3_y = 3*h2
                    # h3_x = 3*h2
                    # h3_y = -3*g2

                    # preload sh3p3 sh3m3 sh3p2 sh3m2 sh3p1 sh3m1
                    sh3m3 = tl.load(c_sh_ptr + 9 * COLOR_DIM, masks, 0.0)
                    sh3m2 = tl.load(c_sh_ptr + 10 * COLOR_DIM, masks, 0.0)
                    sh3m1 = tl.load(c_sh_ptr + 11 * COLOR_DIM, masks, 0.0)
                    sh3p1 = tl.load(c_sh_ptr + 13 * COLOR_DIM, masks, 0.0)
                    sh3p2 = tl.load(c_sh_ptr + 14 * COLOR_DIM, masks, 0.0)
                    sh3p3 = tl.load(c_sh_ptr + 15 * COLOR_DIM, masks, 0.0)

                    # fmt: off
                    """
                    v_x = (
                            C3_3 * g3_x * sh3m3
                        + C3_2 * g2_x * z * sh3m2
                        + C3_1 * g1_x * P3_1 * sh3m1 # g1_x = 0
                        + C3_1 * h1_x * P3_1 * sh3p1
                        + C3_2 * h2_x * z * sh3p2
                        + C3_3 * h3_x * sh3p3
                    )
                    """
                    C3_3_3g2 = -1.7701307697799304 * g2 # = C3_3 * g3_x = -C3_3 * h3_y
                    c3_3_3h2 = -1.7701307697799304 * h2 # = C3_3 * h3_x =  C3_3 * g3_y
                    C3_2_2h1_P3_2 = 2.890611442640554 * yz # 2g1 = g2_x = -h2_y
                    C3_2_2g1_P3_2 = 2.890611442640554 * xz # 2h1 = h2_x =  g2_y

                    v_x += (
                            C3_3_3g2      * sh3m3
                        + C3_2_2h1_P3_2 * sh3m2
                        + C3_1_P3_1     * sh3p1
                        + C3_2_2g1_P3_2 * sh3p2
                        + c3_3_3h2      * sh3p3
                    )
                    """
                    v_y = (
                        C3_3 * g3_y * sh3m3
                        + C3_2 * g2_y * z * sh3m2
                        + C3_1 * g1_y * P3_1 * sh3m1
                        + C3_1 * h1_y * P3_1 * sh3p1 # h1_y = 0
                        + C3_2 * h2_y * z * sh3p2
                        + C3_3 * h3_y * sh3p3
                    )
                    """
                    v_y += (
                            c3_3_3h2      * sh3m3
                        + C3_2_2g1_P3_2 * sh3m2
                        + C3_1_P3_1     * sh3m1
                        - C3_2_2h1_P3_2 * sh3p2
                        - C3_3_3g2      * sh3p3
                    )
                    """
                    v_z = (
                        C3_2 * g2 * sh3m2
                        + C3_1 * g1 * P3_1_z * sh3m1
                        + C3_0 *      P3_0_z * sh3m0
                        + C3_1 * h1 * P3_1_z * sh3p1
                        + C3_2 * h2 * sh3m2
                    )
                    P3_1_z = 10z
                    P3_0_z = 3*P3_1
                    """
                    # C3_0 * P3_0_z = C3_0 * 3 * P3_1  = (3*C3_0/C3_1) * C3_1 * P3_1 
                    v_z += (
                            1.445305721320277 * g2 * sh3m2
                        - 4.570457994644658 * yz * sh3m1
                        - 2.449489742783178 * C3_1_P3_1 * tl.load(c_sh_ptr + 12*COLOR_DIM, masks, 0.0)
                        - 4.570457994644658 * xz * sh3p1
                        + 1.445305721320277 * h2 * sh3p2
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
                    tl.store(v_c_sh_ptr + 16*COLOR_DIM, 0.6258357354491761  * g4     * v_color, masks)
                    tl.store(v_c_sh_ptr + 17*COLOR_DIM, -1.7701307697799304 * g3 * z * v_color, masks)
                    tl.store(v_c_sh_ptr + 18*COLOR_DIM, C4_2_P4_2           * g2     * v_color, masks)
                    tl.store(v_c_sh_ptr + 19*COLOR_DIM, C4_1_P4_1                * y * v_color, masks)
                    tl.store(v_c_sh_ptr + 20*COLOR_DIM, ((zz * (3.7024941420321507 * zz - 3.1735664074561294) + 0.31735664074561293)) * v_color, masks)
                    tl.store(v_c_sh_ptr + 21*COLOR_DIM, C4_1_P4_1                * x * v_color, masks)
                    tl.store(v_c_sh_ptr + 22*COLOR_DIM, C4_2_P4_2           * h2     * v_color, masks)
                    tl.store(v_c_sh_ptr + 23*COLOR_DIM, -1.7701307697799304 * h3 * z * v_color, masks)
                    tl.store(v_c_sh_ptr + 24*COLOR_DIM, 0.6258357354491761  * h4     * v_color, masks)
                    # fmt: on

                    if has_v_dir:
                        # preload sh4p4 sh4m4 sh4p3 sh4m3 sh4p2 sh4m2 sh4p1 sh4m1
                        sh4m4 = tl.load(c_sh_ptr + 16 * COLOR_DIM, masks, 0.0)
                        sh4m3 = tl.load(c_sh_ptr + 17 * COLOR_DIM, masks, 0.0)
                        sh4m2 = tl.load(c_sh_ptr + 18 * COLOR_DIM, masks, 0.0)
                        sh4m1 = tl.load(c_sh_ptr + 19 * COLOR_DIM, masks, 0.0)
                        sh4p1 = tl.load(c_sh_ptr + 21 * COLOR_DIM, masks, 0.0)
                        sh4p2 = tl.load(c_sh_ptr + 22 * COLOR_DIM, masks, 0.0)
                        sh4p3 = tl.load(c_sh_ptr + 23 * COLOR_DIM, masks, 0.0)
                        sh4p4 = tl.load(c_sh_ptr + 24 * COLOR_DIM, masks, 0.0)

                        # fmt: off
                        C4_4_4g3 = 2.5033429417967046 * g3 # = C4_4 * g4_x = -C4_4 * h4_y
                        C4_4_4h3 = 2.5033429417967046 * h3 # = C4_4 * h4_x =  C4_4 * g4_y

                        C4_3_3g2_P4_3 = -5.310392309339791 * g2 * z # 3g2 = g3_x = -h3_y
                        C4_3_3h2_P4_3 = -5.310392309339791 * h2 * z # 3h2 = h3_x =  g3_y

                        C4_2_2g1_P4_2 = C4_2_P4_2 * 2.0 * y # 2g1 = g2_x = -h2_y
                        C4_2_2h1_P4_2 = C4_2_P4_2 * 2.0 * x # 2h1 = h2_x =  g2_y
                        """
                        v_x = (
                            C4_4 * g4_x * sh4m4
                            + C4_3 * g3_x * z * sh4m3
                            + C4_2 * g2_x * P4_2 * sh4m2
                            + C4_1 * g1_x * P4_1 * sh4m1 # g1_x = 0
                            + C4_1 * h1_x * P4_1 * sh4p1
                            + C4_2 * h2_x * P4_2 * sh4p2
                            + C4_3 * h3_x * z * sh4p3
                            + C4_4 * h4_x * sh4p4
                        )
                        """
                        v_x += (
                                C4_4_4g3 * sh4m4
                            + C4_3_3g2_P4_3 * sh4m3
                            + C4_2_2g1_P4_2 * sh4m2
                            + C4_1_P4_1     * sh4p1
                            + C4_2_2h1_P4_2 * sh4p2
                            + C4_3_3h2_P4_3 * sh4p3
                            + C4_4_4h3 * sh4p4
                        )
                        """
                        v_y = (
                            C4_4 * g4_y * sh4m4
                            + C4_3 * g3_y * z * sh4m3
                            + C4_2 * g2_y * P4_2 * sh4m2
                            + C4_1 * g1_y * P4_1 * sh4m1
                            + C4_1 * h1_y * P4_1 * sh4p1 # h1_y = 0
                            + C4_2 * h2_y * P4_2 * sh4p2
                            + C4_3 * h3_y * z * sh4p3
                            + C4_4 * h4_y * sh4p4
                        )
                        """
                        v_y += (
                                C4_4_4h3 * sh4m4
                            + C4_3_3h2_P4_3 * sh4m3
                            + C4_2_2h1_P4_2 * sh4m2
                            + C4_1_P4_1     * sh4m1
                            - C4_2_2g1_P4_2 * sh4p2
                            - C4_3_3g2_P4_3 * sh4p3
                            - C4_4_4g3 * sh4p4
                        )
                        """
                        v_z = (
                            C4_3 * g3 * sh4m3
                            + C4_2 * g2 * P4_2_z * sh4m2
                            + C4_1 * g1 * P4_1_z * sh4m1
                            + C4_0 *      P4_0_z * sh4m0
                            + C4_1 * h1 * P4_1_z * sh4p1
                            + C4_2 * h2 * P4_2_z * sh4p2
                            + C4_3 * h3 * z * sh4p3
                        )
                        P4_2_z = 14.0 * P4_3 = 14z
                        P4_1_z = 3 * P4_2
                        P4_0_z = 20 * P4_1
                        """
                        # C4_0 * P4_0_z = C4_0 * 20 * P4_1 = (20*C4_0/C4_1) * C4_1 * P4_1
                        # C4_1 * g1 * P4_1_z = C4_1 * g1 * 3 * P4_2 = (3*C4_1/C4_2/2) * C4_2 * 2g1 * P4_2
                        # C4_2 * g2 * P4_2_z = C4_2 * g2 * 14 * P4_3 = (14*C4_2/C4_3/3) * C4_3 * 3g2 * P4_3 
                        v_z += (
                                -1.7701307697799304 * g3 * sh4m3
                            - 1.2472191289246473 * C4_3_3g2_P4_3 * sh4m2
                            - 2.1213203435596424 * C4_2_2g1_P4_2 * sh4m1
                            - 3.162277660168379  * C4_1_P4_1 * tl.load(c_sh_ptr + 20*COLOR_DIM, masks, 0.0)
                            - 2.1213203435596424 * C4_2_2h1_P4_2 * sh4p1
                            - 1.2472191289246473 * C4_3_3h2_P4_3 * sh4p2
                            - 1.7701307697799304 * h3 * sh4p3
                        )
                        # fmt: on

    if has_v_dir and deg >= 1:
        v_x *= v_color
        v_y *= v_color
        v_z *= v_color

        # vjp of the normalization
        # d_dir = (d_dir_n - dir_n^T * d_dir_n * dir_n) * inv_norm
        dir_n_dot_v_dir_n = x * v_x + y * v_y + z * v_z
        v_x = (v_x - dir_n_dot_v_dir_n * x) * inorm
        v_y = (v_y - dir_n_dot_v_dir_n * y) * inorm
        v_z = (v_z - dir_n_dot_v_dir_n * z) * inorm
        tl.atomic_add(v_dir_ptr + 0 + offsets * 3, v_x, masks, sem="relaxed")
        tl.atomic_add(v_dir_ptr + 1 + offsets * 3, v_y, masks, sem="relaxed")
        tl.atomic_add(v_dir_ptr + 2 + offsets * 3, v_z, masks, sem="relaxed")


def sh_to_color_bwd(
    degree: int,
    dirs: Float[Tensor, "N 3"],
    coeffs: Float[Tensor, "N K C"],
    v_colors: Float[Tensor, "N C"],
    calc_v_dirs: bool = False,
    block_size: int = None,
) -> Tuple[
    Float[Tensor, "N K C"],  # v_coeffs
    Optional[Float[Tensor, "N 3"]],  # v_dirs
]:

    assert 0 <= degree <= 4

    *_, K, C = coeffs.shape
    N = coeffs.numel() // (K * C)
    assert 25 >= K >= (degree + 1) ** 2

    dirs = dirs.contiguous()
    coeffs = coeffs.contiguous()
    v_colors = v_colors.contiguous()

    v_coeffs = torch.zeros_like(coeffs)
    if calc_v_dirs:
        v_dirs = torch.zeros_like(dirs)
    else:
        v_dirs = torch.empty(1, device=dirs.device, dtype=dirs.dtype)
    # hardcoded kernel configs
    # hardcode autotune
    if block_size is None:
        block_size = {
            0: 32,
            1: 32,
            2: 32,
            3: 16,
            4: 16,
        }[degree]
    n_blocks = int(math.ceil(N * C / block_size))
    _sh_to_color_vjp_kernel[n_blocks,](
        N,
        C,
        K,
        degree,
        coeffs,
        dirs,
        v_colors,
        v_coeffs,
        calc_v_dirs,
        v_dirs,
        block_size,
    )
    if not calc_v_dirs:
        v_dirs = None
    return v_coeffs, v_dirs
