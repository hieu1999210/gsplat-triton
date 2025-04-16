import triton
import triton.language as tl


@triton.jit
def _persp_proj(
    # fmt: off
    # means: float32 [B]
    m3_x, m3_y, m3_z,
    # covariance
    c3_xx, c3_xy, c3_xz,
           c3_yy, c3_yz,
                  c3_zz,
    # fmt: on
    # camera intrinsics
    fx,  # float32
    fy,  # float32
    cx,  # float32
    cy,  # float32
    image_width,  # int32
    image_height,  # int32
    margin=0.15,
):
    """
    perspective projection of 3D gaussian to 2D gaussian

    args:
        m3: [B, 3] means 3D in camera space
        c3: [B, 3, 3] cov 3D in camera space, symmetric matrices
        fx, fy, cx, cy: float32 camera intrinsics
        image_width, image_height: int32 image size
        margin: percentage of screen size to clip the screen space coordinates

    return
        means2D: [B, 2] means 2D
        covs2D: [B, 2, 2] cov 2D symmetric matrices

    FORMULA:
    mean2D = [fx*m3_x/z, fy*m3_y/z] + [cx, cy]
    covs2D = J @ covs3D @ J^T

    NOTE: perspective projection of gaussian is not gaussian
        but we can approximate it using the jacobian of the projection
        covs2D = J @ covs3D @ J^T
        where J is jacobian of the projection
        J = [
            [fx/z,    0, -fx*x/z^2],
            [   0, fy/z, -fy*y/z^2]
        ]
    """
    iz = 1.0 / m3_z

    # screen space coordinates [x/z, y/z]
    ## clipping the screen space coordinates to be within 15% screen_size margin of image boundary
    ## image screen space boundary is
    ## [-cx/fx, (image_width - cx)/fx] x [-cy/fy, (image_height - cy)/fy]
    margin_x = margin * image_width.cast(tl.float32) / fx
    screen_x_min = -margin_x - cx / fx
    screen_x_max = margin_x + (image_width - cx) / fx
    screen_x = tl.clamp(m3_x * iz, screen_x_min, screen_x_max)

    margin_y = margin * image_height.cast(tl.float32) / fy
    screen_y_min = -margin_y - cy / fy
    screen_y_max = margin_y + (image_height - cy) / fy
    screen_y = tl.clamp(m3_y * iz, screen_y_min, screen_y_max)

    ######### compute J #########
    # J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
    Jxx = fx * iz
    Jxz = -fx * screen_x * iz
    Jyy = fy * iz
    Jyz = -fy * screen_y * iz

    # covar 2D = J @ covar 3D @ J^T
    covs2D_xx = Jxx * c3_xx * Jxx + 2.0 * Jxx * c3_xz * Jxz + Jxz * c3_zz * Jxz
    covs2D_xy = Jxx * (c3_xy * Jyy + c3_xz * Jyz) + Jxz * (c3_yz * Jyy + c3_zz * Jyz)
    covs2D_yy = Jyy * c3_yy * Jyy + 2.0 * Jyy * c3_yz * Jyz + Jyz * c3_zz * Jyz
    # project means 2D
    means2D_x = fx * m3_x * iz + cx
    means2D_y = fy * m3_y * iz + cy
    return means2D_x, means2D_y, covs2D_xx, covs2D_xy, covs2D_yy


@triton.jit
def _persp_proj_vjp(
    # fmt: off
    # means 3D float32 [B]
    m3_x, m3_y, m3_z,
    # covariance 3D float32 [B]
    c3_xx, c3_xy, c3_xz,
           c3_yy, c3_yz,
                  c3_zz,
    # camera intrinsics 
    fx, fy, cx, cy,  # float32
    image_width,  # int32
    image_height,  # int32
    # grad mean 2D float32 [B]
    v_m2_x, v_m2_y,
    # grad cov 2D float32 [B]
    v_c2_xx, v_c2_xy, v_c2_yy,
    # fmt: on
    margin=0.15,
):
    """
    compute grad of means 3D and cov 3D in camera space
    from grad of means 2D and cov 2D in image space

    args:
        m3: [B, 3] means 3D in camera space
        c3: [B, 3, 3] cov 3D in camera space, symmetric matrices
        fx, fy, cx, cy: float32 camera intrinsics
        image_width, image_height: int32 image size
        v_m2: [B, 2] grad of means 2D
        v_c2: [B, 2, 2] grad of cov 2D
        margin: float32 percentage of screen size to clip the screen space coordinates

    return
        v_m3: [B, 3] grad means 3D
        v_c3: [B, 3, 3] grad cov 3D symmetric matrices

    FORMULA:
    c2 = J @ c3 @ J^T
    D_c3 = J^T @ D_c2 @ J
    D_m3 is from both D_m2 and D_c2 (see below)

    NOTE: while C3, C2 is symmetric, we treat it as a general matrix in gradient computation
    i.e. Cxy and Cyx are 2 independent variables that happen to have the same values
    """

    iz = 1.0 / m3_z

    # project covariance 2D
    # NOTE: perspective projection of gaussian is not gaussian
    # but we can approximate it using the jacobian of the projection
    # J = [
    #   [fx/z, 0, -fx*x/z^2],
    #   [0, fy/z, -fy*y/z^2],
    # ]

    ######### compute J #########
    # screen space coordinates [x/z, y/z]
    ## clipping the screen space coordinates to be within 15% screen_size margin of image boundary
    ## image screen space boundary is [-cx/fx, (image_width - cx)/fx] x [-cy/fy, (image_height - cy)/fy]
    margin_x = margin * image_width.cast(tl.float32) / fx
    screen_x_min = -margin_x - cx / fx
    screen_x_max = margin_x + (image_width - cx) / fx
    screen_x = m3_x * iz
    # true mean clamped
    clamp_mask_x = (screen_x < screen_x_min) | (screen_x > screen_x_max)
    screen_x = tl.clamp(screen_x, screen_x_min, screen_x_max)

    margin_y = margin * image_height.cast(tl.float32) / fy
    screen_y_min = -margin_y - cy / fy
    screen_y_max = margin_y + (image_height - cy) / fy
    screen_y = m3_y * iz
    # true mean clamped
    clamp_mask_y = (screen_y < screen_y_min) | (screen_y > screen_y_max)
    screen_y = tl.clamp(screen_y, screen_y_min, screen_y_max)

    # J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
    Jxx = fx * iz
    Jxz = -fx * screen_x * iz
    Jyy = fy * iz
    Jyz = -fy * screen_y * iz
    ######### end compute J #########

    # covar 2D = J @ covar 3D @ J^T
    # D_covar3D = J^T @ D_covar2D @ J
    v_c3_xx = Jxx * v_c2_xx * Jxx
    v_c3_yy = Jyy * v_c2_yy * Jyy
    v_c3_zz = Jxz * v_c2_xx * Jxz + Jyz * v_c2_yy * Jyz + 2.0 * Jyz * v_c2_xy * Jxz
    v_c3_xy = Jxx * v_c2_xy * Jyy
    v_c3_xz = Jxx * v_c2_xx * Jxz + Jxx * v_c2_xy * Jyz
    v_c3_yz = Jyy * v_c2_xy * Jxz + Jyy * v_c2_yy * Jyz

    # grad of means 3D is from both mean 2D and cov 2D
    # D_m3 = D(m2/m3) * D_m2 + D(J/m3) * D(c2/J) * D_c2
    #
    # D(m2/m3) = J^T
    #
    # D_J from D_c2 is D(c2/J) * D_c2
    # D_J = D_c2 @ J @ c3^T + D_c2^T @ J @ c3
    #     = 2* D_c2 @ J @ c3   due to symmetry
    #
    # D_m3 from D_J is  D(J/m3) * D_J
    #   D_m3_x = D_Jxz*Jxz/x if not clamp sceen_x else 0
    #   D_m3_y = D_Jyz*Jyz/y if not clamp sceen_y else 0
    #   D_m3_z = - 1/z(
    #       D_Jxx*Jxx
    #       + D_Jyy*Jyy
    #       + 2*D_Jxz*Jxz if not clamp sceen_x else D_Jxz*Jxz
    #       + 2*D_Jyz*Jyz if not clamp sceen_y else D_Jyz*Jyz
    #   )

    # compute D_m3 from D_m2
    v_m3_x = Jxx * v_m2_x
    v_m3_y = Jyy * v_m2_y
    v_m3_z = Jxz * v_m2_x + Jyz * v_m2_y

    # compute D_m3 from D_c2
    ## compute D_J first = 2 * D_c2 @ J @ c3
    ### J @ c3
    J_c3_xx = Jxx * c3_xx + Jxz * c3_xz
    J_c3_xy = Jxx * c3_xy + Jxz * c3_yz
    J_c3_xz = Jxx * c3_xz + Jxz * c3_zz

    J_c3_yx = Jyy * c3_xy + Jyz * c3_xz
    J_c3_yy = Jyy * c3_yy + Jyz * c3_yz
    J_c3_yz = Jyy * c3_yz + Jyz * c3_zz

    ### D_J = 2 * D_c2 @ J @ c3 = 2 * D_c2 @ J_c3
    v_Jxx = 2.0 * (v_c2_xx * J_c3_xx + v_c2_xy * J_c3_yx)
    # v_Jxy = 2.0 * (v_c2_xx * J_c3_xy + v_c2_xy * J_c3_yy)
    v_Jxz = 2.0 * (v_c2_xx * J_c3_xz + v_c2_xy * J_c3_yz)

    # v_Jyx = 2.0 * (v_c2_xy * J_c3_xx + v_c2_yy * J_c3_yx)
    v_Jyy = 2.0 * (v_c2_xy * J_c3_xy + v_c2_yy * J_c3_yy)
    v_Jyz = 2.0 * (v_c2_xy * J_c3_xz + v_c2_yy * J_c3_yz)

    ### add grad from J to m3
    ### Jxz/m3_x = -fx/z^2
    ### Jyz/m3_y = -fy/z^2
    iz2 = iz * iz
    v_m3_x += tl.where(clamp_mask_x, 0.0, -v_Jxz * fx * iz2)
    v_m3_y += tl.where(clamp_mask_y, 0.0, -v_Jyz * fy * iz2)
    ### add grad from J to m3_z
    ### compute  the sum inside -1/z*(...)
    tmp = v_Jxx * Jxx + v_Jyy * Jyy + 2.0 * (v_Jxz * Jxz + v_Jyz * Jyz)
    tmp -= tl.where(clamp_mask_x, v_Jxz * Jxz, 0.0)
    tmp -= tl.where(clamp_mask_y, v_Jyz * Jyz, 0.0)
    v_m3_z -= iz * tmp

    return (
        # fmt: off
        # grad means 3D
        v_m3_x, v_m3_y, v_m3_z, 
        # grad cov 3D
        v_c3_xx, v_c3_xy, v_c3_xz,
                 v_c3_yy, v_c3_yz,
                          v_c3_zz,
        # fmt: on
    )
