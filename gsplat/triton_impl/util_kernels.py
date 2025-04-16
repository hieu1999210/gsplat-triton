import triton
import triton.language as tl


@triton.jit
def _inverse_sym_mat2(
    Mxx,  # Float [B]
    Mxy,  # Float [B]
    Myy,  # Float [B]
):
    """
    M^T=M
    iM @ M = I

    args:
        M: [B, 2, 2] symmetric matrices
    return:
        iM: [B, 2, 2] inverse of M (symmetric)
    """
    inv_Det = 1.0 / (Mxx * Myy - Mxy * Mxy)
    iMxx = inv_Det * Myy
    iMxy = -inv_Det * Mxy
    iMyy = inv_Det * Mxx
    return iMxx, iMxy, iMyy


@triton.jit
def _inverse_sym_mat2_vjp(
    d_iMxx,
    d_iMxy,
    d_iMyy,
    iMxx,
    iMxy,
    iMyy,
):
    """
    compute D_M from D_iM
    for general matrix M we have D_M = -iM^T @ D_iM @ iM^T
    for symmetric matrix M it is simplied to D_M = -iM @ D_iM @ iM

    args:
        d_iM: [B, 2, 2] symmetric matrices
        iM: [B, 2, 2] symmetric matrices

    return:
        D_M: [B, 2, 2] symmetric matrices

    NOTE: while M is symmetric, we treat it as a general matrix in gradient computation
    i.e. Mxy and Myx are 2 independent variables that happen to have the same values
    """
    D_Mxx = -(iMxx * d_iMxx * iMxx + iMxy * d_iMyy * iMxy + 2.0 * iMxy * d_iMxy * iMxx)
    D_Mxy = -(
        # fmt: off
          iMxx * d_iMxy * iMyy
        + iMxy * d_iMxy * iMxy
        + iMxy * d_iMyy * iMyy
        + iMxx * d_iMxx * iMxy
        # fmt: on
    )
    D_Myy = -(iMxy * d_iMxx * iMxy + iMyy * d_iMyy * iMyy + 2.0 * iMyy * d_iMxy * iMxy)
    return D_Mxx, D_Mxy, D_Myy


@triton.jit
def _add_blur(
    # symmetric 2x2 matrix M float32 [B]
    Mxx,
    Mxy,
    Myy,
    eps,  # float32
):
    """
    assume matrix M is semi-positive definite
    add eps*I to the covariance matrix to avoid singularity
    the additional term makes it strictly positive definite

    args:
        M: [B, 2, 2] symmetric matrices
        eps: float32

    return
        new_det: float32 [B]
        compensation: float32 [B]
        new_Mxx, new_Myy: float32 [B] updated entries of M

    NOTE: add_blur is ussually used before taking inverse of the matrix
    to do "pseudo-inverse" of the matrix
    """
    old_det = Mxx * Myy - Mxy * Mxy
    Mxx += eps
    Myy += eps
    new_det = Mxx * Myy - Mxy * Mxy
    compensation = tl.sqrt(tl.clamp(old_det / new_det, min=0.0, max=float("inf")))
    return new_det, compensation, Mxx, Myy


@triton.jit
def _add_blur_vjp(
    # inverse of the blur matrix
    iM_blur_xx,
    iM_blur_xy,
    iM_blur_yy,
    compensation,
    v_compensation,
    eps,
):
    """
    compute D_M from D_compensation
    where compenstation = sqrt(det(M)/det(M + eps*I))

    args:
        iM_blur: [B, 2, 2] symmetric matrices
        compensation: [B]
        D_compensation: [B]
        eps: float32

    return:
        D_M: [B, 2, 2] symmetric matrices

    NOTE: formula derivation

        denote a = det(M)/det(M + eps*I)
        D_comp/D_a = 0.5 / comp

        denote
            N = M + eps*I,
            detN = det(N), detM = det(M)
        D_detN/D_N = adj(N) = detN * N^-T
        adj(N) = adj(M) + epsI

        we have D_(detM/detN)/D_M
        = (detN * D_detM/D_M - detM * D_detN/D_M) / detN^2
        = (detN * adj(M) - detM * adj(N)) / detN^2
        = (adj(N) - epsI) / detN - a * adj(N) / detN
        = (1-a) N^-T - epsI / detN
        = (1-comp^2) * iM_blur - epsI * det(iM_blur)
    """
    det_iM_blur = iM_blur_xx * iM_blur_yy - iM_blur_xy * iM_blur_xy
    D_a = 0.5 * v_compensation / (compensation + 1e-6)
    one_minus_a = 1.0 - compensation * compensation
    D_Mxx = D_a * (one_minus_a * iM_blur_xx - eps * det_iM_blur)
    D_Mxy = D_a * (one_minus_a * iM_blur_xy)
    D_Myy = D_a * (one_minus_a * iM_blur_yy - eps * det_iM_blur)
    return D_Mxx, D_Mxy, D_Myy


@triton.jit
def _matmul3x3(
    # fmt: off
    Axx, Axy, Axz,
    Ayx, Ayy, Ayz,
    Azx, Azy, Azz,

    Bxx, Bxy, Bxz,
    Byx, Byy, Byz,
    Bzx, Bzy, Bzz,
    # fmt: on
):
    """
    C = A @ B

    args:
        A: [B, 3, 3]
        B: [B, 3, 3]

    return:
        C: [B, 3, 3]
    """
    Cxx = Axx * Bxx + Axy * Byx + Axz * Bzx
    Cxy = Axx * Bxy + Axy * Byy + Axz * Bzy
    Cxz = Axx * Bxz + Axy * Byz + Axz * Bzz

    Cyx = Ayx * Bxx + Ayy * Byx + Ayz * Bzx
    Cyy = Ayx * Bxy + Ayy * Byy + Ayz * Bzy
    Cyz = Ayx * Bxz + Ayy * Byz + Ayz * Bzz

    Czx = Azx * Bxx + Azy * Byx + Azz * Bzx
    Czy = Azx * Bxy + Azy * Byy + Azz * Bzy
    Czz = Azx * Bxz + Azy * Byz + Azz * Bzz
    return (
        # fmt: off
        Cxx, Cxy, Cxz, 
        Cyx, Cyy, Cyz, 
        Czx, Czy, Czz,
        # fmt: on
    )
