import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("calc_compensations", [True, False])
def test_fully_fused_projection(
    test_data,
    calc_compensations: bool,
):
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,
        _quat_scale_to_covar_preci,
    )

    from gsplat.triton_impl._wrapper import (
        _FullyFusedProjection as _FullyFusedProjectionTriton,
    )

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]

    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    eps2D = 0.3
    near_plane = 0.01
    far_plane = 1e9
    radius_clip = 0.0

    ################################ torch #############################
    _covars, _ = _quat_scale_to_covar_preci(quats, scales)  # [N, 3, 3]
    _radii, _means2d, _depths, _conics, _compensations = _fully_fused_projection(
        means,
        _covars,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=calc_compensations,
        eps2d=eps2D,
        near_plane=near_plane,
        far_plane=far_plane,
    )
    _masks = _radii.detach() > 0
    ############################## end torch ###########################

    ################################ triton ############################
    radii, means2d, depths, conics, compensations = _FullyFusedProjectionTriton.apply(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2D,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
    )
    masks = radii.detach() > 0
    ############################### end triton ###########################
    valid = _masks & masks
    print(
        f"mask percentage: cuda: {masks.sum().item() / masks.numel() * 100:.2f}% triton: {_masks.sum().item() / _masks.numel() * 100:.2f}%"
    )
    torch.testing.assert_close(radii, _radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid])
    if calc_compensations:
        torch.testing.assert_close(
            compensations[valid], _compensations[valid], atol=1e-3, rtol=5e-4
        )

    ######### backward ##########
    v_means2d = torch.randn_like(means2d)
    v_depths = torch.randn_like(depths)
    v_conics = torch.randn_like(_conics)
    if calc_compensations:
        v_compensations = torch.randn_like(compensations)
    else:
        v_compensations = None

    # Torch
    _loss = (
        (_means2d * v_means2d * valid[..., None]).sum()
        + (_depths * v_depths * valid).sum()
        + (_conics * v_conics * valid[..., None]).sum()
    )
    if calc_compensations:
        _loss += (_compensations * v_compensations * valid).sum()

    (_v_means, _v_quats, _v_scales, _v_viewmats) = torch.autograd.grad(
        _loss, (means, quats, scales, viewmats)
    )

    # TRITON
    loss = (
        (means2d * v_means2d * valid[..., None]).sum()
        + (depths * v_depths * valid).sum()
        + (conics * v_conics * valid[..., None]).sum()
    )
    if calc_compensations:
        loss += (compensations * v_compensations * valid).sum()

    (v_means, v_quats, v_scales, v_viewmats) = torch.autograd.grad(
        loss, (means, quats, scales, viewmats)
    )

    torch.testing.assert_close(v_means, _v_means, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_quats, _v_quats, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(v_scales, _v_scales, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-4, atol=1e-4)
    print("## done bwd")
