# Acknowledgement: This source file was adapted from PIQ
# library with minor modifications.

# [1] PyTorch Image Quality: Metrics for Image Quality Assessment.
# Kastryulin, Sergey and Zakirov, Jamil and Prokopenko, Denis and Dylov, Dmitry V.
# arXiv 2022. https://arxiv.org/abs/2208.14818.


import math
from typing import Optional, Union

import torch
from piq.functional import binomial_filter1d, rgb2yiq
from piq.iw_ssim import _information_content, _pyr_step, _ssim_per_channel
from piq.utils import _reduce, _validate_input


def gaussian_filter(kernel_size: int, sigma: float, device: Optional[str] = None,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=dtype, device=device)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


def cb_information_weighted_ssim(x: torch.Tensor, y: torch.Tensor, eff_crack_map: torch.tensor,
                                 data_range: Union[int, float] = 1.,
                                 kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                                 parent: bool = True, blk_size: int = 3, sigma_nsq: float = 0.4,
                                 scale_weights: Optional[torch.Tensor] = None,
                                 reduction: str = 'mean') -> torch.Tensor:
    r"""Interface of Information Content Weighted Structural Similarity (IW-SSIM) index.
    Inputs supposed to be in range ``[0, data_range]``.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        eff_crack_map: the effective crack map tensor. Shape :math:`(N, 1, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution for sliding window used in comparison.
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        parent: Flag to control dependency on previous layer of pyramid.
        blk_size: The side-length of the sliding window used in comparison for information content.
        sigma_nsq: Parameter of visual distortion model.
        scale_weights: Weights for scaling.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Returns:
        Value of Information Content Weighted Structural Similarity (IW-SSIM) index.

    References:
        Wang, Zhou, and Qiang Li..
        Information content weighting for perceptual image quality assessment.
        IEEE Transactions on image processing 20.5 (2011): 1185-1198.
        https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf DOI:`10.1109/TIP.2010.2092435`

    Note:
        Lack of content in target image could lead to RuntimeError due to singular information content matrix,
        which cannot be inverted.
    """
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

    _validate_input(tensors=[x, y], dim_range=(4, 4), data_range=(0., data_range))

    x = x / float(data_range) * 255
    y = y / float(data_range) * 255

    if x.size(1) == 3:
        x = rgb2yiq(x)[:, :1]
        y = rgb2yiq(y)[:, :1]

    if scale_weights is None:
        scale_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=x.dtype, device=x.device)
    scale_weights = scale_weights / scale_weights.sum()
    if scale_weights.size(0) != scale_weights.numel():
        raise ValueError(f'Expected a vector of weights, got {scale_weights.dim()}D tensor')

    levels = scale_weights.size(0)

    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    blur_pad = math.ceil((kernel_size - 1) / 2)  # Ceil
    iw_pad = blur_pad - math.floor((blk_size - 1) / 2)  # floor
    gauss_kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(x)

    # Size of the kernel size to build Laplacian pyramid
    pyramid_kernel_size = 5
    bin_filter = binomial_filter1d(kernel_size=pyramid_kernel_size).to(x) * 2 ** 0.5

    lo_x, x_diff_old = _pyr_step(x, bin_filter)
    lo_y, y_diff_old = _pyr_step(y, bin_filter)

    x = lo_x
    y = lo_y
    wmcs = []

    for i in range(levels):
        if i < levels - 2:
            lo_x, x_diff = _pyr_step(x, bin_filter)
            lo_y, y_diff = _pyr_step(y, bin_filter)
            x = lo_x
            y = lo_y

        else:
            x_diff = x
            y_diff = y

        ssim_map, cs_map = _ssim_per_channel(x=x_diff_old, y=y_diff_old, kernel=gauss_kernel, data_range=255,
                                             k1=k1, k2=k2)

        if parent and i < levels - 2:
            iw_map = _information_content(x=x_diff_old, y=y_diff_old, y_parent=y_diff, kernel_size=blk_size,
                                          sigma_nsq=sigma_nsq)

            iw_map = iw_map[:, :, iw_pad:-iw_pad, iw_pad:-iw_pad]

        elif i == levels - 1:
            iw_map = torch.ones_like(cs_map)
            cs_map = ssim_map

        else:
            iw_map = _information_content(x=x_diff_old, y=y_diff_old, y_parent=None, kernel_size=blk_size,
                                          sigma_nsq=sigma_nsq)
            iw_map = iw_map[:, :, iw_pad:-iw_pad, iw_pad:-iw_pad]

        if eff_crack_map is not None:
            iw_map = iw_map * eff_crack_map[:, :, blur_pad:-blur_pad, blur_pad:-blur_pad]
            crack_enhanced_map = cs_map * iw_map / torch.sum(iw_map, dim=(-2, -1))
            wmcs.append(torch.sum(crack_enhanced_map, dim=(-2, -1)))
        else:
            wmcs.append(torch.sum(cs_map * iw_map, dim=(-2, -1)) / torch.sum(iw_map, dim=(-2, -1)))

        x_diff_old = x_diff
        y_diff_old = y_diff

        if eff_crack_map is not None:
            eff_crack_map, _ = _pyr_step(eff_crack_map, bin_filter)

    wmcs = torch.stack(wmcs, dim=0).abs()

    score = torch.prod((wmcs ** scale_weights.view(-1, 1, 1)), dim=0)[:, 0]

    return _reduce(x=score, reduction=reduction)
