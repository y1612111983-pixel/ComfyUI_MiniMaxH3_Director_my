"""Reference-image batching helpers for Ref2VA Director Studio."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def batch_reference_images(images: list[torch.Tensor]):
    """Batch NHWC images without cropping, stretching, or changing pixels.

    ComfyUI IMAGE batches require a common height and width. Reference images
    may have arbitrary portrait or landscape dimensions, so every image is
    centered on the largest canvas in the current batch. Missing margins use
    edge replication; this avoids introducing a hard black frame into visual
    conditioning while preserving the complete original image and aspect.
    """
    values = [image for image in images if image is not None]
    if not values:
        return None
    for index, image in enumerate(values, start=1):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(f"参考图 {index} 不是有效的 ComfyUI IMAGE [B,H,W,C]。")
    channels = int(values[0].shape[-1])
    if any(int(image.shape[-1]) != channels for image in values):
        raise ValueError("参考图通道数不一致，无法组成批次。")
    target_height = max(int(image.shape[1]) for image in values)
    target_width = max(int(image.shape[2]) for image in values)
    padded = []
    for image in values:
        height, width = int(image.shape[1]), int(image.shape[2])
        pad_width = target_width - width
        pad_height = target_height - height
        left, right = pad_width // 2, pad_width - pad_width // 2
        top, bottom = pad_height // 2, pad_height - pad_height // 2
        if pad_width or pad_height:
            nchw = image.permute(0, 3, 1, 2)
            nchw = F.pad(nchw, (left, right, top, bottom), mode="replicate")
            image = nchw.permute(0, 2, 3, 1)
        padded.append(image)
    return torch.cat(padded, dim=0).contiguous()
