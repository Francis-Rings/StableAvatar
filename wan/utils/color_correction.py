#!/usr/bin/env python3
import torch
import numpy as np
from skimage import color


def match_and_blend_colors(source_chunk: torch.Tensor, reference_image: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Matches the color of a source video chunk to a reference image and blends with the original.

    Args:
        source_chunk (torch.Tensor): The video chunk to be color-corrected (B, C, T, H, W) in range [-1, 1].
                                     Assumes B=1 (batch size of 1).
        reference_image (torch.Tensor): The reference image (B, C, 1, H, W) in range [-1, 1].
                                        Assumes B=1 and T=1 (single reference frame).
        strength (float): The strength of the color correction (0.0 to 1.0).
                          0.0 means no correction, 1.0 means full correction.

    Returns:
        torch.Tensor: The color-corrected and blended video chunk.
    """
    # print(f"[match_and_blend_colors] Input source_chunk shape: {source_chunk.shape}, reference_image shape: {reference_image.shape}, strength: {strength}")

    if strength == 0.0:
        # print(f"[match_and_blend_colors] Strength is 0, returning original source_chunk.")
        return source_chunk

    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

    device = source_chunk.device
    dtype = source_chunk.dtype

    # Squeeze batch dimension, permute to T, H, W, C for skimage
    # Source: (1, C, T, H, W) -> (T, H, W, C)
    source_np = source_chunk.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    # Reference: (1, C, 1, H, W) -> (H, W, C)
    ref_np = reference_image.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy()  # Squeeze T dimension as well

    # Normalize from [-1, 1] to [0, 1] for skimage
    source_np_01 = (source_np + 1.0) / 2.0
    ref_np_01 = (ref_np + 1.0) / 2.0

    # Clip to ensure values are strictly in [0, 1] after potential float precision issues
    source_np_01 = np.clip(source_np_01, 0.0, 1.0)
    ref_np_01 = np.clip(ref_np_01, 0.0, 1.0)

    # Convert reference to Lab
    try:
        ref_lab = color.rgb2lab(ref_np_01)
    except ValueError as e:
        # Handle potential errors if image data is not valid for conversion
        print(f"Warning: Could not convert reference image to Lab: {e}. Skipping color correction for this chunk.")
        return source_chunk

    corrected_frames_np_01 = []
    for i in range(source_np_01.shape[0]):  # Iterate over time (T)
        source_frame_rgb_01 = source_np_01[i]

        try:
            source_lab = color.rgb2lab(source_frame_rgb_01)
        except ValueError as e:
            print(f"Warning: Could not convert source frame {i} to Lab: {e}. Using original frame.")
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue

        corrected_lab_frame = source_lab.copy()

        # Perform color transfer for L, a, b channels
        for j in range(3):  # L, a, b
            mean_src, std_src = source_lab[:, :, j].mean(), source_lab[:, :, j].std()
            mean_ref, std_ref = ref_lab[:, :, j].mean(), ref_lab[:, :, j].std()

            # Avoid division by zero if std_src is 0
            if std_src == 0:
                # If source channel has no variation, keep it as is, but shift by reference mean
                # This case is debatable, could also just copy source or target mean.
                # Shifting by target mean helps if source is flat but target isn't.
                corrected_lab_frame[:, :, j] = mean_ref
            else:
                corrected_lab_frame[:, :, j] = (corrected_lab_frame[:, :, j] - mean_src) * (
                            std_ref / std_src) + mean_ref

        try:
            fully_corrected_frame_rgb_01 = color.lab2rgb(corrected_lab_frame)
        except ValueError as e:
            print(f"Warning: Could not convert corrected frame {i} back to RGB: {e}. Using original frame.")
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue

        # Clip again after lab2rgb as it can go slightly out of [0,1]
        fully_corrected_frame_rgb_01 = np.clip(fully_corrected_frame_rgb_01, 0.0, 1.0)

        # Blend with original source frame (in [0,1] RGB)
        blended_frame_rgb_01 = (1 - strength) * source_frame_rgb_01 + strength * fully_corrected_frame_rgb_01
        corrected_frames_np_01.append(blended_frame_rgb_01)

    corrected_chunk_np_01 = np.stack(corrected_frames_np_01, axis=0)

    # Convert back to [-1, 1]
    corrected_chunk_np_minus1_1 = (corrected_chunk_np_01 * 2.0) - 1.0

    # Permute back to (C, T, H, W), add batch dim, and convert to original torch.Tensor type and device
    # (T, H, W, C) -> (C, T, H, W)
    corrected_chunk_tensor = torch.from_numpy(corrected_chunk_np_minus1_1).permute(3, 0, 1, 2).unsqueeze(0)
    corrected_chunk_tensor = corrected_chunk_tensor.contiguous()  # Ensure contiguous memory layout
    output_tensor = corrected_chunk_tensor.to(device=device, dtype=dtype)
    # print(f"[match_and_blend_colors] Output tensor shape: {output_tensor.shape}")
    return output_tensor