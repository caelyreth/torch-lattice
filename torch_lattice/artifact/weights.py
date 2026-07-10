from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["PackedWeight", "dequantize_artifact_weight", "pack_quantized_weight"]


@dataclass(frozen=True, slots=True)
class PackedWeight:
    """Affine group-quantized artifact storage tensors."""

    weight: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor


def dequantize_artifact_weight(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    scale_dtype: str = "f16",
) -> torch.Tensor:
    """Round-trip a logical weight through artifact quantization."""

    source = tensor.detach().cpu().contiguous()
    rows, kernel_rows, out_channels = _weight_rows(source.to(torch.float32))
    in_channels = rows.shape[1]
    packed = pack_quantized_weight(
        source,
        bits=bits,
        group_size=group_size,
        scale_dtype=scale_dtype,
    )
    packed_weight = packed.weight
    scales = packed.scales
    biases = packed.biases
    if kernel_rows > 1:
        packed_weight = packed_weight.transpose(1, 2).contiguous()
        scales = scales.transpose(1, 2).contiguous()
        biases = biases.transpose(1, 2).contiguous()
    dequantized_rows = _dequantize_packed_rows(
        packed_weight.reshape(kernel_rows * out_channels, -1),
        scales.reshape(kernel_rows * out_channels, -1),
        biases.reshape(kernel_rows * out_channels, -1),
        group_size=group_size,
        bits=bits,
    )[:, :in_channels]
    if source.ndim == 2:
        return dequantized_rows.reshape(out_channels, in_channels).to(source.dtype)
    if source.ndim == 3:
        return (
            dequantized_rows.reshape(kernel_rows, out_channels, in_channels)
            .transpose(1, 2)
            .contiguous()
            .to(source.dtype)
        )
    if source.ndim == 5:
        _, kx, ky, kz, _ = source.shape
        return (
            dequantized_rows.reshape(kx, ky, kz, out_channels, in_channels)
            .permute(3, 0, 1, 2, 4)
            .contiguous()
            .to(source.dtype)
        )
    raise ValueError("unsupported quantized artifact weight rank")


def pack_quantized_weight(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    scale_dtype: str,
) -> PackedWeight:
    """Pack a logical artifact weight using affine group quantization."""

    if bits not in {4, 8}:
        raise ValueError("artifact weight bits must be 4 or 8")
    if group_size <= 0:
        raise ValueError("artifact weight group_size must be positive")
    if scale_dtype not in {"f16", "f32"}:
        raise ValueError("artifact scale_dtype must be 'f16' or 'f32'")
    rows, kernel_rows, out_channels = _weight_rows(tensor.to(torch.float32))
    storage_channels = _round_up(rows.shape[1], group_size)
    if storage_channels != rows.shape[1]:
        rows = torch.nn.functional.pad(rows, (0, storage_channels - rows.shape[1]))
    grouped = rows.reshape(rows.shape[0], -1, group_size)
    maximum = grouped.amax(dim=2)
    minimum = grouped.amin(dim=2)
    qmax = float((1 << bits) - 1)
    scales = (minimum - maximum) / qmax
    biases = maximum
    normalized = torch.where(
        scales.unsqueeze(2) != 0,
        (grouped - biases.unsqueeze(2)) / scales.unsqueeze(2),
        torch.zeros_like(grouped),
    )
    codes = (
        normalized.round()
        .clamp(0, qmax)
        .to(torch.uint32)
        .reshape(rows.shape[0], storage_channels)
    )
    packed = _pack_codes(codes, bits)
    scale_type = torch.float16 if scale_dtype == "f16" else torch.float32
    packed = packed.reshape(kernel_rows, out_channels, -1).contiguous()
    scales = scales.to(scale_type).reshape(kernel_rows, out_channels, -1).contiguous()
    biases = biases.to(scale_type).reshape(kernel_rows, out_channels, -1).contiguous()
    if kernel_rows > 1:
        packed = packed.transpose(1, 2).contiguous()
        scales = scales.transpose(1, 2).contiguous()
        biases = biases.transpose(1, 2).contiguous()
    return PackedWeight(packed.cpu(), scales.cpu(), biases.cpu())


def _dequantize_packed_rows(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    values_per_word = 32 // bits
    shifts = torch.arange(values_per_word, dtype=torch.int64) * bits
    mask = (1 << bits) - 1
    codes = ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & mask).reshape(
        packed.shape[0], -1
    )
    return codes.to(torch.float32) * scales.to(torch.float32).repeat_interleave(
        group_size, dim=1
    ) + biases.to(torch.float32).repeat_interleave(group_size, dim=1)


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    values_per_word = 32 // bits
    words = codes.to(torch.int64).reshape(codes.shape[0], -1, values_per_word)
    packed = torch.zeros(words.shape[:2], dtype=torch.int64)
    for lane in range(values_per_word):
        packed |= words[:, :, lane] << (bits * lane)
    return packed.to(torch.uint32)


def _weight_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if tensor.ndim == 2:
        out_channels, _ = tensor.shape
        return tensor, 1, int(out_channels)
    if tensor.ndim == 3:
        kernel_rows, in_channels, out_channels = tensor.shape
        rows = tensor.transpose(1, 2).reshape(kernel_rows * out_channels, in_channels)
        return rows, int(kernel_rows), int(out_channels)
    if tensor.ndim == 5:
        out_channels, kx, ky, kz, in_channels = tensor.shape
        rows = tensor.permute(1, 2, 3, 0, 4).reshape(
            kx * ky * kz * out_channels, in_channels
        )
        return rows, int(kx * ky * kz), int(out_channels)
    raise ValueError(
        "quantized artifact supports linear, kernel-major, and 5D convolution weights"
    )


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)
