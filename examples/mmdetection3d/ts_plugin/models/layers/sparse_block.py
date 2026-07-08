from torch import nn
import torch_lattice.nn as spnn

from ..backbones.resnet import BasicBlockTS
from mmcv.cnn import build_conv_layer, build_norm_layer

import logging


def replace_feature_ts(out, new_features):
    out.feats = new_features
    return out


class SparseBasicBlockTS(BasicBlockTS):
    """Sparse basic block for PartA^2."""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        BasicBlockTS.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )
        if act_cfg is not None:
            if act_cfg == "swish":
                self.relu = spnn.SiLU(inplace=True)
            else:
                self.relu = spnn.ReLU(inplace=True)


def _stride_tuple(stride):
    if isinstance(stride, int):
        return (stride, stride, stride)
    return tuple(stride)


def _make_torch_lattice_conv(
    *,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    bias,
    conv_type,
):
    if conv_type == "TorchLatticeConvTranspose3d":
        return spnn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    if conv_type == "TorchLatticeConv3d" and _stride_tuple(stride) == (1, 1, 1):
        return spnn.SubmConv3d(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
        )
    if conv_type == "TorchLatticeConv3d":
        return spnn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    raise ValueError(f"unsupported TorchLattice conv_type: {conv_type}")


def make_sparse_convmodule_ts(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    conv_type="TorchLatticeConv3d",
    norm_cfg=None,
    order=("conv", "norm", "act"),
    activation_type="relu",
    indice_key=None,
):
    """Make sparse convolution module."""
    del indice_key
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {"conv", "norm", "act"} == {"conv", "norm", "act"}

    conv_cfg = {"type": conv_type}

    if norm_cfg is None:
        norm_cfg = dict(type="BN1d")

    layers = []
    for layer in order:
        if layer == "conv":
            if conv_type.startswith("TorchLattice"):
                layers.append(
                    _make_torch_lattice_conv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                        conv_type=conv_type,
                    )
                )
            else:
                layers.append(
                    build_conv_layer(
                        cfg=conv_cfg,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
        elif layer == "norm":
            assert norm_cfg is not None, "norm_cfg must be provided"
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == "act":
            if activation_type == "relu":
                layers.append(spnn.ReLU(inplace=True))
            elif activation_type == "swish":
                layers.append(spnn.SiLU(inplace=True))
            else:
                raise NotImplementedError
    layers = nn.Sequential(*layers)
    logging.info("Made TorchLattice Module")
    return layers
