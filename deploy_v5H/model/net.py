"""The semantic network for the v5H package: nnU-Net's ResEnc-M topology, binary head.

Deliberately NOT imported from scripts/train_v5.py. That module pulls in the training stack
(the generator, the instancer, the benchmark loader) and hard-codes development paths; a
deployment package must stand on its own. The plan below is a verbatim copy of nnU-Net's own
`plans.json` for Dataset501_MTSynth, so the architecture is reproduced from the same source
the training run read, not from memory.

The head is ONE channel, not the K=6 of the previous package. Where that matters is documented
at the call site in predict.py.
"""
from __future__ import annotations

import torch.nn as nn

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

#: nnU-Net 2D ResEnc-M plan for Dataset501_MTSynth. Eight stages: seven successive /2
#: downsamplings give the bottleneck a receptive field spanning the whole 512 px patch, which
#: is what lets it integrate evidence along a filament rather than only across it.
PLAN = dict(
    n_stages=8,
    features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512],
    kernel_sizes=[[3, 3]] * 8,
    strides=[[1, 1]] + [[2, 2]] * 7,
    n_blocks_per_stage=[1, 3, 4, 6, 6, 6, 6, 6],
    n_conv_per_stage_decoder=[1] * 7,
)

#: Input must be divisible by 128 (seven /2 stages) or the residual adds hit a shape mismatch
#: on the way back up. 518 -- the previous package's tile, matching DINOv2's /14 patch grid --
#: is not, and fails at run time rather than at load time.
TILE, STRIDE = 512, 388

IMA_M = [0.485, 0.456, 0.406]
IMA_S = [0.229, 0.224, 0.225]


def build(out_channels: int = 1, deep_supervision: bool = False):
    """Inference-time model. Deep supervision is off: only the full-resolution head is used."""
    return ResidualEncoderUNet(
        input_channels=3, num_classes=out_channels, conv_op=nn.Conv2d,
        conv_bias=True, norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision, **PLAN)


def head_width(state_dict) -> int:
    """Read the head width off the checkpoint rather than assuming it.

    nnU-Net's ResEnc names its output layers ``decoder.seg_layers.N.weight``. Assuming a width
    fails loudly only by luck -- during development a default of 7 happened to match the models
    tested first, so the detection went unexercised until a 1-channel checkpoint reached it.
    """
    seg = [v for k, v in state_dict.items()
           if "decoder.seg_layers." in k and k.endswith(".weight")]
    if not seg:
        raise SystemExit("checkpoint has no decoder.seg_layers.*.weight -- not a ResEnc U-Net?")
    return int(seg[0].shape[0])
