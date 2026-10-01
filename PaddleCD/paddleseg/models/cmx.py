"""CMX-style RGB-X fusion adapted to the project's two-input dataset.

The original CMX implementation uses two MiT/SegFormer streams and applies
feature rectification (FRM) followed by feature fusion (FFM) at every MiT
stage.  This Paddle implementation keeps that structure while accepting the
project's two tensors directly:

    stream 1: the 6-channel MSI+SAR tensor (``t1``)
    stream 2: the HSI tensor (``t2``)

The channel counts are configurable so the same model can be used for BW and
AB, where the HSI branch has 116 and 242 channels respectively.
"""

import math

import paddle
from paddle import nn
from paddle.nn import functional as F

from paddleseg.cvlibs import manager
from paddleseg.models.backbones.mix_transformer import (
    MixVisionTransformer_B0,
    MixVisionTransformer_B1,
    MixVisionTransformer_B2,
    MixVisionTransformer_B3,
    MixVisionTransformer_B4,
    MixVisionTransformer_B5,
)
from paddleseg.models.backbones.transformer_utils import (
    ones_,
    trunc_normal_,
    zeros_,
)


def _init_cmx_weights(layer):
    """Initialize the modules introduced by CMX.

    The MiT branches initialize themselves, so this function is only applied
    to FRM/FFM and decoder modules.  Keeping the initialization local avoids
    overwriting pretrained MiT parameters.
    """
    if isinstance(layer, nn.Linear):
        trunc_normal_(layer.weight)
        if layer.bias is not None:
            zeros_(layer.bias)
    elif isinstance(layer, nn.LayerNorm):
        zeros_(layer.bias)
        ones_(layer.weight)
    elif isinstance(layer, nn.Conv2D):
        fan_out = layer._kernel_size[0] * layer._kernel_size[1] * layer._out_channels
        fan_out //= layer._groups
        std = math.sqrt(2.0 / fan_out)
        paddle.nn.initializer.Normal(0.0, std)(layer.weight)
        if layer.bias is not None:
            zeros_(layer.bias)


class _ChannelWeights(nn.Layer):
    """Channel-wise cross-modal weights used by the CMX FRM."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        hidden_dim = dim * 4 // reduction
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        batch = x1.shape[0]
        x = paddle.concat([x1, x2], axis=1)
        avg = self.avg_pool(x).reshape([batch, self.dim * 2])
        maximum = self.max_pool(x).reshape([batch, self.dim * 2])
        weights = self.mlp(paddle.concat([avg, maximum], axis=1))
        weights = weights.reshape([batch, 2, self.dim, 1, 1])
        return weights.transpose([1, 0, 2, 3, 4])


class _SpatialWeights(nn.Layer):
    """Spatial cross-modal weights used by the CMX FRM."""

    def __init__(self, dim, reduction=1):
        super().__init__()
        hidden_dim = max(dim // reduction, 1)
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2D(dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(hidden_dim, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        batch, _, height, width = x1.shape
        weights = self.mlp(paddle.concat([x1, x2], axis=1))
        weights = weights.reshape([batch, 2, 1, height, width])
        return weights.transpose([1, 0, 2, 3, 4])


class _FeatureRectifyModule(nn.Layer):
    """Rectify each stream using channel and spatial evidence from the other."""

    def __init__(self, dim, reduction=1, lambda_c=0.5, lambda_s=0.5):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = _ChannelWeights(dim, reduction)
        self.spatial_weights = _SpatialWeights(dim, reduction)
        self.apply(_init_cmx_weights)

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2
        out_x1 = out_x1 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1
        out_x2 = out_x2 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


class _CrossAttention(nn.Layer):
    """Efficient cross-path attention used by the CMX FFM."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                'CMX cross attention requires dim divisible by num_heads: '
                '{} vs {}'.format(dim, num_heads))
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads)**-0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)

    def forward(self, x1, x2):
        batch, tokens, channels = x1.shape
        head_dim = channels // self.num_heads
        q1 = x1.reshape([batch, tokens, self.num_heads, head_dim])
        q1 = q1.transpose([0, 2, 1, 3])
        q2 = x2.reshape([batch, tokens, self.num_heads, head_dim])
        q2 = q2.transpose([0, 2, 1, 3])

        kv1 = self.kv1(x1).reshape(
            [batch, tokens, 2, self.num_heads, head_dim])
        kv1 = kv1.transpose([2, 0, 3, 1, 4])
        kv2 = self.kv2(x2).reshape(
            [batch, tokens, 2, self.num_heads, head_dim])
        kv2 = kv2.transpose([2, 0, 3, 1, 4])

        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]
        context1 = paddle.matmul(k1.transpose([0, 1, 3, 2]), v1)
        context1 = F.softmax(context1 * self.scale, axis=-2)
        context2 = paddle.matmul(k2.transpose([0, 1, 3, 2]), v2)
        context2 = F.softmax(context2 * self.scale, axis=-2)

        out1 = paddle.matmul(q1, context2)
        out1 = out1.transpose([0, 2, 1, 3]).reshape([batch, tokens, channels])
        out2 = paddle.matmul(q2, context1)
        out2 = out2.transpose([0, 2, 1, 3]).reshape([batch, tokens, channels])
        return out1, out2


class _CrossPath(nn.Layer):
    """Split-projection and cross-attention path of the CMX FFM."""

    def __init__(self, dim, reduction=1, num_heads=None):
        super().__init__()
        reduced_dim = dim // reduction
        self.channel_proj1 = nn.Linear(dim, reduced_dim * 2)
        self.channel_proj2 = nn.Linear(dim, reduced_dim * 2)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.cross_attn = _CrossAttention(reduced_dim, num_heads=num_heads)
        self.end_proj1 = nn.Linear(reduced_dim * 2, dim)
        self.end_proj2 = nn.Linear(reduced_dim * 2, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        y1, u1 = paddle.split(self.act1(self.channel_proj1(x1)), 2, axis=-1)
        y2, u2 = paddle.split(self.act2(self.channel_proj2(x2)), 2, axis=-1)
        v1, v2 = self.cross_attn(u1, u2)
        out_x1 = self.norm1(x1 + self.end_proj1(paddle.concat([y1, v1], axis=-1)))
        out_x2 = self.norm2(x2 + self.end_proj2(paddle.concat([y2, v2], axis=-1)))
        return out_x1, out_x2


class _ChannelEmbed(nn.Layer):
    """Turn the concatenated token streams into a spatial fused feature."""

    def __init__(self, in_channels, out_channels, reduction=1):
        super().__init__()
        hidden_channels = out_channels // reduction
        self.residual = nn.Conv2D(
            in_channels, out_channels, kernel_size=1, bias_attr=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2D(in_channels, hidden_channels, kernel_size=1),
            nn.Conv2D(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels),
            nn.ReLU(),
            nn.Conv2D(hidden_channels, out_channels, kernel_size=1),
            nn.BatchNorm2D(out_channels),
        )
        self.norm = nn.BatchNorm2D(out_channels)

    def forward(self, x, height, width):
        batch, _, channels = x.shape
        x = x.transpose([0, 2, 1]).reshape([batch, channels, height, width])
        residual = self.residual(x)
        x = self.channel_embed(x)
        return self.norm(residual + x)


class _FeatureFusionModule(nn.Layer):
    """Cross-path plus channel embedding fusion used by CMX."""

    def __init__(self, dim, reduction=1, num_heads=1):
        super().__init__()
        self.cross = _CrossPath(dim, reduction, num_heads)
        self.channel_emb = _ChannelEmbed(dim * 2, dim, reduction)
        self.apply(_init_cmx_weights)

    def forward(self, x1, x2):
        batch, _, height, width = x1.shape
        x1 = x1.flatten(2).transpose([0, 2, 1])
        x2 = x2.flatten(2).transpose([0, 2, 1])
        x1, x2 = self.cross(x1, x2)
        return self.channel_emb(paddle.concat([x1, x2], axis=-1), height, width)


_MIT_FACTORIES = {
    'mit_b0': MixVisionTransformer_B0,
    'mit_b1': MixVisionTransformer_B1,
    'mit_b2': MixVisionTransformer_B2,
    'mit_b3': MixVisionTransformer_B3,
    'mit_b4': MixVisionTransformer_B4,
    'mit_b5': MixVisionTransformer_B5,
}


@manager.BACKBONES.add_component
class CMXBackbone(nn.Layer):
    """Two-stream MiT backbone with CMX FRM/FFM fusion at four stages."""

    def __init__(self,
                 in_channels1=6,
                 in_channels2=116,
                 backbone='mit_b2',
                 pretrained=None):
        super().__init__()
        backbone_name = backbone.lower()
        if backbone_name not in _MIT_FACTORIES:
            raise ValueError(
                'Unsupported CMX backbone {!r}; choose from {}'.format(
                    backbone, ', '.join(sorted(_MIT_FACTORIES))))

        factory = _MIT_FACTORIES[backbone_name]
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.stream1 = factory(in_channels=in_channels1, pretrained=pretrained)
        self.stream2 = factory(in_channels=in_channels2, pretrained=pretrained)
        self.feat_channels = list(self.stream1.feat_channels)

        num_heads = {
            'mit_b0': [1, 2, 5, 8],
            'mit_b1': [1, 2, 5, 8],
            'mit_b2': [1, 2, 5, 8],
            'mit_b3': [1, 2, 5, 8],
            'mit_b4': [1, 2, 5, 8],
            'mit_b5': [1, 2, 5, 8],
        }[backbone_name]
        self.frms = nn.LayerList([
            _FeatureRectifyModule(dim, reduction=1)
            for dim in self.feat_channels
        ])
        self.ffms = nn.LayerList([
            _FeatureFusionModule(dim, reduction=1, num_heads=head)
            for dim, head in zip(self.feat_channels, num_heads)
        ])

    @staticmethod
    def _stage_features(stream, x, stage_index):
        """Run one ordinary MiT stage and return a BCHW feature map."""
        patch_embed = getattr(stream, 'patch_embed{}'.format(stage_index))
        blocks = getattr(stream, 'block{}'.format(stage_index))
        norm = getattr(stream, 'norm{}'.format(stage_index))

        x, height, width = patch_embed(x)
        for block in blocks:
            x = block(x, height, width)
        x = norm(x)
        batch = x.shape[0]
        x = x.reshape([batch, height, width, -1]).transpose([0, 3, 1, 2])
        return x

    def forward(self, x1, x2):
        expected1 = self.in_channels1
        expected2 = self.in_channels2
        if x1.shape[1] not in (None, -1, expected1):
            raise ValueError(
                'CMX stream 1 expects {} channels, got {}'.format(
                    expected1, x1.shape[1]))
        if x2.shape[1] not in (None, -1, expected2):
            raise ValueError(
                'CMX stream 2 expects {} channels, got {}'.format(
                    expected2, x2.shape[1]))

        outputs = []
        for stage_index, (frm, ffm) in enumerate(zip(self.frms, self.ffms), 1):
            x1 = self._stage_features(self.stream1, x1, stage_index)
            x2 = self._stage_features(self.stream2, x2, stage_index)
            x1, x2 = frm(x1, x2)
            outputs.append(ffm(x1, x2))
        return outputs


class _CMXMLP(nn.Layer):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose([0, 2, 1]))


@manager.MODELS.add_component
class CMX(nn.Layer):
    """CMX-style two-stream semantic segmentation model."""

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim=768,
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()
        self.backbone = backbone
        self.align_corners = align_corners
        self.embedding_dim = embedding_dim
        c1, c2, c3, c4 = self.backbone.feat_channels
        self.linear_c4 = _CMXMLP(c4, embedding_dim)
        self.linear_c3 = _CMXMLP(c3, embedding_dim)
        self.linear_c2 = _CMXMLP(c2, embedding_dim)
        self.linear_c1 = _CMXMLP(c1, embedding_dim)
        self.linear_fuse = nn.Sequential(
            nn.Conv2D(embedding_dim * 4, embedding_dim, kernel_size=1),
            nn.BatchNorm2D(embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout2D(dropout_ratio)
        self.linear_pred = nn.Conv2D(embedding_dim, num_classes, kernel_size=1)
        # Do not call ``self.apply`` here: it would recursively reinitialize
        # the two MiT streams and erase their pretrained parameters.
        for layer in (
                self.linear_c1, self.linear_c2, self.linear_c3,
                self.linear_c4, self.linear_fuse, self.linear_pred):
            layer.apply(_init_cmx_weights)

    def _decode(self, features):
        c1, c2, c3, c4 = features
        target_size = c1.shape[2:]

        def project(feature, layer):
            batch, _, height, width = feature.shape
            projected = layer(feature).transpose([0, 2, 1])
            return projected.reshape([batch, self.embedding_dim, height, width])

        c4 = F.interpolate(
            project(c4, self.linear_c4),
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners)
        c3 = F.interpolate(
            project(c3, self.linear_c3),
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners)
        c2 = F.interpolate(
            project(c2, self.linear_c2),
            size=target_size,
            mode='bilinear',
            align_corners=self.align_corners)
        c1 = project(c1, self.linear_c1)
        return self.linear_pred(self.dropout(
            self.linear_fuse(paddle.concat([c4, c3, c2, c1], axis=1))))

    def forward(self, t1, t2):
        logits = self._decode(self.backbone(t1, t2))
        logits = F.interpolate(
            logits,
            size=paddle.shape(t1)[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return [logits]
