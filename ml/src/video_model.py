"""
MultimodalVideoProctorNet: 36-Layer, ~45M Parameter Multimodal Video Vision Transformer.
Processes 16-frame 112x112 RGB video and 64x300 Mel-spectrogram audio streams.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ==============================================================================
# 1. Spatiotemporal R(2+1)D Video Backbone Components (17 Conv Layers)
# ==============================================================================

class Conv2Plus1D(nn.Module):
    """
    Decomposes a 3D spatio-temporal convolution (kernel: t x d x d) into:
    1. Spatial Conv: (1 x d x d)
    2. Temporal Conv: (t x 1 x 1)
    """
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        mid_planes: Optional[int] = None,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()
        if mid_planes is None:
            # Standard mid-planes formula from Tran et al. CVPR 2018
            mid_planes = int(
                (1 * 3 * 3 * in_planes * out_planes)
                / (in_planes * 3 * 3 + 3 * out_planes)
            )

        # 1. Spatial convolution (1 x 3 x 3)
        self.spatial_conv = nn.Conv3d(
            in_planes,
            mid_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride[1], stride[2]),
            padding=(0, padding[1], padding[2]),
            bias=False,
        )
        self.bn_s = nn.BatchNorm3d(mid_planes)
        self.relu_s = nn.ReLU(inplace=False)

        # 2. Temporal convolution (3 x 1 x 1)
        self.temporal_conv = nn.Conv3d(
            mid_planes,
            out_planes,
            kernel_size=(3, 1, 1),
            stride=(stride[0], 1, 1),
            padding=(padding[0], 0, 0),
            bias=False,
        )
        self.bn_t = nn.BatchNorm3d(out_planes)
        self.relu_t = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_s(self.bn_s(self.spatial_conv(x)))
        x = self.relu_t(self.bn_t(self.temporal_conv(x)))
        return x


class R2Plus1DBlock(nn.Module):
    """Residual block containing two R(2+1)D decomposed convolutions."""
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = Conv2Plus1D(in_planes, planes, stride=stride)
        self.conv2 = Conv2Plus1D(planes, planes, stride=(1, 1, 1))
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.relu(out)


class VideoR2Plus1DBackbone(nn.Module):
    """
    17-Layer R(2+1)D ResNet Backbone for 16-frame 112x112 video.
    Stem (1 spatial + 1 temporal conv) + 4 Stages (each 2 blocks * 2 convs) = 17 Conv layers.
    Outputs: [B, 16, 512] spatiotemporal tokens.
    """
    def __init__(self):
        super().__init__()
        self.in_planes = 64

        # Stem: decomposed 7x7 spatial and 3 temporal (1 Conv layer composite)
        self.stem_spatial = nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.stem_bn_s = nn.BatchNorm3d(45)
        self.stem_relu_s = nn.ReLU(inplace=False)
        self.stem_temporal = nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.stem_bn_t = nn.BatchNorm3d(64)
        self.stem_relu_t = nn.ReLU(inplace=False)

        # 4 Residual Stages: 2 blocks each
        self.stage1 = self._make_stage(64, num_blocks=2, stride=(1, 1, 1))   # 4 convs
        self.stage2 = self._make_stage(128, num_blocks=2, stride=(1, 2, 2))  # 4 convs
        self.stage3 = self._make_stage(256, num_blocks=2, stride=(1, 2, 2))  # 4 convs
        self.stage4 = self._make_stage(512, num_blocks=2, stride=(1, 2, 2))  # 4 convs
        # Total conv layers = 1 stem + 4*4 = 17 conv layers

        # Hook for Grad-CAM
        self.last_conv_features = None

    def _make_stage(self, planes: int, num_blocks: int, stride: Tuple[int, int, int]) -> nn.Sequential:
        downsample = None
        if stride != (1, 1, 1) or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

        layers = [R2Plus1DBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(R2Plus1DBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: (B, 3, 16, 112, 112)
        Returns:
            tokens: (B, 16, 512)
            feature_map: (B, 512, 16, H, W) for Grad-CAM
        """
        x = self.stem_relu_s(self.stem_bn_s(self.stem_spatial(x)))
        x = self.stem_relu_t(self.stem_bn_t(self.stem_temporal(x)))

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feature_map = self.stage4(x)  # (B, 512, 16, H', W')
        self.last_conv_features = feature_map

        # Spatial average pooling over (H, W) -> (B, 512, 16)
        pooled = torch.mean(feature_map, dim=(-2, -1))
        tokens = pooled.transpose(1, 2)  # (B, 16, 512)
        return tokens, feature_map


# ==============================================================================
# 2. Audio 2D-CNN Backbone (4 Layers)
# ==============================================================================

class Audio2DCNN(nn.Module):
    """
    4-Layer Conv2D audio feature extractor for 64-band Log-Mel Spectrograms [B, 1, 64, 300].
    Extracts 16 temporal audio tokens with dimension 256: [B, 16, 256].
    """
    def __init__(self):
        super().__init__()
        # Layer 1: 1 -> 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.GELU()

        # Layer 2: 64 -> 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.GELU()

        # Layer 3: 128 -> 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.GELU()

        # Layer 4: 256 -> 256
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 1, 64, 300)
        Returns: (B, 16, 256)
        """
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))

        # Average pool over frequency dimension: (B, 256, T)
        x_freq = torch.mean(x, dim=2)
        # Interpolate temporal dimension to 16 tokens: (B, 256, 16)
        pooled = F.interpolate(x_freq, size=16, mode='linear', align_corners=False)
        tokens = pooled.transpose(1, 2)  # (B, 16, 256)
        return tokens


# ==============================================================================
# 3. Deep Cross-Modal Transformer Backbone (12 Hidden Layers)
# ==============================================================================

class CrossModalTransformerBlock(nn.Module):
    """
    Bidirectional Cross-Modal Transformer Block (Pre-LayerNorm, GELU).
    Audio cross-attends to Video (gaze/mouth dynamics),
    and Video cross-attends to Audio (speech/silence rhythm).
    Each block provides 2 sub-layers (Cross-Attention + Feed-Forward Network).
    """
    def __init__(self, d_model: int = 384, num_heads: int = 8, ffn_dim: int = 1536, dropout: float = 0.1):
        super().__init__()
        # Pre-LayerNorms
        self.ln_v1 = nn.LayerNorm(d_model)
        self.ln_a1 = nn.LayerNorm(d_model)
        self.ln_v2 = nn.LayerNorm(d_model)
        self.ln_a2 = nn.LayerNorm(d_model)

        # Cross-Attention: Video queries Audio
        self.cross_v_to_a = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Cross-Attention: Audio queries Video
        self.cross_a_to_v = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Feed-Forward Networks with GELU activation
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, v: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        v: (B, 16, 384) video tokens
        a: (B, 16, 384) audio tokens
        """
        # Pre-LN Cross Attention
        v_norm = self.ln_v1(v)
        a_norm = self.ln_a1(a)

        attn_v, _ = self.cross_v_to_a(query=v_norm, key=a_norm, value=a_norm)
        attn_a, _ = self.cross_a_to_v(query=a_norm, key=v_norm, value=v_norm)

        v = v + attn_v
        a = a + attn_a

        # Pre-LN FFN
        v = v + self.ffn_v(self.ln_v2(v))
        a = a + self.ffn_a(self.ln_a2(a))

        return v, a


# ==============================================================================
# 4. MultimodalVideoProctorNet (Full 36-Layer Architecture)
# ==============================================================================

class MultimodalVideoProctorNet(nn.Module):
    """
    36-Layer, ~45-Million Parameter Multimodal Video Vision Transformer.
    Layer breakdown:
        • 17 Video Conv Layers (R(2+1)D ResNet-18)
        • 4 Audio Conv Layers (2D-CNN)
        • 2 Linear Modality Projections (512->384, 256->384)
        • 12 Transformer Hidden Layers (6 Bidirectional Cross-Modal Blocks * 2 sublayers)
        • 1 Classification Hidden Layer (Linear 768->256)
        Total Hidden Layers = 36 layers.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # 1. Video Backbone (17 Conv layers)
        self.video_backbone = VideoR2Plus1DBackbone()

        # 2. Audio Backbone (4 Conv layers)
        self.audio_backbone = Audio2DCNN()

        # 3. Modality Alignment Projections (2 layers)
        self.video_proj = nn.Linear(512, 384)
        self.audio_proj = nn.Linear(256, 384)

        # 4. Temporal Positional Encodings
        self.video_pos_emb = nn.Parameter(torch.randn(1, 16, 384) * 0.02)
        self.audio_pos_emb = nn.Parameter(torch.randn(1, 16, 384) * 0.02)

        # 5. Deep Cross-Modal Transformer (6 Blocks = 12 Layers)
        self.transformer_blocks = nn.ModuleList([
            CrossModalTransformerBlock(d_model=384, num_heads=8, ffn_dim=1536, dropout=0.1)
            for _ in range(6)
        ])
        self.final_norm = nn.LayerNorm(384)

        # 6. Classification Head (1 Dense Hidden Layer + Output Layer)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(
        self, video: torch.Tensor, audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video: (B, 16, 3, 112, 112) Normalized candidate video frames in [-1.0, 1.0]
            audio: (B, 1, 64, 300) Log-Mel Spectrogram of candidate audio
        Returns:
            logits: (B, 4) Unnormalized class scores
        """
        # Shape preparation: Conv3D expects (B, C, T, H, W)
        if video.dim() == 5 and video.size(2) == 3:
            # Transpose (B, T, C, H, W) -> (B, C, T, H, W)
            video = video.permute(0, 2, 1, 3, 4).contiguous()

        # 1. Feature extraction
        v_tokens, _ = self.video_backbone(video)  # (B, 16, 512)
        a_tokens = self.audio_backbone(audio)     # (B, 16, 256)

        # 2. Modality projection
        v = self.video_proj(v_tokens) + self.video_pos_emb  # (B, 16, 384)
        a = self.audio_proj(a_tokens) + self.audio_pos_emb  # (B, 16, 384)

        # 3. Cross-modal transformer blocks
        for block in self.transformer_blocks:
            v, a = block(v, a)

        # Fuse fused multimodal tokens (video aware of audio, audio aware of video)
        fused = self.final_norm((v + a) * 0.5)  # (B, 16, 384)

        # 4. Global Temporal Pooling (AvgPool + MaxPool = 768-dim)
        avg_rep = torch.mean(fused, dim=1)           # (B, 384)
        max_rep = torch.max(fused, dim=1).values     # (B, 384)
        rep = torch.cat([avg_rep, max_rep], dim=-1)  # (B, 768)

        # 5. Classification head
        logits = self.classifier(rep)  # (B, 4)
        return logits


def get_model(num_classes: int = 4) -> MultimodalVideoProctorNet:
    """Helper to instantiate MultimodalVideoProctorNet."""
    return MultimodalVideoProctorNet(num_classes=num_classes)


if __name__ == "__main__":
    model = get_model()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MultimodalVideoProctorNet initialized successfully.")
    print(f"Total Trainable Parameters: {num_params:,} (~{num_params / 1e6:.1f}M)")

    # Test forward pass with dummy inputs
    dummy_video = torch.randn(2, 16, 3, 112, 112)
    dummy_audio = torch.randn(2, 1, 64, 300)
    out = model(dummy_video, dummy_audio)
    print(f"Forward pass output shape: {out.shape}")
