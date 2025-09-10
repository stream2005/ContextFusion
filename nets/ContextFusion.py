"""
Multi-Modal Image Fusion Network with Dual-Branch Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers
from typing import Tuple, Optional
from einops import rearrange

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lsnet import Block
from DSwinIR.basicsr.archs.deformable_self_attention import DeformableNeighborhoodAttention

# Global configuration for multi-scale feature extraction
DOWNSAMPLE_STRIDES = [[1, 1], [1, 1]]
ATTENTION_KERNELS = [[7, 7], [5, 5]]


class AdaptiveFeatureBlock(nn.Module):
    """
    Adaptive Feature Extraction Block with configurable processing modes.
    
    This block serves as the core building component that can operate in different modes:
    1. Encoder mode: Local features + Block sequence for hierarchical feature extraction
    2. Single-path fusion mode: Only global attention for focused processing  
    3. Dual-path fusion mode: Combined local + global features for comprehensive fusion
    
    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        stage (int): Processing stage (-1 for fusion mode, >=0 for encoder mode)
        depth (int): Number of blocks in sequence (only for encoder mode)
        kernel_size_idx (int): Index for attention kernel size selection
        fusion_mode (int): Fusion processing mode (1 for single-path, others for dual-path)
    """
    
    def __init__(self, 
                 input_channels: int, 
                 output_channels: int, 
                 stage: int = -1, 
                 depth: int = -1, 
                 kernel_size_idx: int = 0, 
                 fusion_mode: int = 0):
        super(AdaptiveFeatureBlock, self).__init__()
        
        # Feature embedding layer with reflection padding for boundary handling
        self.feature_embedding = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=1, 
            padding=1, bias=False, padding_mode="reflect"
        )
        
        self.stage = stage
        self.fusion_mode = fusion_mode
        
        # Configure processing modules based on stage
        if stage != -1:  # Encoder mode: hierarchical feature extraction
            self._setup_encoder_mode(output_channels, depth, stage)
        else:  # Fusion mode: adaptive attention-based processing
            self._setup_fusion_mode(output_channels, kernel_size_idx, fusion_mode)
            
        # Feature fusion layer for dual-path processing
        self.feature_fusion = nn.Conv2d(
            output_channels * 2, output_channels, kernel_size=3, stride=1, 
            padding=1, bias=False, padding_mode="reflect"
        )
    
    def _setup_encoder_mode(self, channels: int, depth: int, stage: int):
        """Setup components for encoder mode with local features and block sequence."""
        self.local_processor = LocalConvolutionalModule(dim=channels)
        self.block_sequence = nn.Sequential()
        for block_idx in range(depth):
            self.block_sequence.append(
                Block(ed=channels, kd=16, stage=stage, depth=block_idx)
            )
    
    def _setup_fusion_mode(self, channels: int, kernel_size_idx: int, fusion_mode: int):
        """Setup components for fusion mode with configurable attention mechanisms."""
        if fusion_mode == 1:  # Single-path: only global attention
            self.global_processor = GlobalAttentionModule(
                dim=channels, 
                kernel_size=ATTENTION_KERNELS[kernel_size_idx][1],
                num_heads=channels // 4, 
                stage=fusion_mode, 
                stride=DOWNSAMPLE_STRIDES[kernel_size_idx][1]
            )
        else:  # Dual-path: global + local processing
            self.global_processor = GlobalAttentionModule(
                dim=channels, 
                kernel_size=kernel_size_idx,
                num_heads=channels // 4, 
                stage=fusion_mode
            )
            self.local_processor = LocalConvolutionalModule(dim=channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive processing based on configuration.
        
        Args:
            x: Input feature tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Processed features of shape (B, output_channels, H, W)
        """
        # Initial feature embedding
        embedded_features = self.feature_embedding(x)
        
        # Adaptive processing based on fusion mode
        if self.fusion_mode != 1:  # Dual-path processing
            if hasattr(self, 'local_processor'):
                local_features = self.local_processor(embedded_features)
                
                if hasattr(self, 'global_processor'):
                    global_features = self.global_processor(embedded_features)
                else:
                    global_features = self.block_sequence(embedded_features)
                
                # Concatenate and fuse dual-path features
                concatenated_features = torch.cat([local_features, global_features], dim=1)
                return self.feature_fusion(concatenated_features)
            else:
                # Encoder mode: use transformer for local processing
                return self.local_processor(embedded_features)
        else:  # Single-path processing
            return self.global_processor(embedded_features)


class GlobalAttentionModule(nn.Module):
    """
    Global Feature Extraction Module with Self-Attention Mechanisms.
    
    Implements either deformable neighborhood attention or standard multi-head attention
    based on the processing stage, following Transformer architecture with residual connections.
    
    Args:
        dim (int): Feature dimension
        kernel_size (int): Attention kernel size
        num_heads (int): Number of attention heads
        ffn_expansion_factor (float): Expansion factor for feed-forward network
        qkv_bias (bool): Whether to use bias in QKV projection
        stride (int): Stride for attention computation
        stage (int): Processing stage (1 for deformable attention, others for standard)
    """
    
    def __init__(self,
                 dim: int,
                 kernel_size: int,
                 num_heads: int,
                 ffn_expansion_factor: float = 1.,
                 qkv_bias: bool = False,
                 stride: int = 1,
                 stage: int = 1):
        super(GlobalAttentionModule, self).__init__()
        
        # Layer normalization for stable training
        self.norm_attention = LayerNorm(dim, 'WithBias')
        
        # Attention mechanism selection based on stage
        if stage == 1:  # Advanced deformable attention for stage 1
            self.attention = DeformableNeighborhoodAttention(
                dim=dim, 
                num_heads=num_heads, 
                kernel_size=kernel_size,
                offset_range_factor=2.0, 
                stride=stride, 
                use_pe=True, 
                dwc_pe=True, 
                no_off=False
            )
        else:  # Standard multi-head attention for other stages
            self.attention = StandardAttentionModule(dim=dim, num_heads=num_heads)
        
        self.norm_ffn = LayerNorm(dim, 'WithBias')
        self.feed_forward = MultilayerPerceptron(
            in_features=dim, 
            out_features=dim,
            expansion_factor=ffn_expansion_factor
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections and layer normalization.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Processed features with global context
        """
        # Self-attention with residual connection
        x = x + self.attention(self.norm_attention(x))
        # Feed-forward with residual connection
        x = x + self.feed_forward(self.norm_ffn(x))
        return x


class LocalConvolutionalModule(nn.Module):
    """
    Local Feature Extraction Module using Residual Convolutional Blocks.
    
    Extracts local spatial patterns through a sequence of residual blocks,
    preserving fine-grained details essential for image fusion tasks.
    
    Args:
        dim (int): Feature dimension
        num_blocks (int): Number of residual blocks
    """
    
    def __init__(self, dim: int = 64, num_blocks: int = 2):
        super(LocalConvolutionalModule, self).__init__()
        self.feature_extractor = nn.Sequential(*[
            ResidualBlock(dim, dim) for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract local features through residual processing."""
        return self.feature_extractor(x)


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with reflection padding for boundary preservation.
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.convolution_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, 
                     padding=1, bias=True, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                     padding=1, bias=True, padding_mode="reflect"),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.convolution_block(x) + x


class StandardAttentionModule(nn.Module):
    """
    Standard Multi-Head Self-Attention Module.
    
    Implements scaled dot-product attention with multi-head mechanism for
    capturing global dependencies in feature representations.
    
    Args:
        dim (int): Input feature dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in QKV projection
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super(StandardAttentionModule, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # QKV projection with depth-wise convolution for spatial awareness
        self.qkv_projection = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv_enhancement = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, 
                                       padding=1, bias=qkv_bias)
        self.output_projection = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head attention computation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Attention-weighted features
        """
        batch_size, channels, height, width = x.shape
        
        # Generate QKV representations
        qkv = self.qkv_enhancement(self.qkv_projection(x))
        query, key, value = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        query = rearrange(query, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        key = rearrange(key, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        value = rearrange(value, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        # L2 normalization for stable attention computation
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        
        # Scaled dot-product attention
        attention_weights = (query @ key.transpose(-2, -1)) * self.scale
        attention_weights = attention_weights.softmax(dim=-1)
        
        # Apply attention to values
        attended_features = attention_weights @ value
        
        # Reshape back to spatial format
        attended_features = rearrange(
            attended_features, 'b head c (h w) -> b (head c) h w', 
            head=self.num_heads, h=height, w=width
        )
        
        return self.output_projection(attended_features)


class MultilayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron with GELU activation for feed-forward processing.
    
    Implements channel-wise processing with expansion and compression,
    commonly used in Transformer architectures for feature transformation.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        expansion_factor (int): Hidden layer expansion factor
        bias (bool): Whether to use bias in convolutions
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 expansion_factor: int = 2, bias: bool = False):
        super(MultilayerPerceptron, self).__init__()
        hidden_features = int(in_features * expansion_factor)

        # Expansion layer
        self.expansion_layer = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias
        )
        
        # Depth-wise convolution for spatial processing
        self.spatial_processing = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3,
            stride=1, padding=1, groups=hidden_features, bias=bias, 
            padding_mode="reflect"
        )
        
        # Compression layer
        self.compression_layer = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with GELU activation and gating mechanism."""
        x = self.expansion_layer(x)
        x1, x2 = self.spatial_processing(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2  # Gating mechanism with GELU activation
        return self.compression_layer(x)


# Layer Normalization Components
def convert_to_3d(x: torch.Tensor) -> torch.Tensor:
    """Convert 4D tensor to 3D for layer normalization."""
    return rearrange(x, 'b c h w -> b (h w) c')


def convert_to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert 3D tensor back to 4D after layer normalization."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    """Bias-free Layer Normalization for stable training."""
    
    def __init__(self, normalized_shape):
        super(BiasFreeLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class BiasedLayerNorm(nn.Module):
    """Layer Normalization with learnable bias."""
    
    def __init__(self, normalized_shape):
        super(BiasedLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Configurable Layer Normalization wrapper."""
    
    def __init__(self, dim: int, layer_norm_type: str):
        super(LayerNorm, self).__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = BiasedLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        return convert_to_4d(self.body(convert_to_3d(x)), height, width)


class DualBranchFusionNetwork(nn.Module):
    """
    Dual-Branch Encoder-Decoder Network for Multi-Modal Image Fusion.
    
    This network architecture processes infrared and visible images through separate
    encoding branches, fuses multi-scale features, and reconstructs high-quality
    fusion results through a decoder with skip connections.
    
    Architecture Components:
    - Dual-branch encoders for infrared and visible modalities
    - Multi-scale feature fusion modules  
    - U-Net style decoder with skip connections
    - Progressive upsampling using pixel shuffle
    
    Paper Reference: Multi-Modal Image Fusion via Unified Feature Extraction
    """
    
    def __init__(self):
        super(DualBranchFusionNetwork, self).__init__()
        
        # Multi-scale channel configuration for progressive feature extraction
        self.channel_dimensions = [8, 16, 32, 32]
        
        # Infrared image encoding branch
        self.infrared_encoder_L1 = AdaptiveFeatureBlock(1, self.channel_dimensions[0], stage=0, depth=1)
        self.infrared_encoder_L2 = AdaptiveFeatureBlock(self.channel_dimensions[0], self.channel_dimensions[1], stage=1, depth=4)
        self.infrared_encoder_L3 = AdaptiveFeatureBlock(self.channel_dimensions[1], self.channel_dimensions[2], stage=2, depth=8)
        self.infrared_encoder_L4 = AdaptiveFeatureBlock(self.channel_dimensions[2], self.channel_dimensions[3], stage=2, depth=10)

        # Visible image encoding branch  
        self.visible_encoder_L1 = AdaptiveFeatureBlock(1, self.channel_dimensions[0], stage=0, depth=1)
        self.visible_encoder_L2 = AdaptiveFeatureBlock(self.channel_dimensions[0], self.channel_dimensions[1], stage=1, depth=4)
        self.visible_encoder_L3 = AdaptiveFeatureBlock(self.channel_dimensions[1], self.channel_dimensions[2], stage=2, depth=8)
        self.visible_encoder_L4 = AdaptiveFeatureBlock(self.channel_dimensions[2], self.channel_dimensions[3], stage=2, depth=10)

        # Cross-modal feature fusion modules
        self.fusion_L1 = AdaptiveFeatureBlock(self.channel_dimensions[0] * 2, self.channel_dimensions[0])
        self.fusion_L2 = AdaptiveFeatureBlock(self.channel_dimensions[1] * 2, self.channel_dimensions[1])
        self.fusion_L3 = AdaptiveFeatureBlock(self.channel_dimensions[2] * 2, self.channel_dimensions[2])
        self.fusion_L4 = AdaptiveFeatureBlock(self.channel_dimensions[3] * 2, self.channel_dimensions[3])

        # Downsampling layers for multi-scale processing
        self._setup_downsampling_layers()
        
        # Progressive upsampling layers using pixel shuffle
        self._setup_upsampling_layers()
        
        # Decoder modules with skip connections
        self.decoder_L1 = AdaptiveFeatureBlock(self.channel_dimensions[0] * 2, self.channel_dimensions[0], kernel_size_idx=1, fusion_mode=1)
        self.decoder_L2 = AdaptiveFeatureBlock(self.channel_dimensions[1] * 2, self.channel_dimensions[1])
        self.decoder_L3 = AdaptiveFeatureBlock(self.channel_dimensions[2] * 2, self.channel_dimensions[2], kernel_size_idx=0, fusion_mode=1)
        self.decoder_L4 = AdaptiveFeatureBlock(self.channel_dimensions[3], self.channel_dimensions[3])

        # Final reconstruction layer
        self.output_reconstruction = nn.Sequential(
            nn.Conv2d(self.channel_dimensions[0], 1, kernel_size=3, stride=1, 
                     padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )

    def _setup_downsampling_layers(self):
        """Initialize downsampling layers for multi-scale feature extraction."""
        # Visible image downsampling
        self.visible_downsample_L1 = nn.Conv2d(self.channel_dimensions[0], self.channel_dimensions[0], 
                                              kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect")
        self.visible_downsample_L2 = nn.Conv2d(self.channel_dimensions[1], self.channel_dimensions[1], 
                                              kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect")
        self.visible_downsample_L3 = nn.Conv2d(self.channel_dimensions[2], self.channel_dimensions[2], 
                                              kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect")
        
        # Infrared image downsampling  
        self.infrared_downsample_L1 = nn.Conv2d(self.channel_dimensions[0], self.channel_dimensions[0], 
                                               kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect")
        self.infrared_downsample_L2 = nn.Conv2d(self.channel_dimensions[1], self.channel_dimensions[1], 
                                               kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect")
        self.infrared_downsample_L3 = nn.Conv2d(self.channel_dimensions[2], self.channel_dimensions[2], 
                                               kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect")

    def _setup_upsampling_layers(self):
        """Initialize progressive upsampling layers using pixel shuffle."""
        self.upsample_L4 = nn.Sequential(
            nn.Conv2d(self.channel_dimensions[3], self.channel_dimensions[2] * 4, 
                     kernel_size=3, padding=1, padding_mode="reflect"),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.upsample_L3 = nn.Sequential(
            nn.Conv2d(self.channel_dimensions[2], self.channel_dimensions[1] * 4, 
                     kernel_size=3, padding=1, padding_mode="reflect"),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.upsample_L2 = nn.Sequential(
            nn.Conv2d(self.channel_dimensions[1], self.channel_dimensions[0] * 4, 
                     kernel_size=3, padding=1, padding_mode="reflect"),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

    def forward(self, infrared_input: torch.Tensor, visible_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for dual-branch image fusion.
        
        Args:
            infrared_input: Infrared image tensor of shape (B, 1, H, W)
            visible_input: Visible image tensor of shape (B, 1, H, W)
            
        Returns:
            torch.Tensor: Fused image tensor of shape (B, 1, H, W)
        """
        # Multi-scale infrared feature extraction
        infrared_feat_L1 = self.infrared_encoder_L1(infrared_input)
        infrared_feat_L2 = self.infrared_encoder_L2(self.infrared_downsample_L1(infrared_feat_L1))
        infrared_feat_L3 = self.infrared_encoder_L3(self.infrared_downsample_L2(infrared_feat_L2))
        infrared_feat_L4 = self.infrared_encoder_L4(self.infrared_downsample_L3(infrared_feat_L3))

        # Multi-scale visible feature extraction
        visible_feat_L1 = self.visible_encoder_L1(visible_input)
        visible_feat_L2 = self.visible_encoder_L2(self.visible_downsample_L1(visible_feat_L1))
        visible_feat_L3 = self.visible_encoder_L3(self.visible_downsample_L2(visible_feat_L2))
        visible_feat_L4 = self.visible_encoder_L4(self.visible_downsample_L3(visible_feat_L3))

        # Cross-modal feature fusion at multiple scales
        fused_feat_L1 = self.fusion_L1(torch.cat([infrared_feat_L1, visible_feat_L1], dim=1))
        fused_feat_L2 = self.fusion_L2(torch.cat([infrared_feat_L2, visible_feat_L2], dim=1))
        fused_feat_L3 = self.fusion_L3(torch.cat([infrared_feat_L3, visible_feat_L3], dim=1))
        fused_feat_L4 = self.fusion_L4(torch.cat([infrared_feat_L4, visible_feat_L4], dim=1))

        # Progressive decoding with skip connections
        decoded_feat = self.upsample_L4(self.decoder_L4(fused_feat_L4))
        decoded_feat = self.upsample_L3(self.decoder_L3(torch.cat([decoded_feat, fused_feat_L3], dim=1)))
        decoded_feat = self.upsample_L2(self.decoder_L2(torch.cat([decoded_feat, fused_feat_L2], dim=1)))
        decoded_feat = self.decoder_L1(torch.cat([decoded_feat, fused_feat_L1], dim=1))
        
        # Final image reconstruction
        fusion_result = self.output_reconstruction(decoded_feat)
        return fusion_result

ContextFusion = DualBranchFusionNetwork


