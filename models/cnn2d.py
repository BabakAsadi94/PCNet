# models/cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the weight initialization function
def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, PixelDifferenceConvolution):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)

# Implement the Pixel Difference Convolution (PDC) layer
class PixelDifferenceConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(PixelDifferenceConvolution, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_neighbors = kernel_size * kernel_size - 1
        # Learnable weights for differences
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels * self.num_neighbors))
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        # Bias term
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        B, C, H, W = x.size()
        # Pad the input
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        # Unfold to get local patches
        patches = F.unfold(x_padded, kernel_size=self.kernel_size)  # [B, C * K*K, L]
        L = patches.size(2)
        # Reshape to [B, C, K*K, L]
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, L)
        # Separate central pixel and neighbors
        center_idx = self.kernel_size * self.kernel_size // 2
        center_pixel = patches[:, :, center_idx:center_idx+1, :]  # [B, C, 1, L]
        neighbors = torch.cat([patches[:, :, :center_idx, :], patches[:, :, center_idx+1:, :]], dim=2)  # [B, C, K*K-1, L]
        # Compute differences
        diffs = neighbors - center_pixel  # [B, C, K*K-1, L]
        # Reshape diffs to [B, C * self.num_neighbors, L]
        diffs = diffs.view(B, C * self.num_neighbors, L)
        # Apply weights
        out = torch.matmul(self.weight, diffs)  # [B, out_channels, L]
        out = out + self.bias.view(1, -1, 1)
        # Reshape back to [B, out_channels, H_out, W_out]
        H_out = int((H + 2 * self.padding - self.kernel_size) / 1 + 1)
        W_out = int((W + 2 * self.padding - self.kernel_size) / 1 + 1)
        out = out.view(B, self.out_channels, H_out, W_out)
        return out

# Modified _DenseLayer class with optional PDC
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation=1, use_pdc=False):
        super(_DenseLayer, self).__init__()
        self.use_pdc = use_pdc
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        if self.use_pdc:
            self.conv1 = PixelDifferenceConvolution(in_channels, growth_rate, kernel_size=3, padding=dilation)
        else:
            self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=dilation,
                                   bias=False, dilation=dilation)

    def forward(self, x):
        new_features = self.norm1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        # Ensure the input and output spatial dimensions match before concatenation
        if new_features.size(2) != x.size(2) or new_features.size(3) != x.size(3):
            new_features = F.interpolate(new_features, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, dilation=1, pdc_layers=None):
        super(_DenseBlock, self).__init__()
        if pdc_layers is None:
            pdc_layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                dilation=dilation,
                use_pdc=(i in pdc_layers)
            )
            self.add_module('denselayer%d' % (i + 1), layer)

class TransitionLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

# Attentional Feature Pyramid Network Block (used)
class AttentionalFPNBlock(nn.Module):
    def __init__(self, feature_channels, out_channels=128):  # Reduced out_channels to 128
        super(AttentionalFPNBlock, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        for in_channels in feature_channels:
            # Lateral convolution
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.lateral_convs.append(lateral_conv)
            # Attention block
            attention_block = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
            self.attention_blocks.append(attention_block)

    def forward(self, feature_maps):
        P = []
        last_inner = None
        for idx in reversed(range(len(feature_maps))):
            lateral_feat = self.lateral_convs[idx](feature_maps[idx])
            if last_inner is None:
                inner_top_down = lateral_feat
            else:
                inner_top_down = lateral_feat + F.interpolate(
                    last_inner, size=lateral_feat.shape[2:], mode='bilinear', align_corners=False)
            # Apply attention
            attention = self.attention_blocks[idx](inner_top_down)
            inner_top_down = inner_top_down * attention
            last_inner = inner_top_down
            P.append(last_inner)
        P = P[::-1]
        return P

# Improved SEFusion Block for Local Feature Enhancement
class SEFusion(nn.Module):
    def __init__(self, in_channels, feature_dim=768, reduction=16):  # Set feature_dim to 768
        super(SEFusion, self).__init__()
        # Spatial attention mechanism for edge, contour, and boundary detection
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        # Edge enhancement layer
        self.edge_enhance = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        nn.init.xavier_normal_(self.edge_enhance.weight)
        
        # Reduce number of channels before passing to fully connected layer
        reduced_dim = in_channels // 4
        self.reduce_conv = nn.Conv2d(in_channels, reduced_dim, kernel_size=1, bias=False)
        
        # Final transformation to produce feature vector
        self.final_fc = nn.Linear(reduced_dim, feature_dim, bias=False)
        self.bn_final = nn.BatchNorm1d(feature_dim)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C, H, W]
        batch_size, num_channels, _, _ = x.size()
        
        # Spatial attention
        s = self.spatial_attention(x)  # [B, 1, H, W]
        x = x * s  # Apply spatial attention to focus on edges, contours, and boundaries
        
        # Edge enhancement
        x = self.edge_enhance(x)  # Further enhance edges and boundaries
        
        # Reduce number of channels
        x = self.reduce_conv(x)  # [B, reduced_dim, H, W]
        
        # Apply global average pooling to get the final feature vector
        feature_vector = F.adaptive_avg_pool2d(x, 1).view(batch_size, -1)
        feature_vector = self.final_fc(feature_vector)
        feature_vector = self.bn_final(feature_vector)
        feature_vector = self.relu_final(feature_vector)
        
        return feature_vector  # [B, feature_dim]

class LDCFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=768):  # Set feature_dim to 768
        super(LDCFeatureExtractor, self).__init__()
        # Initial convolution with PDC
        self.conv0 = PixelDifferenceConvolution(3, 16, kernel_size=3, padding=1)  # Reduced filters from 32 to 16
        self.bn0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU(inplace=True)
        # No initial pooling to preserve spatial resolution

        # Block configurations
        growth_rate = 8  # Reduced growth rate from 16 to 8

        # Block 1: 2 layers, 1 PDC
        num_layers1 = 2
        self.block1 = _DenseBlock(num_layers1, 16, growth_rate, dilation=1, pdc_layers=[0])
        num_features1 = 16 + num_layers1 * growth_rate  # 16 + 2*8 = 32
        self.trans1 = TransitionLayer(num_features1, 32)

        # Block 2: 2 layers, 1 PDC
        num_layers2 = 2
        self.block2 = _DenseBlock(num_layers2, 32, growth_rate, dilation=2, pdc_layers=[0])
        num_features2 = 32 + num_layers2 * growth_rate  # 32 + 2*8 = 48
        self.trans2 = TransitionLayer(num_features2, 64)

        # Block 3: 3 layers, 1 PDC
        num_layers3 = 3
        self.block3 = _DenseBlock(num_layers3, 64, growth_rate, dilation=4, pdc_layers=[0])
        num_features3 = 64 + num_layers3 * growth_rate  # 64 + 3*8 = 88
        self.trans3 = TransitionLayer(num_features3, 128)

        # Block 4: 4 layers, 1 PDC, 2 vanilla conv
        num_layers4 = 4
        self.block4 = _DenseBlock(num_layers4, 128, growth_rate, dilation=2, pdc_layers=[0])
        num_features4 = 128 + num_layers4 * growth_rate  # 128 + 4*8 = 160

        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Conv2d(num_features1, 48, kernel_size=1, bias=False),  # Match channels with c2
            nn.Conv2d(num_features2, 88, kernel_size=1, bias=False),  # Match channels with c3
        ])

        # Attentional FPN
        self.attentional_fpn = AttentionalFPNBlock(
            feature_channels=[
                num_features1,  # Output channels from block1
                num_features2,  # Output channels from block2
                num_features3,  # Output channels from block3
                num_features4   # Output channels from block4
            ],
            out_channels=128  # Reduced out_channels to 128
        )

        # SEFusion
        self.se_fusion = SEFusion(128 * 4, feature_dim=feature_dim)

        # Initialize weights
        self.apply(weight_init)

    def forward(self, x):
        assert x.ndim == 4, x.shape
        # Initial convolution
        x = self.conv0(x)      # [B, 16, H, W]
        x = self.bn0(x)
        x = self.relu0(x)
        # No initial pooling to preserve spatial resolution

        # Block 1
        c1 = self.block1(x)    # [B, 32, H, W]
        t1 = self.trans1(c1)   # [B, 32, H/2, W/2]

        # Block 2
        c2 = self.block2(t1)   # [B, 48, H/2, W/2]
        # Skip connection from c1
        skip1 = F.interpolate(self.skip_connections[0](c1), size=c2.shape[2:], mode='bilinear', align_corners=False)
        c2 = c2 + skip1
        t2 = self.trans2(c2)   # [B, 64, H/4, W/4]

        # Block 3
        c3 = self.block3(t2)   # [B, 88, H/4, W/4]
        # Skip connection from c2
        skip2 = F.interpolate(self.skip_connections[1](c2), size=c3.shape[2:], mode='bilinear', align_corners=False)
        c3 = c3 + skip2
        t3 = self.trans3(c3)   # [B, 128, H/8, W/8]

        # Block 4
        c4 = self.block4(t3)   # [B, 160, H/8, W/8]

        # Collect feature maps for FPN
        features = [c1, c2, c3, c4]

        # Attentional FPN
        fpn_features = self.attentional_fpn(features)  # List of feature maps

        # Concatenate feature maps from all levels
        upsampled_features = []
        for feature in fpn_features:
            upsampled = F.interpolate(feature, size=fpn_features[0].shape[2:], mode='bilinear', align_corners=False)
            upsampled_features.append(upsampled)
        fused_feature_map = torch.cat(upsampled_features, dim=1)  # [B, 128*4, H, W]

        # SEFusion for final feature extraction
        feature_vector = self.se_fusion(fused_feature_map)  # [B, feature_dim]

        return feature_vector  # [B, feature_dim]
