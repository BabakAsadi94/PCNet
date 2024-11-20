# models/dual_branch.py

import torch
import torch.nn as nn
import timm

from .cnn import LDCFeatureExtractor
from .fusion import CrossAttentionFusion

class AsphaltNetDualBranch(nn.Module):
    def __init__(self, num_classes=1, feature_dim_cnn=768, feature_dim_transformer=1536, common_dim=512):
        """
        Args:
            num_classes (int): Number of output classes. For regression, typically 1.
            feature_dim_cnn (int): Output feature dimension from CNN branch.
            feature_dim_transformer (int): Output feature dimension from Transformer branch.
            common_dim (int): Common dimension to project both features for fusion.
        """
        super(AsphaltNetDualBranch, self).__init__()
        
        # Branch 1: Swin Transformer
        self.feature_extractor_transformer = timm.create_model(
            'swin_large_patch4_window12_384', 
            pretrained=True, 
            num_classes=0
        )
        
        # Branch 2: Custom CNN (LDCFeatureExtractor)
        self.feature_extractor_cnn = LDCFeatureExtractor(feature_dim=feature_dim_cnn)
        
        # Feature dimensions
        self.transformer_feature_dim = feature_dim_transformer  # Example: 1536 for swin_large_patch4_window12_384
        self.cnn_feature_dim = feature_dim_cnn  # 768 as per LDCFeatureExtractor
        
        # Project features to the same common dimension
        self.transformer_projection = nn.Linear(self.transformer_feature_dim, common_dim)
        self.cnn_projection = nn.Linear(self.cnn_feature_dim, common_dim)
        
        # Cross-Attention Fusion module
        self.caf_module = CrossAttentionFusion(dim=common_dim)
        
        # Regressor to predict Ductility from concatenated features
        self.regressor = nn.Sequential(
            nn.Linear(common_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights in the regressor
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x_bottom, x_top):
        """
        Args:
            x_bottom (Tensor): Bottom images [B, 3, H, W]
            x_top (Tensor): Top images [B, 3, H, W]
        
        Returns:
            Tensor: Predicted Ductility [B, 1]
        """
        # Process bottom images through Transformer
        features_bottom_transformer = self.feature_extractor_transformer(x_bottom)  # [B, transformer_feature_dim]
        
        # Process bottom images through CNN
        features_bottom_cnn = self.feature_extractor_cnn(x_bottom)  # [B, cnn_feature_dim]
        
        # Process top images through Transformer
        features_top_transformer = self.feature_extractor_transformer(x_top)  # [B, transformer_feature_dim]
        
        # Process top images through CNN
        features_top_cnn = self.feature_extractor_cnn(x_top)  # [B, cnn_feature_dim]
        
        # Project features to common dimension
        features_bottom_transformer = self.transformer_projection(features_bottom_transformer)  # [B, common_dim]
        features_bottom_cnn = self.cnn_projection(features_bottom_cnn)  # [B, common_dim]
        features_top_transformer = self.transformer_projection(features_top_transformer)  # [B, common_dim]
        features_top_cnn = self.cnn_projection(features_top_cnn)  # [B, common_dim]
        
        # Apply CAF module for bottom images
        fused_features_bottom = self.caf_module(features_bottom_cnn, features_bottom_transformer)  # [B, common_dim]
        
        # Apply CAF module for top images
        fused_features_top = self.caf_module(features_top_cnn, features_top_transformer)  # [B, common_dim]
        
        # Concatenate fused features from bottom and top images
        fused_features = torch.cat((fused_features_bottom, fused_features_top), dim=1)  # [B, common_dim * 2]
        
        # Pass concatenated features through the regressor
        output = self.regressor(fused_features)  # [B, num_classes]
        
        return output.view(-1, num_classes)  # [B, num_classes]
