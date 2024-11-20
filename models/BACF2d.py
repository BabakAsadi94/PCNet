# models/fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionFusion, self).__init__()
        self.dim = dim
        self.WQ_conv = nn.Linear(dim, dim)
        self.WK_conv = nn.Linear(dim, dim)
        self.WV_conv = nn.Linear(dim, dim)
        
        self.WQ_trans = nn.Linear(dim, dim)
        self.WK_trans = nn.Linear(dim, dim)
        self.WV_trans = nn.Linear(dim, dim)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, X_conv, X_trans):
        # X_conv and X_trans have shape [batch_size, dim]
        
        # Reshape to [batch_size, seq_len=1, dim]
        X_conv = X_conv.unsqueeze(1)  # [batch_size, 1, dim]
        X_trans = X_trans.unsqueeze(1)  # [batch_size, 1, dim]
        
        # For X1 (local features embedded with global information)
        Q_conv = self.WQ_conv(X_conv)  # [batch_size, 1, dim]
        K_trans = self.WK_conv(X_trans)  # [batch_size, 1, dim]
        V_conv = self.WV_conv(X_conv)  # [batch_size, 1, dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q_conv, K_trans.transpose(-2, -1)) / (self.dim ** 0.5)  # [batch_size, 1, 1]
        attention_weights = self.softmax(attention_scores)  # [batch_size, 1, 1]
        # Multiply by V_conv
        X1 = attention_weights * V_conv  # [batch_size, 1, dim]
        X1 = X1.squeeze(1)  # [batch_size, dim]
        
        # For X2 (global features embedded with local information)
        Q_trans = self.WQ_trans(X_trans)  # [batch_size, 1, dim]
        K_conv = self.WK_trans(X_conv)  # [batch_size, 1, dim]
        V_trans = self.WV_trans(X_trans)  # [batch_size, 1, dim]
        
        attention_scores_2 = torch.matmul(Q_trans, K_conv.transpose(-2, -1)) / (self.dim ** 0.5)  # [batch_size, 1, 1]
        attention_weights_2 = self.softmax(attention_scores_2)  # [batch_size, 1, 1]
        X2 = attention_weights_2 * V_trans  # [batch_size, 1, dim]
        X2 = X2.squeeze(1)  # [batch_size, dim]
        
        # Final fused feature
        X_fused = X1 + X2  # [batch_size, dim]
        
        return X_fused
