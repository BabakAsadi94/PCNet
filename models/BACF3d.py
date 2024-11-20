import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        """
        Initializes the PositionalEncoding module.
        
        Args:
            d_model (int): Embedding dimension.
            max_len (int, optional): Maximum length of input sequences. Default is 10000.
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model/2]
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Register as buffer to avoid updating during training

    def forward(self, x):
        """
        Adds positional encoding to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, D]
        
        Returns:
            torch.Tensor: Positionally encoded tensor of shape [B, N, D]
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)  # [B, N, D]
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, cnn_channels=None, ffn_hidden_dim=None, activation='gelu', dropout_prob=0.1):
        """
        Initializes the Bidirectional Cross-Attention Fusion (BCAF) module with Positional Encoding and FNN.
        
        Args:
            embed_dim (int): The embedding dimension (D) of the input features.
            num_heads (int, optional): The number of attention heads. Default is 8.
            cnn_channels (int, optional): Number of CNN channels (C). If provided and C != D, projections are applied.
            ffn_hidden_dim (int, optional): The hidden dimension of the FNN. Defaults to 4 * embed_dim if None.
            activation (str, optional): Activation function ('gelu' or 'relu'). Default is 'gelu'.
            dropout_prob (float, optional): Dropout probability. Default is 0.1.
        """
        super(CrossAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Initialize Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=embed_dim)

        # Pre-attention LayerNorms
        self.pre_norm_C = nn.LayerNorm(embed_dim)
        self.pre_norm_T = nn.LayerNorm(embed_dim)

        # MultiheadAttention modules for both cross-attention directions
        self.attn_C_to_T = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_prob, batch_first=True)
        self.attn_T_to_C = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_prob, batch_first=True)

        # Layer Normalization layers for both cross-attention outputs
        self.norm_C_to_T = nn.LayerNorm(embed_dim)
        self.norm_T_to_C = nn.LayerNorm(embed_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_proj = nn.Dropout(dropout_prob)

        # Final Layer Normalization
        self.norm_output = nn.LayerNorm(embed_dim)

        # Feedforward Neural Network
        ffn_hidden_dim = ffn_hidden_dim if ffn_hidden_dim is not None else 4 * embed_dim
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU() if activation.lower() == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(ffn_hidden_dim, embed_dim),
            nn.Dropout(dropout_prob)
        )

        # Layer Normalization after FNN
        self.norm_ffn = nn.LayerNorm(embed_dim)

        # Optional: Linear layers to align dimensions if C != D
        if cnn_channels is not None and cnn_channels != embed_dim:
            self.cnn_to_transformer_proj = nn.Linear(cnn_channels, embed_dim)
        else:
            self.cnn_to_transformer_proj = None

    def forward(self, X_cnn, X_transformer):
        """
        Forward pass of the BCAF module.
        
        Args:
            X_cnn (torch.Tensor): CNN feature maps. Shape: [B, C, H, W]
            X_transformer (torch.Tensor): Transformer feature embeddings. Shape: [B, N, D]
        
        Returns:
            torch.Tensor: Fused feature embeddings. Shape: [B, N, D]
        """
        B, C, H, W = X_cnn.size()
        N = H * W
        D = self.embed_dim

        # ---------------------------
        # Reshape CNN Features to [B, N, C]
        # ---------------------------
        X_cnn_flat = X_cnn.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # If C != D, project CNN features to Transformer embedding dimension
        if self.cnn_to_transformer_proj is not None:
            X_cnn_flat = self.cnn_to_transformer_proj(X_cnn_flat)  # [B, N, D]

        # ---------------------------
        # Apply Positional Encoding
        # ---------------------------
        X_cnn_flat = self.positional_encoding(X_cnn_flat)  # [B, N, D]
        X_transformer = self.positional_encoding(X_transformer)  # [B, N, D]

        # ---------------------------
        # Pre-Attention LayerNorm
        # ---------------------------
        X_cnn_norm = self.pre_norm_C(X_cnn_flat)  # [B, N, D]
        X_transformer_norm = self.pre_norm_T(X_transformer)  # [B, N, D]

        # ---------------------------
        # Local-to-Global Attention (C → T)
        # ---------------------------
        # Query: X_cnn_norm, Key & Value: X_transformer_norm
        attn_output_C_to_T, _ = self.attn_C_to_T(X_cnn_norm, X_transformer_norm, X_transformer_norm)  # [B, N, D]
        attn_output_C_to_T = self.dropout(attn_output_C_to_T)
        # Residual Connection + Layer Normalization
        X_cnn_fused = self.norm_C_to_T(X_cnn_flat + attn_output_C_to_T)  # [B, N, D]

        # ---------------------------
        # Global-to-Local Attention (T → C)
        # ---------------------------
        # Query: X_transformer_norm, Key & Value: X_cnn_norm
        attn_output_T_to_C, _ = self.attn_T_to_C(X_transformer_norm, X_cnn_norm, X_cnn_norm)  # [B, N, D]
        attn_output_T_to_C = self.dropout(attn_output_T_to_C)
        # Residual Connection + Layer Normalization
        X_transformer_fused = self.norm_T_to_C(X_transformer + attn_output_T_to_C)  # [B, N, D]

        # ---------------------------
        # Fuse the Outputs from Both Attentions
        # ---------------------------
        # Summing the fused CNN and Transformer features
        fused_output = self.dropout(X_cnn_fused + X_transformer_fused)  # [B, N, D]

        # ---------------------------
        # Final Output Projection
        # ---------------------------
        output = self.output_proj(fused_output)  # [B, N, D]
        output = self.dropout_proj(output)

        # Residual Connection + Layer Normalization
        # Adding the fused output to the projected output
        output = self.norm_output(output + fused_output)  # [B, N, D]

        # ---------------------------
        # Feedforward Neural Network (FNN)
        # ---------------------------
        ffn_output = self.ffn(output)  # [B, N, D]

        # Residual Connection + Layer Normalization
        # Adding the FNN output to the output before FNN
        output = self.norm_ffn(ffn_output + output)  # [B, N, D]

        return output
