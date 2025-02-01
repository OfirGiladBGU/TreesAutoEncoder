import torch
import torch.nn as nn
from einops import rearrange


# Self-Attention Block
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)  # Split into Q, K, V
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.dropout(self.proj(out))


# Transformer Block (Self-Attention + MLP)
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Vision Transformer that works on ANY image size
class DynamicViT(nn.Module):
    def __init__(self, patch_size=4, num_classes=10, dim=64, depth=3, heads=4, mlp_dim=128):
        super().__init__()

        self.patch_size = patch_size
        self.dim = dim

        # Patch Embedding (Linear Projection)
        self.patch_embedding = nn.Linear(patch_size * patch_size * 3, dim)

        # Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer Encoder
        self.transformer = nn.Sequential(*[TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)])

        # Classification Head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        # Compute Number of Patches (Dynamic)
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # Convert Image to Patches
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)

        # Patch Embedding
        x = self.patch_embedding(x)

        # Add Class Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, dim)

        # Create Positional Embeddings Dynamically
        pos_embed = torch.nn.Parameter(torch.randn(1, num_patches + 1, self.dim)).to(x.device)
        x += pos_embed

        # Transformer Encoder
        x = self.transformer(x)

        # Take CLS Token Output
        x = self.norm(x[:, 0])
        return self.head(x)
