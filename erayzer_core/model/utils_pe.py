import torch
import math
import numpy as np


def get_1d_sincos_pos_emb_from_grid(embed_dim, pos, device="cpu"):
    """
    Generate 1D sinusoidal positional embeddings from grid positions.
    
    Args:
        embed_dim (int): The embedding dimension (must be even).
        pos (torch.Tensor): The grid positions (e.g., [0, 1, 2, ..., v-1]).
                           Shape: [b * gh * gw] or [batch_size, sequence_length].
        device (str): Device for the output tensor.
    
    Returns:
        torch.Tensor: Sinusoidal positional embeddings.
                      Shape: [len(pos), embed_dim]
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even for sine and cosine."
    
    # Convert positions to float
    pos = pos.float()
    
    # Compute the sinusoidal frequencies
    dim = torch.arange(embed_dim // 2, dtype=torch.float32, device=device)  # [0, 1, ..., embed_dim // 2 - 1]
    freq = 1.0 / (10000 ** (dim / (embed_dim // 2)))  # Scale frequencies logarithmically
    
    # Calculate sine and cosine embeddings
    pos_emb_sin = torch.sin(pos[:, None] * freq)  # Shape: [len(pos), embed_dim // 2]
    pos_emb_cos = torch.cos(pos[:, None] * freq)  # Shape: [len(pos), embed_dim // 2]
    
    # Concatenate sine and cosine along the last dimension
    pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=-1)  # Shape: [len(pos), embed_dim]
    
    return pos_emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, device="cpu"):
    """
    Generate 2D sine-cosine positional embeddings with separate grid height and width.
    
    Args:
        embed_dim (int): The embedding dimension.
        grid_size (tuple): Tuple specifying the grid height and width (grid_h, grid_w).
        cls_token (bool): Whether to include a [CLS] token embedding.
        device (str): The device to place the embeddings on.
    
    Returns:
        torch.Tensor: Positional embeddings of shape [grid_h*grid_w, embed_dim]
    """
    grid_h, grid_w = grid_size  # Unpack grid dimensions

    # Create the grid for height and width
    grid_h = torch.arange(grid_h, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_w, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")  # w goes first
    grid = torch.stack(grid, dim=0)  # Shape: [2, grid_h, grid_w]

    # Reshape grid to [2, 1, grid_h, grid_w]
    grid = grid.view(2, 1, grid.size(1), grid.size(2))

    # Get the positional embeddings from the grid
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device=device)  # Shape: [grid_h*grid_w, embed_dim]

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device="cpu"):
    """
    Generate 2D sine-cosine positional embeddings from a grid with separate height and width.
    
    Args:
        embed_dim (int): The embedding dimension.
        grid (torch.Tensor): The grid of shape [2, 1, grid_h, grid_w].
        device (str): The device to place the embeddings on.
    
    Returns:
        torch.Tensor: Positional embeddings of shape [grid_h*grid_w, embed_dim].
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    # Use half of the dimensions for each grid dimension
    grid_h = grid[0].view(-1)  # Flatten grid height dimension: [H*W]
    grid_w = grid[1].view(-1)  # Flatten grid width dimension: [H*W]

    # Generate 1D sine-cosine embeddings for grid_h and grid_w
    emb_h = get_1d_sincos_pos_emb_from_grid(embed_dim // 2, grid_h, device=device)  # Shape: [H*W, D/2]
    emb_w = get_1d_sincos_pos_emb_from_grid(embed_dim // 2, grid_w, device=device)  # Shape: [H*W, D/2]

    # Concatenate along the last dimension
    pos_embed = torch.cat([emb_h, emb_w], dim=-1)  # Shape: [H*W, D]
    return pos_embed
