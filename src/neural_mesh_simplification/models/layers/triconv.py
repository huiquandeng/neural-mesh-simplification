import torch
import torch.nn as nn
from torch_scatter import scatter_add


class TriConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TriConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Calculate the correct input dimension for the MLP
        mlp_input_dim = in_channels + 9  # 9 is from the relative position encoding

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.last_edge_index = None

    def forward(self, x, pos, edge_index):
        self.last_edge_index = edge_index
        row, col = edge_index

        rel_pos = self.compute_relative_position_encoding(pos, row, col)
        x_diff = x[row] - x[col]
        mlp_input = torch.cat([rel_pos, x_diff], dim=-1)

        mlp_output = self.mlp(mlp_input)
        out = scatter_add(mlp_output, col, dim=0, dim_size=x.size(0))

        return out

    def compute_relative_position_encoding(self, pos, row, col):
        """Relative position encoding for an edge (i, j) between two triangle
        barycenters.

        `pos` holds the triangle barycenters [num_faces, 3]. For the paper's
        TriConv we encode the relative geometry of neighboring triangles using
        the barycenter difference together with per-axis max/min spread of the
        edge vector, giving a 9-dim descriptor.
        """
        edge_vec = pos[row] - pos[col]  # [E, 3]

        # Per-edge max / min spread (relative to the two endpoints' geometry).
        t_max = torch.maximum(pos[row], pos[col])
        t_min = torch.minimum(pos[row], pos[col])
        t_max_diff = t_max - pos[col]
        t_min_diff = t_min - pos[col]

        rel_pos = torch.cat([t_max_diff, t_min_diff, edge_vec], dim=-1)  # [E, 9]

        return rel_pos
