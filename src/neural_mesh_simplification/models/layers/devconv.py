import torch.nn as nn
from torch_scatter import scatter_max


class DevConv(nn.Module):
    """DevConv layer (eq. 1 of the paper):

        f_i = W_phi @ max_{j in N(i)} ( W_theta (x_i - x_j) )

    The max aggregation runs over the relative coordinates of each
    node's neighborhood, then W_phi projects the aggregated feature.
    """

    def __init__(self, in_channels, out_channels):
        super(DevConv, self).__init__()
        self.W_theta = nn.Linear(in_channels, out_channels)
        self.W_phi = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        row, col = edge_index  # edge (i, j): row = i (source), col = j (neighbor)
        rel_pos = x[row] - x[col]
        rel_pos_transformed = self.W_theta(rel_pos)  # [num_edges, out_channels]

        # max over each source node's neighborhood
        aggr_out, _ = scatter_max(
            rel_pos_transformed, row, dim=0, dim_size=x.size(0)
        )

        return self.W_phi(aggr_out)
