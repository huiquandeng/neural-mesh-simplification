import warnings

import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_softmax
from torch_sparse import SparseTensor

from .layers.devconv import DevConv

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")


class EdgePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, k):
        super(EdgePredictor, self).__init__()
        self.k = k
        self.devconv = DevConv(in_channels, hidden_channels)

        # Self-attention components
        self.W_q = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.W_k = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            raise ValueError("Edge index is empty")

        # Step 1: Extend original mesh connectivity with k-nearest neighbors
        knn_edges = knn_graph(x, k=self.k, flow="target_to_source")

        # Ensure knn_edges indices are within bounds
        max_idx = x.size(0) - 1
        valid_edges = (knn_edges[0] <= max_idx) & (knn_edges[1] <= max_idx)
        knn_edges = knn_edges[:, valid_edges]

        # Combine original edges with knn edges
        if edge_index.numel() > 0:
            extended_edges = torch.cat([edge_index, knn_edges], dim=1)
            # Remove duplicate edges
            extended_edges = torch.unique(extended_edges, dim=1)
        else:
            extended_edges = knn_edges

        # Step 2: Apply DevConv on the extended connectivity
        features = self.devconv(x, extended_edges)

        # Step 3: Apply sparse self-attention over the *extended* edges.
        # Computing attention/adjacency on the extended graph (not the sparse
        # sampled subgraph) is what guarantees every sampled node keeps
        # connectivity, avoiding isolated points and holes after simplification.
        attention_scores = self.compute_attention_scores(features, extended_edges)

        # Step 4: Compute simplified adjacency matrix on the extended graph
        simplified_adj_indices, simplified_adj_values = (
            self.compute_simplified_adjacency(attention_scores, extended_edges)
        )

        return simplified_adj_indices, simplified_adj_values

    def compute_attention_scores(self, features, edges):
        if edges.numel() == 0:
            raise ValueError("Edge index is empty")

        row, col = edges
        q = self.W_q(features)
        k = self.W_k(features)

        # Compute (W_q f_j)^T (W_k f_i)
        attention = (q[row] * k[col]).sum(dim=-1)

        # Apply softmax for each source node
        attention_scores = scatter_softmax(attention, row, dim=0)

        return attention_scores

    def compute_simplified_adjacency(self, attention_scores, edge_index):
        if edge_index.numel() == 0:
            raise ValueError("Edge index is empty")

        num_nodes = edge_index.max().item() + 1
        row, col = edge_index

        # Ensure indices are within bounds
        if row.numel() > 0:
            assert torch.all(row < num_nodes) and torch.all(
                row >= 0
            ), f"Row indices out of bounds: min={row.min()}, max={row.max()}, num_nodes={num_nodes}"
        if col.numel() > 0:
            assert torch.all(col < num_nodes) and torch.all(
                col >= 0
            ), f"Column indices out of bounds: min={col.min()}, max={col.max()}, num_nodes={num_nodes}"

        # Create sparse attention matrix
        S = SparseTensor(
            row=row,
            col=col,
            value=attention_scores,
            sparse_sizes=(num_nodes, num_nodes),
            trust_data=True,  # Since we verified the indices above
        )

        # Create original adjacency matrix
        A = SparseTensor(
            row=row,
            col=col,
            value=torch.ones(edge_index.size(1), device=edge_index.device),
            sparse_sizes=(num_nodes, num_nodes),
            trust_data=True,  # Since we verified the indices above
        )

        # Compute A_s = S * A * S^T using coalesced sparse tensors
        A_s = S.matmul(A).matmul(S.t())

        # Convert to COO format
        row, col, value = A_s.coo()
        indices = torch.stack([row, col], dim=0)

        return indices, value
