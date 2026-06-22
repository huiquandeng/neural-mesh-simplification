import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data

from ..models import PointSampler, EdgePredictor, FaceClassifier


class NeuralMeshSimplification(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        edge_hidden_dim,  # Separate hidden dim for edge predictor
        num_layers,
        k,
        edge_k,
        target_ratio,
        max_candidate_triangles=8000,
        device=torch.device("cpu"),
    ):
        super(NeuralMeshSimplification, self).__init__()
        self.device = device
        self.point_sampler = PointSampler(input_dim, hidden_dim, num_layers).to(
            self.device
        )
        self.edge_predictor = EdgePredictor(
            input_dim,
            hidden_channels=edge_hidden_dim,
            k=edge_k,
        ).to(self.device)
        self.face_classifier = FaceClassifier(input_dim, hidden_dim, num_layers, k).to(
            self.device
        )
        self.k = k
        self.target_ratio = target_ratio
        self.max_candidate_triangles = max_candidate_triangles

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)

        sampled_indices, sampled_probs = self.sample_points(data)

        sampled_x = x[sampled_indices].to(self.device)
        sampled_pos = (
            data.pos[sampled_indices]
            if hasattr(data, "pos") and data.pos is not None
            else sampled_x
        ).to(self.device)

        sampled_vertices = sampled_pos  # Use sampled_pos directly as vertices

        # Update edge_index to reflect the new indices
        sampled_edge_index, _ = torch_geometric.utils.subgraph(
            sampled_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )

        # Predict edges
        sampled_edge_index = sampled_edge_index.to(self.device)
        edge_index_pred, edge_probs = self.edge_predictor(sampled_x, sampled_edge_index)

        # Generate candidate triangles
        candidate_triangles, triangle_probs = self.generate_candidate_triangles(
            edge_index_pred, edge_probs
        )

        # Classify faces
        if candidate_triangles.shape[0] > 0:
            # Create triangle features by averaging vertex features
            triangle_features = torch.zeros(
                (candidate_triangles.shape[0], sampled_x.shape[1]),
                device=self.device,
            )
            for i in range(3):
                triangle_features += sampled_x[candidate_triangles[:, i]]
            triangle_features /= 3

            # Calculate triangle centers
            triangle_centers = torch.zeros(
                (candidate_triangles.shape[0], sampled_pos.shape[1]),
                device=self.device,
            )
            for i in range(3):
                triangle_centers += sampled_pos[candidate_triangles[:, i]]
            triangle_centers /= 3

            face_probs = self.face_classifier(
                triangle_features, triangle_centers, batch=None,
                prior_probs=triangle_probs,
            )
        else:
            face_probs = torch.empty(0, device=self.device)

        if candidate_triangles.shape[0] == 0:
            simplified_faces = torch.empty((0, 3), dtype=torch.long, device=self.device)
        else:
            simplified_faces = self.select_faces(candidate_triangles, face_probs)

        return {
            "sampled_indices": sampled_indices,
            "sampled_probs": sampled_probs,
            "sampled_vertices": sampled_vertices,
            "edge_index": edge_index_pred,
            "edge_probs": edge_probs,
            "candidate_triangles": candidate_triangles,
            "triangle_probs": triangle_probs,
            "face_probs": face_probs,
            "simplified_faces": simplified_faces,
        }

    def sample_points(self, data: Data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)

        target_nodes = min(
            max(int(self.target_ratio * num_nodes), 1),
            num_nodes,
        )

        # Sample points
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        sampled_probs = self.point_sampler(x, edge_index)
        sampled_indices = self.point_sampler.sample(
            sampled_probs, num_samples=target_nodes
        )

        return sampled_indices, sampled_probs[sampled_indices]

    def select_faces(self, candidate_triangles, face_probs):
        """Select the final set of faces from the candidates.

        Two stages:
          1. Threshold the per-face inclusion probability (0.5).
          2. Manifold filtering: each undirected edge may be shared by at most
             two faces. When more than two candidate faces share an edge, keep
             only the two with the highest probability. This is what prevents
             the overlapping / non-manifold fans that otherwise read as holes
             and isolated patches in the output.
        """
        if candidate_triangles.shape[0] == 0:
            return torch.empty((0, 3), dtype=torch.long, device=self.device)

        # Stage 1: probability threshold
        keep_mask = face_probs > 0.5
        if not torch.any(keep_mask):
            # Fall back to the strongest faces so we never emit an empty mesh
            num_keep = max(1, int(self.target_ratio * candidate_triangles.shape[0]))
            topk = torch.topk(face_probs, k=min(num_keep, face_probs.shape[0])).indices
            keep_mask = torch.zeros_like(face_probs, dtype=torch.bool)
            keep_mask[topk] = True

        faces = candidate_triangles[keep_mask]
        probs = face_probs[keep_mask]

        # Stage 2: manifold edge filtering
        return self._filter_manifold(faces, probs)

    def _filter_manifold(self, faces, probs):
        if faces.shape[0] == 0:
            return faces

        # Process faces from highest to lowest probability; accept a face only
        # if none of its three edges already belongs to two accepted faces.
        order = torch.argsort(probs, descending=True)
        edge_count = {}
        accepted = []

        faces_cpu = faces[order].tolist()
        for tri in faces_cpu:
            a, b, c = tri
            edges = (
                (min(a, b), max(a, b)),
                (min(b, c), max(b, c)),
                (min(a, c), max(a, c)),
            )
            if any(edge_count.get(e, 0) >= 2 for e in edges):
                continue
            for e in edges:
                edge_count[e] = edge_count.get(e, 0) + 1
            accepted.append(tri)

        if not accepted:
            return torch.empty((0, 3), dtype=torch.long, device=self.device)

        return torch.tensor(accepted, dtype=torch.long, device=self.device)

    def generate_candidate_triangles(self, edge_index, edge_probs):

        # Handle the case when edge_index is empty
        if edge_index.numel() == 0:
            return (
                torch.empty((0, 3), dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
            )

        num_nodes = edge_index.max().item() + 1

        # Build a (symmetric) dense adjacency holding edge probabilities.
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=self.device)
        if isinstance(edge_probs, tuple):
            edge_indices, edge_values = edge_probs
            adj_matrix[edge_indices[0], edge_indices[1]] = edge_values
        else:
            adj_matrix[edge_index[0], edge_index[1]] = edge_probs
        # Symmetrize so triangle detection is direction-agnostic
        adj_matrix = torch.maximum(adj_matrix, adj_matrix.t())

        # Adjust k based on the number of nodes
        k = min(self.k, num_nodes - 1)
        if k < 2:
            return (
                torch.empty((0, 3), dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
            )

        # Find k strongest neighbors for each node
        _, knn_indices = torch.topk(adj_matrix, k=k, dim=1)  # [N, k]

        # Enumerate all (i, n1, n2) candidates with n1, n2 among i's knn.
        nodes = torch.arange(num_nodes, device=self.device).view(-1, 1, 1)
        # all unordered pairs (j, l) with j < l within the k neighbors
        jj, ll = torch.triu_indices(k, k, offset=1, device=self.device)
        n1 = knn_indices[:, jj]  # [N, P]
        n2 = knn_indices[:, ll]  # [N, P]
        i_idx = nodes.expand(-1, n1.shape[1], 1).reshape(-1)
        n1 = n1.reshape(-1)
        n2 = n2.reshape(-1)

        # Probabilities of the three edges of each candidate
        p_in1 = adj_matrix[i_idx, n1]
        p_in2 = adj_matrix[i_idx, n2]
        p_n1n2 = adj_matrix[n1, n2]

        # Keep only candidates where all three edges exist AND the three
        # vertices are distinct. topk pads with index 0 when a node has fewer
        # than k non-zero neighbors, which would otherwise produce degenerate
        # triangles like [i, 0, 0].
        distinct = (i_idx != n1) & (i_idx != n2) & (n1 != n2)
        valid = (p_in1 > 0) & (p_in2 > 0) & (p_n1n2 > 0) & distinct
        if not torch.any(valid):
            return (
                torch.empty((0, 3), dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
            )

        triangles = torch.stack([i_idx[valid], n1[valid], n2[valid]], dim=1)

        # Paper's initial triangle probability: arithmetic mean of edge probs
        triangle_probs = (p_in1[valid] + p_in2[valid] + p_n1n2[valid]) / 3.0

        # De-duplicate triangles (same vertex set found from different anchors)
        sorted_tri, _ = torch.sort(triangles, dim=1)
        _, unique_idx = torch.unique(sorted_tri, dim=0, return_inverse=True)
        keep = torch.ones(triangles.shape[0], dtype=torch.bool, device=self.device)
        seen = torch.full(
            (unique_idx.max().item() + 1,), -1, dtype=torch.long, device=self.device
        )
        order = torch.arange(triangles.shape[0], device=self.device)
        seen[unique_idx] = order  # last write wins -> one representative per group
        keep = seen[unique_idx] == order

        triangles = triangles[keep]
        triangle_probs = triangle_probs[keep]

        # Cap the number of candidates to bound memory in the downstream
        # geometric losses (which compute dense all-pairs over candidates).
        # Keep the highest-probability triangles.
        if (
            self.max_candidate_triangles is not None
            and triangles.shape[0] > self.max_candidate_triangles
        ):
            top = torch.topk(triangle_probs, k=self.max_candidate_triangles).indices
            triangles = triangles[top]
            triangle_probs = triangle_probs[top]

        return triangles, triangle_probs
