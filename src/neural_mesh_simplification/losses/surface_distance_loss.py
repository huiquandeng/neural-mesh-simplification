import torch
import torch.nn as nn
from torch_cluster import knn


class ProbabilisticSurfaceDistanceLoss(nn.Module):
    def __init__(self, k: int = 3, num_samples: int = 100, epsilon: float = 1e-8):
        super().__init__()
        self.k = k
        self.num_samples = num_samples
        self.epsilon = epsilon

    def forward(
        self,
        original_vertices: torch.Tensor,
        original_faces: torch.Tensor,
        simplified_vertices: torch.Tensor,
        simplified_faces: torch.Tensor,
        face_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        if original_vertices.shape[0] == 0 or simplified_vertices.shape[0] == 0:
            return torch.tensor(0.0, device=original_vertices.device)

        # Pad face probabilities once for both terms
        face_probabilities = torch.nn.functional.pad(
            face_probabilities,
            (0, max(0, simplified_faces.shape[0] - face_probabilities.shape[0])),
        )[: simplified_faces.shape[0]]

        forward_term = self.compute_forward_term(
            original_vertices,
            original_faces,
            simplified_vertices,
            simplified_faces,
            face_probabilities,
        )

        reverse_term = self.compute_reverse_term(
            original_vertices,
            original_faces,
            simplified_vertices,
            simplified_faces,
            face_probabilities,
        )

        total_loss = forward_term + reverse_term
        return total_loss

    def compute_forward_term(
        self,
        original_vertices: torch.Tensor,
        original_faces: torch.Tensor,
        simplified_vertices: torch.Tensor,
        simplified_faces: torch.Tensor,
        face_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        # If there are no faces, return zero loss
        if simplified_faces.shape[0] == 0:
            return torch.tensor(0.0, device=original_vertices.device)

        simplified_barycenters = self.compute_barycenters(
            simplified_vertices, simplified_faces
        )
        original_barycenters = self.compute_barycenters(
            original_vertices, original_faces
        )

        distances = self.compute_squared_distances(
            simplified_barycenters, original_barycenters
        )

        min_distances, _ = distances.min(dim=1)

        del distances  # Free memory

        # Compute total loss with probability penalty
        total_loss = (face_probabilities * min_distances).sum()
        probability_penalty = 1e-4 * (1.0 - face_probabilities).sum()

        del min_distances  # Free memory

        return total_loss + probability_penalty

    def compute_reverse_term(
        self,
        original_vertices: torch.Tensor,
        original_faces: torch.Tensor,
        simplified_vertices: torch.Tensor,
        simplified_faces: torch.Tensor,
        face_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        # If there are no faces, return zero loss
        if simplified_faces.shape[0] == 0:
            return torch.tensor(0.0, device=original_vertices.device)

        # If meshes are identical, reverse term should be zero
        if torch.equal(original_vertices, simplified_vertices) and torch.equal(
            original_faces, simplified_faces
        ):
            return torch.tensor(0.0, device=original_vertices.device)

        # Step 1: Sample points from the simplified mesh
        sampled_points = self.sample_points_from_triangles(
            simplified_vertices, simplified_faces, self.num_samples
        )

        # Step 2: Compute the minimum distance from each sampled point to the original mesh
        distances = self.compute_min_distances_to_original(
            sampled_points, original_vertices
        )

        # Use squared distances directly (consistent with the forward term)
        # instead of the previous max-normalize-and-scale heuristic, which made
        # the reverse term's magnitude depend on the worst outlier rather than
        # on actual surface deviation.
        distances = distances ** 2

        # Reshape face probabilities to match the sampled points
        face_probs_expanded = face_probabilities.repeat_interleave(self.num_samples)

        # Compute weighted distances
        reverse_term = (face_probs_expanded * distances).sum()

        return reverse_term

    def sample_points_from_triangles(
        self, vertices: torch.Tensor, faces: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """Vectorized point sampling from triangles"""
        num_faces = faces.shape[0]
        face_vertices = vertices[faces]

        # Generate random values for all samples at once
        sqrt_r1 = torch.sqrt(
            torch.rand(num_faces, num_samples, 1, device=vertices.device)
        )
        r2 = torch.rand(num_faces, num_samples, 1, device=vertices.device)

        # Compute barycentric coordinates
        a = 1 - sqrt_r1
        b = sqrt_r1 * (1 - r2)
        c = sqrt_r1 * r2

        # Compute samples using broadcasting
        samples = (
            a * face_vertices[:, None, 0]
            + b * face_vertices[:, None, 1]
            + c * face_vertices[:, None, 2]
        )

        del a, b, c, sqrt_r1, r2, face_vertices  # Free memory

        return samples.reshape(-1, 3)

    def compute_min_distances_to_original(
        self, sampled_points: torch.Tensor, target_vertices: torch.Tensor
    ) -> torch.Tensor:
        """Minimum distance from each sampled point to the nearest original
        vertex, via a 1-NN lookup."""
        sp_float = sampled_points.float()
        tv_float = target_vertices.float()

        # torch_cluster.knn returns (row, col) index pairs, NOT distances.
        # row indexes into sp_float (the query set), col into tv_float.
        row, col = knn(tv_float, sp_float, k=1)
        nearest = tv_float[col]
        distances = torch.norm(sp_float[row] - nearest, dim=1)

        del sp_float, tv_float

        return distances.view(-1).float()

    @staticmethod
    def compute_squared_distances(
        points1: torch.Tensor, points2: torch.Tensor
    ) -> torch.Tensor:
        """Compute squared distances efficiently using torch.cdist"""
        return torch.cdist(points1, points2, p=2).float()

    def compute_barycenters(
        self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        return vertices[faces].mean(dim=1)
