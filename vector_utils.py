import torch

def rotate_vectors(vectors, angles, axes):
    """
    Rotates a vector in 3D space around a given axis by a specified angle using Rodrigues' rotation formula.

    Args:
        vectors (torch.Tensor): The vector to be rotated of shape (num_vectors, 3).
        angles (torch.Tensor): The angle of rotation in radians of shape (num_vectors).
        axes (torch.Tensor): The axis of rotation of shape((num_vectors, 3)).

    Returns:
        torch.Tensor: The rotated vector of shape (num_vectors, 3).
    """
    axes = normalize_vectors(axes)
    cosTheta = torch.cos(angles)
    sinTheta = torch.sin(angles)
    term1 = vectors * cosTheta.view(-1, 1)
    term2 = torch.cross(axes, vectors, dim=1) * sinTheta.view(-1, 1)
    term3 = axes * torch.sum(axes * vectors, dim=1).view(-1, 1) * (1 - cosTheta.view(-1, 1))
    return term1 + term2 + term3


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize the given vectors.

    Args:
        vectors (torch.Tensor): The vectors to normalize of shape (... , 3).

    Returns:
        torch.Tensor: The normalized vectors of shape (... , 3).
    """
    return vectors / torch.norm(vectors, dim=-1, keepdim=True)

