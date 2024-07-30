import argparse
from typing import Tuple
import torch

def euclidean_dist(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean distance between each points of two points clouds.

    Args:
        p1 (torch.Tensor): Point 1 of shape (N1, d).
        p2 (torch.Tensor): Point 2 of shape (N2, d).

    Returns:
        torch.Tensor: L1 distance of each point in p1 to each point in p2 of shape (N1, N2).
    """
    diff = p1.unsqueeze(0) - p2.unsqueeze(1)
    return torch.linalg.vector_norm(diff, ord=2, dim=-2)

def weight_function(dists: torch.Tensor, h: float) -> torch.Tensor:
    """
    Compute the weight function for the given distances.

    Args:
        dists (torch.Tensor): Distances of shape (...).
        h (float): Support radius.

    Returns:
        torch.Tensor: Weights of shape (...).
    """
    return torch.exp(-dists.pow(2) / (2 * h**2))

def bounding_box(points: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding box of the given point cloud.

    Args:
        points (torch.Tensor): Point cloud of shape (N, d).

    Returns:
        torch.Tensor: Bounding box of shape (2, d).
    """
    min_vals, _ = torch.min(points, dim=0)
    max_vals, _ = torch.max(points, dim=0)
    bounding_box_tensor = torch.stack((min_vals, max_vals))
    return bounding_box_tensor

def initial_support_radius(points: torch.Tensor) -> float:
    """
    Compute the initial support radius for the given point cloud.

    Args:
        points (torch.Tensor): Point cloud of shape (N, d).

    Returns:
        float: Initial support radius.
    """
    J = points.shape[0]
    bounding_box_tensor = bounding_box(points)
    dbb = torch.linalg.vector_norm(bounding_box_tensor[1] - bounding_box_tensor[0], ord=2)
    return 2*dbb / J**(1/3)

def l1_medial_loss(p1: torch.Tensor, p2: torch.Tensor, h: float) -> torch.Tensor:
    """
    Compute the L1 medial loss between two point clouds.

    Args:
        p1 (torch.Tensor): Point cloud 1 of shape (N1, ...).
        p2 (torch.Tensor): Point cloud 2 of shape (N2, ...).
        h (float): Support radius.

    Returns:
        torch.Tensor: L1 medial loss.
    """
    dists = euclidean_dist(p1, p2)
    weights = weight_function(dists, h)
    return torch.sum(dists*weights)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Obj file to extract skeleton from")

    # Add the file path argument
    parser.add_argument(
        'file_path', 
        nargs='?', 
        type=str, 
        default='models/plant.obj', 
        help="Path to the OBJ file"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get the file path
    file_path = args.file_path
    
    # leaf_fit_main(file_path)
    # parameterized_fit_main(file_path)