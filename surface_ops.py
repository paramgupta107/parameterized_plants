import torch
import os

def write_traingle_patch_bv(patches: torch.Tensor, degree: int, filename: str):
    """
    Write triangular total degree patces to a bv file.
    https://www.cise.ufl.edu/research/SurfLab/bview/#file-format

    Args:
        patches (torch.Tensor): The control points of the Bezier pathes of shape (num_patches, (degree+1) * (degree+2)/2, 3).
        degree (int): The degree of the Bezier patches.
        filename (str): The name of the output file.

    Returns:
        None
    """
    if (degree+1) * (degree+2)//2 != patches.shape[1]:
        raise ValueError(f"Number of control points does not match the degree of the Bezier volume. Expected {(degree+1) * (degree+2)//2}, got {patches.shape[1]}")
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as file:
        for patch in patches:
            file.write(f"3 {degree}\n")
            for vertex in patch:
                file.write(" ".join([str(x.item()) for x in vertex]) + "\n")

def write_tensor_product_bv(patches: torch.Tensor, degree: int, filename: str):
    """
    Write tensor product patches to a bv file.
    https://www.cise.ufl.edu/research/SurfLab/bview/#file-format

    Args:
        patches (torch.Tensor): The control points of the Bezier pathes of shape (num_patches, (degree+1)**2, 3).
        degree (int): The degree of the Bezier patches.
        filename (str): The name of the output file.

    Returns:
        None
    """
    if (degree+1)**2 != patches.shape[1]:
        raise ValueError(f"Number of control points does not match the degree of the Bezier volume. Expected {(degree+1)**2}, got {patches.shape[1]}")
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as file:
        for patch in patches:
            file.write(f"4 {degree}\n")
            for vertex in patch:
                file.write(" ".join([str(x.item()) for x in vertex]) + "\n")

def sample_cubic_bezier_surfaces_points(patches: torch.Tensor, num_points_per_dim: int, device: str = 'cuda') -> torch.Tensor:
    """
    Sample points from cubic Bezier surfaces defined by the given patches.

    Args:
        patches (torch.Tensor): The control points of the Bezier patches of shape (num_patches, 16, 3).
        num_points_per_dim (int): The number of points to sample per dimension.
        device (str, optional): The device to use for computation.

    Returns:
        torch.Tensor: The sampled points from the Bezier surfaces of shape (num_patches, num_points_per_dim*num_points_per_dim, 3).
    """
    patches = patches.to(device)
    B = torch.tensor([
        [-1.0, 3.0, -3.0, 1.0],
        [3.0, -6.0, 3.0, 0.0],
        [-3.0, 3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ]).to(device)
    u = torch.linspace(0.0, 1.0, num_points_per_dim).to(device)
    U = torch.stack([u**3, u**2, u, torch.ones_like(u)], dim=1)
    v = torch.linspace(0.0, 1.0, num_points_per_dim).to(device)
    V = torch.stack([v**3, v**2, v, torch.ones_like(v)], dim=1)
    patches = patches.transpose(1, 2).view(-1, 3, 4, 4)
    points = U.matmul(B).matmul(patches).matmul(B).matmul(V.transpose(0, 1)).view(patches.shape[0], 3, -1).transpose(1, 2)
    return points

def sample_cubic_bezier_curve_points(control_points: torch.Tensor, num_points: int, device: str = 'cuda') -> torch.Tensor:
    """
    Sample points from a cubic Bezier curve defined by the given control points.

    Args:
        control_points (torch.Tensor): The control points of the Bezier curve of shape (num_curves, 4, 3).
        num_points (int): The number of points to sample.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
    Returns:
        torch.Tensor: The sampled points from the Bezier curve of shape (num_curves, num_points, 3).
    """
    control_points = control_points.to(device)
    B = torch.tensor([
        [-1.0, 3.0, -3.0, 1.0],
        [3.0, -6.0, 3.0, 0.0],
        [-3.0, 3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ]).to(device)
    t = torch.linspace(0.0, 1.0, num_points).to(device)
    T = torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=1)
    points = T.matmul(B).matmul(control_points)
    return points