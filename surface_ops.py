import torch
import os
from vector_utils import normalize_vectors, rotate_vectors

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

def sample_quadratic_bezier_curve_points(control_points: torch.Tensor, num_points: int, device: str = 'cuda') -> torch.Tensor:
    """
    Sample points from a quadratic Bezier curve defined by the given control points.

    Args:
        control_points (torch.Tensor): The control points of the Bezier curve of shape (num_curves, 3, 3).
        num_points (int): The number of points to sample.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
    Returns:
        torch.Tensor: The sampled points from the Bezier curve of shape (num_curves, num_points, 3).
    """
    control_points = control_points.to(device)
    B = torch.tensor([
        [1.0, -2.0, 1.0],
        [-2.0, 2.0, 0.0],
        [1.0, 0.0, 0.0]
    ]).to(device)
    t = torch.linspace(0.0, 1.0, num_points).to(device)
    T = torch.stack([t**2, t, torch.ones_like(t)], dim=1)
    points = T.matmul(B).matmul(control_points)
    return points

def sample_linear_bezier_curve_points(control_points: torch.Tensor, num_points: int, device: str = 'cuda') -> torch.Tensor:
    """
    Sample points from a linear Bezier curve defined by the given control points.

    Args:
        control_points (torch.Tensor): The control points of the Bezier curve of shape (num_curves, 2, 3).
        num_points (int): The number of points to sample.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
    Returns:
        torch.Tensor: The sampled points from the Bezier curve of shape (num_curves, num_points, 3).
    """
    control_points = control_points.to(device)
    B = torch.tensor([
        [-1.0, 1.0],
        [1.0, 0.0]
    ]).to(device)
    t = torch.linspace(0.0, 1.0, num_points).to(device)
    T = torch.stack([t, torch.ones_like(t)], dim=1)
    points = T.matmul(B).matmul(control_points)
    return points

def cylinderical_cubic_spline_vertices(control_points: torch.Tensor, thickness: torch.Tensor, num_segs: int, points_per_circle: int, device: str = 'cuda'):
    """
    Create a cylinrical cubic spline from the given control points and thickness.

    Args:
        control_points (torch.Tensor): The control points of the cubic spline of shape (num_curves, 4, 3).
        thickness (torch.Tensor): The thickness of the cylinder at each control point of shape (num_curves, num_segs).
        num_segs (int): The number of segments.
        points_per_circle (int): The number of points per circle.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The vertices of the cylindrical cubic spline of shape (num_curves, num_segs * points_per_circle, 3).
    """
    curve_points = sample_cubic_bezier_curve_points(control_points, num_segs, device).unsqueeze(0)
    control_points_diff = control_points[:, 1:] - control_points[:, :-1]
    control_points_secod_diff = control_points_diff[:, 1:] - control_points_diff[:, :-1]
    tangent_vectors = sample_quadratic_bezier_curve_points(control_points_diff, num_segs, device).unsqueeze(0)
    tangent_vectors = normalize_vectors(tangent_vectors)
    # curvature_vectors = sample_linear_bezier_curve_points(control_points_secod_diff, num_segs, device).unsqueeze(0)
    # normal_vectors = torch.cross(tangent_vectors, curvature_vectors, dim=-1)
    normal_vectors = torch.cross(tangent_vectors, torch.tensor([1.0,0,0], device=device).view(1,1,1,3), dim=-1)
    normal_vectors = normalize_vectors(normal_vectors)
    binormal_vectors = torch.cross(tangent_vectors, normal_vectors, dim=-1)


    angles = torch.linspace(0.0, 2.0 * 3.14159265359, points_per_circle+1, device=device)[:-1]
    angles = angles.view(-1,1,1,1)

    thickness = thickness.view(1, -1, num_segs, 1)

    circle_points = curve_points + (binormal_vectors * torch.sin(angles) + normal_vectors * torch.cos(angles)) * torch.sigmoid(thickness)
    circle_points = circle_points.transpose(0,1).transpose(1,2)
    circle_points = circle_points.reshape(-1, num_segs * points_per_circle, 3)
    return circle_points

def cylinderical_cubic_spline_faces(num_segs: int, points_per_circle: int, traingulated: bool = True, device: str = 'cuda'):
    """
    Create faces for a cylindrical cubic spline for vertrices generated using cylinderical_cubic_spline_vertices.

    Args:
        num_segs (int): The number of segments.
        points_per_circle (int): The number of points per circle.
        traingulated (bool, optional): Whether to triangulate the faces. Defaults to True.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.LongTensor: The faces of the cylindrical cubic spline of shape (num_curves * num_segs * points_per_circle, 3).
    """
    i0 = torch.arange(points_per_circle, device=device)
    i1 = i0 + points_per_circle
    i2 = (i0 + 1) % points_per_circle + points_per_circle
    i3 = (i0 + 1) % points_per_circle
    if not traingulated:
        seg =  torch.stack([i0, i1, i2, i3], dim=1)
    else:
        tri1 = torch.stack([i0, i1, i2], dim=1)
        tri2 = torch.stack([i0, i2, i3], dim=1)
        seg = torch.cat([tri1, tri2], dim=0)
    f = torch.arange(num_segs-1, device=device).view(-1, 1, 1)
    faces = (seg + f * points_per_circle).view(-1, 3 if traingulated else 4)
    return faces
    
def cubic_curve_segments_control_points(handles: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """
    Compute the control points of the cubic Bezier curve segments from the given handles.

    Args:
        handles (torch.Tensor): The handles of the cubic Bezier curve segments of shape (num_curves+1, 2, 3).
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The control points of the cubic Bezier curve segments of shape (num_curves, 4, 3).
    """
    left_handles = handles[:-1] # (num_curves, 2, 3) 
    right_handles = handles[1:] # (num_curves, 2, 3)
    p0 = torch.mean(left_handles, dim=1) # (num_curves, 3)
    p1 = left_handles[:, 1] # (num_curves, 3)
    p2 = right_handles[:, 0] # (num_curves, 3)
    p3 = torch.mean(right_handles, dim=1) # (num_curves, 3)
    control_points = torch.stack([p0, p1, p2, p3], dim=1) # (num_curves, 4, 3)
    return control_points

def cubic_bezier_curve_curvature(control_points: torch.Tensor, num_points: int, device: str = 'cuda') -> torch.Tensor:
    """
    Compute the total curvature of the cubic Bezier curve defined by the given control points.

    Args:
        control_points (torch.Tensor): The control points of the Bezier curve of shape (num_curves, 4, 3).
        num_points (int): The number of points to sample for computing the curvature.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The total curvature of the cubic Bezier curve of shape (num_curves).
    """
    control_points_diff = control_points[:, 1:] - control_points[:, :-1]
    control_points_secod_diff = control_points_diff[:, 1:] - control_points_diff[:, :-1]
    curvature_vectors = sample_linear_bezier_curve_points(control_points_secod_diff, num_points, device)
    return torch.abs(torch.max(torch.norm(curvature_vectors, dim=-1)))

def cubic_bezier_arclength(control_points: torch.Tensor, num_points: int, device: str = 'cuda') -> torch.Tensor:
    """
    Compute the arclength of the cubic Bezier curve defined by the given control points.

    Args:
        control_points (torch.Tensor): The control points of the Bezier curve of shape (num_curves, 4, 3).
        num_points (int): The number of points to sample for computing the arclength.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The arclength of the cubic Bezier curve of shape (num_curves).
    """
    control_points_diff = control_points[:, 1:] - control_points[:, :-1]
    tangent_vectors = 3 * sample_quadratic_bezier_curve_points(control_points_diff, num_points, device)
    dists = torch.norm(tangent_vectors, dim=-1)
    return torch.sum(dists, dim=-1)