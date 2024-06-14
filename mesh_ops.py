import torch
import numpy as np
import kaolin
from typing import Tuple
import os

def save_mesh(vertices: torch.Tensor, faces: torch.LongTensor, filename: str):
    """
    Create an OBJ file from the given vertices and faces.

    Args:
        vertices (torch.Tensor): vertex coordinates of shape (num_vertices, 3) or (1, num_vertices, 3).
        faces (torch.LongTensor): face indices of shape (num_faces, 3).
        filename (str): Name of the output OBJ file.

    Returns:
        None
    """
    if len(vertices.shape) > 2:
        vertices = vertices.squeeze(0)
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as file:
        for vertex in vertices:
           file.write("v " + " ".join([str(x.item()) for x in vertex]) + "\n")
        for face in faces:
            file.write("f " + " ".join([str(x.item()+1) for x in face]) + "\n")

def save_point_cloud_to_ply(point_cloud: torch.Tensor, filename: str):
    """
    Save a point cloud to a PLY file.

    Args:
        point_cloud (torch.Tensor): Point cloud coordinates of shape (num_points, 3).
        filename (str): Name of the output PLY file.

    Returns:
        None
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as file:
        num_vertices = point_cloud.shape[0]
        
        # Write PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {num_vertices}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        
        # Write point cloud vertices
        for point in point_cloud:
            file.write(f"{point[0].item()} {point[1].item()} {point[2].item()}\n")

def load_mesh(mesh_path: str, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a mesh from the given OBJ file.

    Args:
        mesh_path (str): Path to the OBJ file.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The vertices of the mesh of shape (1, num_vertices, 3).
        torch.Tensor: The faces of the mesh of shape (num_faces, 3).
    """
    mesh = kaolin.io.obj.import_mesh(mesh_path, triangulate=True)
    vertices = mesh.vertices
    faces = mesh.faces
    return vertices[None, :, :].to(device=device), faces.to(device=device)

def gen_starting_net(radius: float = 1, subdiv: int = 5, requires_grad: bool = True, device: str = 'cuda') -> Tuple[torch.Tensor, torch.LongTensor]:
    """
    Generate a triangular net by subdividing an octahedron.

    Args:
        radius (float, optional): Circumradius of the octahedron before subdivision.
        subdiv (int, optional): The number of subdivisions to perform on the mesh.
        requires_grad (bool, optional): Whether the generated network requires gradient computation.
        device (str, optional): The device to use for computation.

    Returns:
        torch.Tensor: The new vertices of the subdivided mesh of shape (1, num_vertices, 3).
        torch.LongTensor: The new faces of the subdivided mesh of shape (num_faces, 3).
    """
    vertices = [
        [1.000000, 0.000000, 0.000000],
        [-1.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, -1.000000],
        [0.000000, 0.000000, 1.000000],
        [0.000000, 1.000000, 0.000000],
        [0.000000, -1.000000, 0.000000]
    ]

    faces = [
        [4, 0, 2],
        [4, 2, 1],
        [4, 1, 3],
        [4, 3, 0],
        [5, 2, 0],
        [5, 1, 2],
        [5, 3, 1],
        [5, 0, 3]
    ]
    vertsNp = np.asarray(vertices)[None, :, :] * radius
    faceNp = np.asarray(faces)
    v = torch.tensor(vertsNp)
    f = torch.tensor(faceNp, dtype=torch.long)
    newV, newF = kaolin.ops.mesh.subdivide_trianglemesh(v.float(), f, subdiv)
    return torch.tensor(newV.numpy(), requires_grad=requires_grad, device=device), torch.tensor(newF.numpy(), device=device)

def laplace_regularizer_const(mesh_verts: torch.Tensor, mesh_faces: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Laplace regularization constant for a given mesh.
    Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
    https://mgarland.org/class/geom04/material/smoothing.pdf
    Args:
        mesh_verts (torch.Tensor): Tensor containing the vertices of the mesh of shape (num_vertices, 3).
        mesh_faces (torch.Tensor): Tensor containing the faces of the mesh of shape (num_faces, 3).

    Returns:
        torch.Tensor: The Laplace regularization constant.

    """
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

def vertices_faces_to_patches(vertices: torch.Tensor, faces: torch.LongTensor) -> torch.Tensor:
    """
    Convert vertices and faces to linear traingular patches.

    Args:
        vertices (torch.Tensor): List of vertex coordinates of shape (num_vertices, 3) or (1, num_vertices, 3).
        faces (torch.LongTensor): List of face indices of shape (num_faces, 3).

    Returns:
        patches (torch.Tensor): return tensor of shape (num_faces, 3, 3)
    """
    if len(vertices.shape) > 2:
        vertices = vertices.squeeze()
    flattened_faces = faces.flatten()
    flattened_vertices = vertices.index_select(0, flattened_faces)
    return flattened_vertices.view(faces.shape[0], 3, 3)

def fit_mesh(targetMeshVertices: torch.Tensor, targetMeshFaces: torch.LongTensor, startingVertices: torch.Tensor, startingFaces: torch.LongTensor, epochs: int = 1000, lr: float = 0.01, device: str = 'cuda', logs_path: str = 'logs') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fits a mesh to a target mesh using optimization.

    Args:
        targetMeshVertices (torch.Tensor): The vertices of the target mesh of shape (1, num_vertices, 3).
        targetMeshFaces (torch.LongTensor): The faces of the target mesh of shape (num_faces, 3).
        startingVertices (torch.Tensor): The starting vertices of the mesh of shape (1, num_vertices, 3)
        startingFaces (torch.LongTensor): The faces of the mesh of shape (num_faces, 3).
        epochs (int, optional): The number of optimization epochs. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.

    Returns:
        torch.Tensor: The fitted vertices of the mesh of shape (1, num_vertices, 3).
        torch.Tensor: The faces of the mesh of shape (num_faces, 3).
    """
    
    timelapse = kaolin.visualize.Timelapse(logs_path)
    timelapse.add_mesh_batch(category='target',
                         faces_list=[targetMeshFaces.cpu()],
                         vertices_list=[targetMeshVertices.cpu()])
    targetPointCloud, _ = kaolin.ops.mesh.sample_points(targetMeshVertices, targetMeshFaces, 10000)
    optimizer = torch.optim.Adam([startingVertices], lr=lr)
    for epoch in range(epochs):
        startingPointCloud, _ = kaolin.ops.mesh.sample_points(startingVertices, startingFaces, 10000)
        loss = kaolin.metrics.pointcloud.chamfer_distance(startingPointCloud, targetPointCloud) + 10 * laplace_regularizer_const(startingVertices.squeeze(), startingFaces)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")
        if epoch % 10 == 0:
            timelapse.add_mesh_batch(
            iteration=epoch+1,
            category='fitted_mesh',
            vertices_list=[startingVertices.cpu()],
            faces_list=[startingFaces.cpu()]
            )
    print("Optimization finished.")
    return startingVertices, startingFaces
