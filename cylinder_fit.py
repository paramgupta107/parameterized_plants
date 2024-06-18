from surface_ops import cubic_bezier_arclength, cylinderical_cubic_spline_vertices, cylinderical_cubic_spline_faces, sample_cubic_bezier_curve_points, cubic_curve_segments_control_points, cubic_bezier_curve_curvature
from mesh_ops import load_mesh, polylines_edges, save_mesh
import torch
import kaolin
from typing import Tuple

def fit_cubic_bezier_cylinder(targetMeshVertices: torch.Tensor, targetMeshFaces: torch.LongTensor, control_points: torch.Tensor, thickness: torch.Tensor, num_segs: int, points_per_circle: int, epochs: int = 1000, lr: float = 0.01, device: str = 'cuda', logs_path: str = 'logs_cylinder') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fits a cubic bezier cylinder to a target mesh using optimization.

    Args:
        targetMeshVertices (torch.Tensor): The vertices of the target mesh of shape (1, num_vertices, 3).
        targetMeshFaces (torch.LongTensor): The faces of the target mesh of shape (num_faces, 3).
        control_points (torch.Tensor): The control points of the cubic bezier curve of shape (1, num_control_points, 3).
        thickness (torch.Tensor): The thickness of the cylinder at each control point of shape (1, num_segs).
        num_segs (int): The number of segments.
        points_per_circle (int): The number of points per circle.
        epochs (int, optional): The number of optimization epochs. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.

    Returns:
        torch.Tensor: The fitted control points of the cubic bezier curve of shape (1, num_control_points, 3).
    """
    timelapse = kaolin.visualize.Timelapse(logs_path)
    timelapse.add_mesh_batch(category='target',
                         faces_list=[targetMeshFaces.cpu()],
                         vertices_list=[targetMeshVertices.cpu()])
    targetPointCloud, _ = kaolin.ops.mesh.sample_points(targetMeshVertices, targetMeshFaces, 10000)
    optimizer = torch.optim.Adam([control_points, thickness], lr=lr)
    startingFaces = cylinderical_cubic_spline_faces(num_segs, points_per_circle, device=device)
    for epoch in range(epochs):
        startingVertices = cylinderical_cubic_spline_vertices(control_points, thickness, num_segs, points_per_circle, device=device)
        
        startingPointCloud, _ = kaolin.ops.mesh.sample_points(startingVertices, startingFaces, 10000)
        loss = kaolin.metrics.pointcloud.chamfer_distance(startingPointCloud, targetPointCloud)
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
    return control_points, thickness

def fit_cubic_bezier_cylinder_segs(targetMeshVertices: torch.Tensor, targetMeshFaces: torch.LongTensor, handles: torch.Tensor, thickness: torch.Tensor, num_segs: int, points_per_circle: int, epochs: int = 1000, lr: float = 0.01, device: str = 'cuda', logs_path: str = 'logs_cylinder_segs') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fits a cubic bezier cylinder segments to a target mesh using optimization.

    Args:
        targetMeshVertices (torch.Tensor): The vertices of the target mesh of shape (1, num_vertices, 3).
        targetMeshFaces (torch.LongTensor): The faces of the target mesh of shape (num_faces, 3).
        handles (torch.Tensor): The handles of the cubic Bezier curve segments of shape (num_curves+1, 2, 3).
        thickness (torch.Tensor): The thickness of the cylinder at each control point of shape (num_curves, num_segs).
        num_segs (int): The number of segments.
        points_per_circle (int): The number of points per circle.
        epochs (int, optional): The number of optimization epochs. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.

    Returns:
        torch.Tensor: The fitted control points of the cubic bezier curve of shape (1, num_control_points, 3).
    """
    timelapse = kaolin.visualize.Timelapse(logs_path)
    timelapse.add_mesh_batch(category='target',
                         faces_list=[targetMeshFaces.cpu()],
                         vertices_list=[targetMeshVertices.cpu()])
    targetPointCloud, _ = kaolin.ops.mesh.sample_points(targetMeshVertices, targetMeshFaces, 10000)
    num_curves = handles.shape[0] - 1
    startingFaces = cylinderical_cubic_spline_faces(num_segs*num_curves, points_per_circle, device=device)
    transform = torch.eye(3, device=device, requires_grad=True)
    translate = torch.zeros(1, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform, translate], lr=lr)
    for epoch in range(epochs):
        if epoch == epochs//3:
            optimizer = torch.optim.Adam([transform, translate, handles], lr=lr/10)
            print("------------------Optimizing handles.------------------")
        if epoch == epochs//6:
            optimizer = torch.optim.Adam([transform, translate, handles, thickness], lr=lr/100)
            print("------------------Optimizing thickness.------------------")
        movedHandles = torch.matmul(handles, transform) + translate
        control_points = cubic_curve_segments_control_points(movedHandles, device=device)
        startingVertices = cylinderical_cubic_spline_vertices(control_points, thickness, num_segs, points_per_circle, device=device).reshape(1, -1, 3)
        
        startingPointCloud, _ = kaolin.ops.mesh.sample_points(startingVertices, startingFaces, 10000)
        chamfer_loss = kaolin.metrics.pointcloud.chamfer_distance(startingPointCloud, targetPointCloud)
        curvature_loss = cubic_bezier_curve_curvature(control_points, num_segs, device=device)
        arc_lengths = cubic_bezier_arclength(control_points, num_segs, device=device)
        arc_length_loss = torch.max(arc_lengths) - torch.min(arc_lengths)
        loss = chamfer_loss# + 0.01 * curvature_loss + 0.1*arc_length_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()} \t| Chamfer Loss {chamfer_loss.item()} \t| Curvature Loss {curvature_loss.item()} \t| Arc Length Loss {arc_length_loss.item()}")
        if epoch % 10 == 0:
            timelapse.add_mesh_batch(
            iteration=epoch+1,
            category='fitted_mesh',
            vertices_list=[startingVertices.cpu()],
            faces_list=[startingFaces.cpu()]
            )
    print("Optimization finished.")
    return torch.matmul(handles, transform) + translate, thickness

def main_cubic_curve():
    device = 'cuda'
    num_segs, points_per_circle = 100, 25
    control_points = torch.tensor([[[0.0,0,0], [-10,30,0], [0,60,0], [20,100,0]]], device=device, requires_grad=True)
    thickness = torch.ones(1, num_segs, device=device, requires_grad=True)
    
    targetVerts, targetFaces = load_mesh('models/stem_1.obj')
    control_points, thickness = fit_cubic_bezier_cylinder(targetVerts, targetFaces, control_points, thickness, num_segs, points_per_circle, lr=0.1, epochs=5000, device='cuda')
    
    fit_verts = cylinderical_cubic_spline_vertices(control_points, thickness, num_segs, points_per_circle, device=device)
    faces = cylinderical_cubic_spline_faces(num_segs, points_per_circle, traingulated=False, device=device)
    save_mesh(fit_verts, faces, "out/cylinder.obj")
    curveEdges = polylines_edges(num_segs, device=device)
    curve_points = sample_cubic_bezier_curve_points(control_points, num_segs, device=device)
    save_mesh(curve_points, curveEdges, "out/cylinder_curve.obj")

    

def main_cubic_curve_segs():
    device = 'cuda'
    num_segs, points_per_circle = 10, 25
    # handles = torch.tensor([[[10.0, -30.0, 0.0], [-10.0, 30.0, 0.0]], [[0,60,0], [40,140,0]], [[-5,15,0], [5,25,0]]], device=device, requires_grad=True)
    handles = torch.tensor([[[-5, -5.0, 0.0], [5.0, 5.0, 0.0]], [[10,10,0], [15,15,0]], [[20,20,0], [25,25,0]], [[30, 30, 0], [35,35,0]], [[40, 40, 0], [45, 45, 0]]], device=device, requires_grad=True)
    num_curves = handles.shape[0] - 1
    thickness = torch.ones(num_curves, num_segs, device=device, requires_grad=True)
    
    targetVerts, targetFaces = load_mesh('models/stem_nh.obj')
    handles, thickness = fit_cubic_bezier_cylinder_segs(targetVerts, targetFaces, handles, thickness, num_segs, points_per_circle, lr=0.1, epochs=5000, device='cuda')

    control_points = cubic_curve_segments_control_points(handles, device=device)
    fit_verts = cylinderical_cubic_spline_vertices(control_points, thickness, num_segs, points_per_circle, device=device).reshape(-1, 3)
    faces = cylinderical_cubic_spline_faces(num_segs*num_curves, points_per_circle, traingulated=False, device=device).reshape(-1, 4)
    save_mesh(fit_verts, faces, "out/cylinder_seg_h.obj")
    curveEdges = polylines_edges(num_segs*num_curves, device=device)
    curvePoints = sample_cubic_bezier_curve_points(control_points, num_segs, device=device).reshape(-1, 3)
    save_mesh(curvePoints, curveEdges, "out/cylinder_curve_seg_h.obj")

if __name__ == '__main__':
    # main_cubic_curve()
    main_cubic_curve_segs()