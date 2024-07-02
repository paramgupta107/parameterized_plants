from surface_ops import cubic_bezier_arclength, cylinderical_cubic_spline_faces_seperated, cylinderical_cubic_spline_vertices, cylinderical_cubic_spline_faces, sample_cubic_bezier_curve_points, cubic_c1_curve_segments_control_points, cubic_bezier_curve_curvature, sample_quadratic_bezier_curve_points, cubic_c0_curve_segments_control_points, subdivide_c0_quadratic_control_points, subdivide_c1_cubic_handles
from mesh_ops import load_mesh, polylines_edges, save_mesh
import torch
import kaolin
from typing import Tuple
import matplotlib.pyplot as plt

def subdivide_thickness(thickness: torch.Tensor, device: str = 'cuda'):
    """
    Subidivides the thickness points of the cylinder.

    Args:
        thickness (torch.Tensor): The thickness of the cylinder at each control point of shape (num_curves, num_segs, 1).\
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The subdivided thickness of the cylinder at each control point of shape (2*num_curves, num_segs, 1).
    """
    num_curves = thickness.shape[0]
    num_segs = thickness.shape[1]
    return torch.stack([thickness, thickness], dim=1).transpose(1, 2).reshape(num_curves*2,num_segs, -1)

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

def fit_cubic_c1_bezier_cylinder_segs(targetMeshVertices: torch.Tensor, targetMeshFaces: torch.LongTensor, handles: torch.Tensor, thickness_control_points: torch.Tensor, num_segs: int, points_per_circle: int, epochs: int = 1000, lr: float = 0.01, device: str = 'cuda', logs_path: str = 'logs_cylinder_segs') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fits a cubic bezier cylinder segments to a target mesh using optimization.

    Args:
        targetMeshVertices (torch.Tensor): The vertices of the target mesh of shape (1, num_vertices, 3).
        targetMeshFaces (torch.LongTensor): The faces of the target mesh of shape (num_faces, 3).
        handles (torch.Tensor): The handles of the cubic Bezier curve segments of shape (num_curves+1, 2, 3).
        thickness_control_points (torch.Tensor): The thickness control of the cylinder at each control point of shape (num_curves, 3, 1).
        num_segs (int): The number of segments.
        points_per_circle (int): The number of points per circle.
        epochs (int, optional): The number of optimization epochs. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.

    Returns:
        torch.Tensor: The fitted handles of the cubic Bezier curve segments of shape (num_curves+1, 2, 3).
        torch.Tensor: The thickness control of the cylinder at each control point of shape (num_curves, 3, 1).
    """
    timelapse = kaolin.visualize.Timelapse(logs_path)
    timelapse.add_mesh_batch(category='target',
                         faces_list=[targetMeshFaces.cpu()],
                         vertices_list=[targetMeshVertices.cpu()])
    targetPointCloud, _ = kaolin.ops.mesh.sample_points(targetMeshVertices, targetMeshFaces, 10000)
    num_curves = handles.shape[0] - 1
    startingFaces = cylinderical_cubic_spline_faces_seperated(num_segs, num_curves, points_per_circle, device=device)
    transform = torch.eye(3, device=device, requires_grad=True)
    translate = torch.zeros(1, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform, translate], lr=lr)
    for epoch in range(epochs):
        if epoch == 100:
            optimizer = torch.optim.Adam([transform, translate, handles], lr=lr/10)
            print("------------------Optimizing handles.------------------")
        if epoch == 500:
            optimizer = torch.optim.Adam([transform, translate, handles, thickness_control_points], lr=lr/100)
            print("------------------Optimizing thickness.------------------")
        movedHandles = torch.matmul(handles, transform) + translate
        control_points = cubic_c1_curve_segments_control_points(movedHandles, device=device)
        thickness = sample_quadratic_bezier_curve_points(thickness_control_points, num_segs, device=device)
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
    return torch.matmul(handles, transform) + translate, thickness_control_points

def fit_cubic_c0_bezier_cylinder_segs(targetMeshVertices: torch.Tensor, targetMeshFaces: torch.LongTensor, points: torch.Tensor, thickness_control_points: torch.Tensor, num_segs: int, points_per_circle: int, epochs: int = 1000, lr: float = 0.01, device: str = 'cuda', logs_path: str = 'logs_cylinder_segs') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fits a cubic bezier cylinder segments to a target mesh using optimization.

    Args:
        targetMeshVertices (torch.Tensor): The vertices of the target mesh of shape (1, num_vertices, 3).
        targetMeshFaces (torch.LongTensor): The faces of the target mesh of shape (num_faces, 3).
        points (torch.Tensor): The points of the cubic Bezier curve segments of shape (3*num_curves + 1, 3).
        thickness_control_points (torch.Tensor): The thickness control of the cylinder at each control point of shape (num_curves, 3, 1).
        num_segs (int): The number of segments.
        points_per_circle (int): The number of points per circle.
        epochs (int, optional): The number of optimization epochs. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.

    Returns:
        torch.Tensor: The fitted handles of the cubic Bezier curve segments of shape (num_curves+1, 2, 3).
        torch.Tensor: The thickness control of the cylinder at each control point of shape (num_curves, 3, 1).
    """
    timelapse = kaolin.visualize.Timelapse(logs_path)
    timelapse.add_mesh_batch(category='target',
                         faces_list=[targetMeshFaces.cpu()],
                         vertices_list=[targetMeshVertices.cpu()])
    targetPointCloud, _ = kaolin.ops.mesh.sample_points(targetMeshVertices, targetMeshFaces, 10000)
    num_curves = (points.shape[0]-1)//3
    startingFaces = cylinderical_cubic_spline_faces_seperated(num_segs, num_curves, points_per_circle, device=device)
    transform = torch.eye(3, device=device, requires_grad=True)
    translate = torch.zeros(1, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform, translate], lr=lr)
    for epoch in range(epochs):
        if epoch == 100:
            optimizer = torch.optim.Adam([transform, translate, points], lr=lr/10)
            print("------------------Optimizing handles.------------------")
        if epoch == 500:
            optimizer = torch.optim.Adam([transform, translate, points, thickness_control_points], lr=lr/100)
            print("------------------Optimizing thickness.------------------")
        movedPoints = torch.matmul(points, transform) + translate
        control_points = cubic_c0_curve_segments_control_points(movedPoints, device=device)
        thickness = sample_quadratic_bezier_curve_points(thickness_control_points, num_segs, device=device)
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
    return torch.matmul(points, transform) + translate, thickness_control_points

def fit_cubic_c1_bezier_cylinder_subidvided_segs(targetMeshVertices: torch.Tensor, targetMeshFaces: torch.LongTensor, handles: torch.Tensor, thickness: torch.Tensor, num_segs: int, points_per_circle: int, epochs: int = 1000, lr: float = 0.01, device: str = 'cuda', logs_path: str = 'logs_cylinder_segs') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fits a cubic bezier cylinder segments to a target mesh using optimization.

    Args:
        targetMeshVertices (torch.Tensor): The vertices of the target mesh of shape (1, num_vertices, 3).
        targetMeshFaces (torch.LongTensor): The faces of the target mesh of shape (num_faces, 3).
        handles (torch.Tensor): The handles of the cubic Bezier curve segments of shape (num_curves+1, 2, 3).
        thickness_control_points (torch.Tensor): The thickness control of the cylinder at each control point of shape (num_curves, 3, 1).
        num_segs (int): The number of segments.
        points_per_circle (int): The number of points per circle.
        epochs (int, optional): The number of optimization epochs. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.

    Returns:
        torch.Tensor: The fitted handles of the cubic Bezier curve segments of shape (num_subdivided_curves+1, 2, 3).
        torch.Tensor: The thickness control of the cylinder at each control point of shape (num_subdivided_curves, 3, 1).
    """
    timelapse = kaolin.visualize.Timelapse(logs_path)
    timelapse.add_mesh_batch(category='target',
                         faces_list=[targetMeshFaces.cpu()],
                         vertices_list=[targetMeshVertices.cpu()])
    targetPointCloud, _ = kaolin.ops.mesh.sample_points(targetMeshVertices, targetMeshFaces, 10000)
    num_curves = handles.shape[0] - 1
    startingFaces = cylinderical_cubic_spline_faces_seperated(num_segs, num_curves, points_per_circle, device=device)
    transform = torch.eye(3, device=device, requires_grad=True)
    translate = torch.zeros(1, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform, translate], lr=lr)
    numSubdivisions = 0
    for epoch in range(epochs):
        if epoch == 100:
            optimizer = torch.optim.Adam([transform, translate, handles], lr=lr/10)
            print("------------------Optimizing handles.------------------")
        if epoch == 500:
            optimizer = torch.optim.Adam([transform, translate, handles, thickness], lr=lr/100)
            print("------------------Optimizing thickness.------------------")
        if epoch > 500 and (epoch % 500 == 0) and numSubdivisions < 5:
            handles = subdivide_c1_cubic_handles(handles, device=device).detach().requires_grad_(True)
            # thickness_control_points = subdivide_c0_quadratic_control_points(thickness, device=device).detach().requires_grad_(True)
            thickness = subdivide_thickness(thickness, device=device).detach().requires_grad_(True)
            num_curves = handles.shape[0] - 1
            startingFaces = cylinderical_cubic_spline_faces_seperated(num_segs, num_curves, points_per_circle, device=device)
            numSubdivisions += 1
            optimizer = torch.optim.Adam([transform, translate, handles, thickness], lr=lr/100)
            print(f"Subdivisions: {numSubdivisions}")
        movedHandles = torch.matmul(handles, transform) + translate
        control_points = cubic_c1_curve_segments_control_points(movedHandles, device=device)
        # thickness = sample_quadratic_bezier_curve_points(thickness_control_points, num_segs, device=device)
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

def main_cubic_c1_curve_segs():
    device = 'cuda'
    num_segs, points_per_circle = 5, 25
    num_handles = 10
    # handles = torch.tensor([[[10.0, -30.0, 0.0], [-10.0, 30.0, 0.0]], [[0,60,0], [40,140,0]], [[-5,15,0], [5,25,0]]], device=device, requires_grad=True)
    # handles = torch.tensor([[[-5, -5.0, 0.0], [5.0, 5.0, 0.0]], [[10,10,0], [15,15,0]], [[20,20,0], [25,25,0]], [[30, 30, 0], [35,35,0]], [[40, 40, 0], [45, 45, 0]]], device=device, requires_grad=True)
    # handles = torch.arange(0,120, 5, device=device).reshape(-1, 2, 1).repeat(1,1,3).float()
    handles = torch.linspace(0, 100, steps=2*num_handles, device=device).reshape(-1, 2, 1).repeat(1,1,3)
    handles = handles.detach().requires_grad_(True)
    num_curves = handles.shape[0] - 1
    thickness_control_points = torch.ones(num_curves, 3, 1, device=device, requires_grad=True)
    targetVerts, targetFaces = load_mesh('models/stem_nh.obj')
    handles, thickness_control_points = fit_cubic_c1_bezier_cylinder_segs(targetVerts, targetFaces, handles, thickness_control_points, num_segs, points_per_circle, lr=0.1, epochs=6000, device='cuda')

    control_points = cubic_c1_curve_segments_control_points(handles, device=device)
    thickness = sample_quadratic_bezier_curve_points(thickness_control_points, num_segs, device=device)
    fit_verts = cylinderical_cubic_spline_vertices(control_points, thickness, num_segs, points_per_circle, device=device).reshape(-1, 3)
    faces = cylinderical_cubic_spline_faces_seperated(num_segs, num_curves, points_per_circle, traingulated=False, device=device).reshape(-1, 4)
    save_mesh(fit_verts, faces, "out/cylinder_seg_h.obj")
    curveEdges = polylines_edges(num_segs*num_curves, device=device)
    curvePoints = sample_cubic_bezier_curve_points(control_points, num_segs, device=device).reshape(-1, 3)
    save_mesh(curvePoints, curveEdges, "out/cylinder_curve_seg_h.obj")

def main_cubic_c1_curve_subdiv_segs():
    device = 'cuda'
    num_segs, points_per_circle = 50, 25
    num_handles = 2
    # handles = torch.tensor([[[10.0, -30.0, 0.0], [-10.0, 30.0, 0.0]], [[0,60,0], [40,140,0]], [[-5,15,0], [5,25,0]]], device=device, requires_grad=True)
    # handles = torch.tensor([[[-5, -5.0, 0.0], [5.0, 5.0, 0.0]], [[10,10,0], [15,15,0]], [[20,20,0], [25,25,0]], [[30, 30, 0], [35,35,0]], [[40, 40, 0], [45, 45, 0]]], device=device, requires_grad=True)
    # handles = torch.arange(0,120, 5, device=device).reshape(-1, 2, 1).repeat(1,1,3).float()
    handles = torch.linspace(0, 100, steps=2*num_handles, device=device).reshape(-1, 2, 1).repeat(1,1,3)
    handles = handles.detach().requires_grad_(True)
    num_curves = handles.shape[0] - 1
    thickness = torch.ones(num_curves, num_segs, 1, device=device, requires_grad=True)
    # thickness_control_points = torch.ones(num_curves, 3, 1, device=device, requires_grad=True)
    targetVerts, targetFaces = load_mesh('models/stem_nh.obj')
    handles, thickness = fit_cubic_c1_bezier_cylinder_subidvided_segs(targetVerts, targetFaces, handles, thickness, num_segs, points_per_circle, lr=0.1, epochs=6000, device='cuda')
    graph_data = thickness.reshape(-1, 1).cpu().detach().numpy()
    plt.plot(graph_data)
    plt.show()

    num_curves = handles.shape[0] - 1

    control_points = cubic_c1_curve_segments_control_points(handles, device=device)
    # thickness = sample_quadratic_bezier_curve_points(thickness_control_points, num_segs, device=device)
    fit_verts = cylinderical_cubic_spline_vertices(control_points, thickness, num_segs, points_per_circle, device=device).reshape(-1, 3)
    faces = cylinderical_cubic_spline_faces_seperated(num_segs, num_curves, points_per_circle, traingulated=False, device=device).reshape(-1, 4)
    save_mesh(fit_verts, faces, "out/cylinder_seg_h.obj")
    curveEdges = polylines_edges(num_segs*num_curves, device=device)
    curvePoints = sample_cubic_bezier_curve_points(control_points, num_segs, device=device).reshape(-1, 3)
    save_mesh(curvePoints, curveEdges, "out/cylinder_curve_seg_h.obj")

def main_cubic_c0_curve_segs():
    device = 'cuda'
    num_segs, points_per_circle = 5, 25
    num_curves = 50
    points = torch.linspace(0, 100, steps=3*num_curves+1, device=device).reshape(-1, 1).repeat(1,3)
    points = points.detach().requires_grad_(True)
    thickness_control_points = torch.ones(num_curves, 3, 1, device=device, requires_grad=True)
    targetVerts, targetFaces = load_mesh('models/stem_nh.obj')
    points, thickness_control_points = fit_cubic_c0_bezier_cylinder_segs(targetVerts, targetFaces, points, thickness_control_points, num_segs, points_per_circle, lr=0.1, epochs=6000, device='cuda')
    control_points = cubic_c0_curve_segments_control_points(points, device=device)
    thickness = sample_quadratic_bezier_curve_points(thickness_control_points, num_segs, device=device)
    fit_verts = cylinderical_cubic_spline_vertices(control_points, thickness, num_segs, points_per_circle, device=device).reshape(-1, 3)
    faces = cylinderical_cubic_spline_faces_seperated(num_segs, num_curves, points_per_circle, traingulated=False, device=device).reshape(-1, 4)
    save_mesh(fit_verts, faces, "out/cylinder_seg_c0.obj")
    curveEdges = polylines_edges(num_segs*num_curves, device=device)
    curvePoints = sample_cubic_bezier_curve_points(control_points, num_segs, device=device).reshape(-1, 3)
    save_mesh(curvePoints, curveEdges, "out/cylinder_curve_seg_c0.obj")


if __name__ == '__main__':
    # main_cubic_curve()
    main_cubic_c1_curve_subdiv_segs()
    # main_cubic_c0_curve_segs()