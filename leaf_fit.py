import argparse
import torch
import kaolin
from parametersized_leaf import clip_parameters, get_target_leaf_params, parameterized_leaf
from surface_ops import cubic_bezier_surface_faces, sample_cubic_bezier_surfaces_points, subdivide_c1_cubic_handles, cubic_c1_curve_segments_control_points, sample_cubic_bezier_curve_points, parallel_curves_surface_faces, cubic_bezier_curve_curvature, write_tensor_product_bv
from mesh_ops import laplace_regularizer_const, load_mesh, save_mesh

def fit_cubic_c1_bezier_cylinder_subidvided_segs(targetMeshVertices: torch.Tensor, targetMeshFaces: torch.LongTensor, handles: torch.Tensor, num_segs: int, epochs: int = 1000, lr: float = 0.01, device: str = 'cuda', logs_path: str = 'logs_leaf') -> torch.Tensor:
    """
    Fits a cubic bezier cylinder segments to a target mesh using optimization.

    Args:
        targetMeshVertices (torch.Tensor): The vertices of the target mesh of shape (1, num_vertices, 3).
        targetMeshFaces (torch.LongTensor): The faces of the target mesh of shape (num_faces, 3).
        handles (torch.Tensor): The handles of the cubic Bezier curve segments of shape (num_veins, num_curves+1, 2, 3).
        num_segs (int): The number of segments.
        epochs (int, optional): The number of optimization epochs. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.

    Returns:
        torch.Tensor: The fitted handles of the cubic Bezier curve segments of shape (num_subdivided_curves+1, 2, 3).
    """
    timelapse = kaolin.visualize.Timelapse(logs_path)
    timelapse.add_mesh_batch(category='target',
                         faces_list=[targetMeshFaces.cpu()],
                         vertices_list=[targetMeshVertices.cpu()])
    targetPointCloud, _ = kaolin.ops.mesh.sample_points(targetMeshVertices, targetMeshFaces, 10000)
    num_veins = handles.shape[0]
    num_curves_per_vein = handles.shape[1]-1
    num_segs_per_vein = num_segs * num_curves_per_vein
    startingFaces = parallel_curves_surface_faces(num_segs_per_vein, num_veins, device=device)
    transform = torch.eye(3, device=device, requires_grad=True)
    translate = torch.zeros(1, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform, translate], lr=lr)
    numSubdivisions = 0
    for epoch in range(epochs):
        if epoch == 100:
            optimizer = torch.optim.Adam([transform, translate, handles], lr=lr/10)
            print("------------------Optimizing handles.------------------")
        if epoch == 500:
            optimizer = torch.optim.Adam([transform, translate, handles], lr=lr/100)
            print("------------------Optimizing thickness.------------------")
        if epoch > 500 and (epoch % 500 == 0) and numSubdivisions < 3:
            handles = subdivide_c1_cubic_handles(handles, device=device).detach().requires_grad_(True)
            num_veins = handles.shape[0]
            num_curves_per_vein = handles.shape[1]-1
            num_segs_per_vein = num_segs * num_curves_per_vein
            startingFaces = parallel_curves_surface_faces(num_segs_per_vein, num_veins, device=device)
            numSubdivisions += 1
            optimizer = torch.optim.Adam([transform, translate, handles], lr=lr/100)
            print(f"Subdivisions: {numSubdivisions}")
        movedHandles = torch.matmul(handles, transform) + translate
        control_points = cubic_c1_curve_segments_control_points(movedHandles, device=device)
        # thickness = sample_quadratic_bezier_curve_points(thickness_control_points, num_segs, device=device)
        startingVertices = sample_cubic_bezier_curve_points(control_points, num_segs, device=device).reshape(1, -1, 3)
        
        startingPointCloud, _ = kaolin.ops.mesh.sample_points(startingVertices, startingFaces, 10000)
        chamfer_loss = kaolin.metrics.pointcloud.chamfer_distance(startingPointCloud, targetPointCloud)
        curvature_loss = cubic_bezier_curve_curvature(control_points, num_segs, device=device)
        loss = chamfer_loss# + 0.01 * curvature_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()} \t| Chamfer Loss {chamfer_loss.item()} \t| Curvature Loss {curvature_loss.item()}")
        if epoch % 10 == 0:
            timelapse.add_mesh_batch(
            iteration=epoch+1,
            category='fitted_mesh',
            vertices_list=[startingVertices.cpu()],
            faces_list=[startingFaces.cpu()]
            )
    print("Optimization finished.")
    return torch.matmul(handles, transform) + translate

def leaf_fit_main(file_path: str) -> None:
    device = 'cuda'
    num_segs = 20
    targetVerts, targetFaces = load_mesh(file_path)
    # handles = torch.tensor([[[[-2, 0, 0], [-1.8, 4, 0]], [[-1.6, 8, 0], [-1.4, 12, 0]]], [[[-1, 0, 0], [-0.8, 4, 0]], [[-0.6, 8, 0], [-0.4, 14, 0]]], [[[0,0,0], [0,4,0]], [[0,8,0], [0,16,0]]],
    #                     [[[1, 0, 0], [0.8, 4, 0]], [[0.6, 8, 0], [0.4, 14, 0]]], [[[2, 0, 0], [1.8, 4, 0]], [[1.6, 8, 0], [1.4, 12, 0]]]], device=device, dtype=torch.float32, requires_grad=True)
    
    handles = torch.tensor([[[[-2, 0, 0], [-1.8, 4, 0]], [[-1.6, 8, 0], [-1.4, 12, 0]]], [[[0,0,0], [0,4,0]], [[0,8,0], [0,16,0]]],
                        [[[2, 0, 0], [1.8, 4, 0]], [[1.6, 8, 0], [1.4, 12, 0]]]], device=device, dtype=torch.float32, requires_grad=True)
    handles = fit_cubic_c1_bezier_cylinder_subidvided_segs(targetVerts, targetFaces, handles, num_segs, lr=0.1, epochs=6000, device='cuda')

    num_veins = handles.shape[0]
    num_curves_per_vein = handles.shape[1]-1
    num_segs_per_vein = num_segs * num_curves_per_vein
    control_points = cubic_c1_curve_segments_control_points(handles, device)
    points = sample_cubic_bezier_curve_points(control_points, num_segs, device).reshape(-1, 3)
    faces = parallel_curves_surface_faces(num_segs_per_vein, num_veins, device)

    save_mesh(points, faces, "out/leaf.obj")

def fit_paramterized_leaf(targetMeshVertices: torch.Tensor, targetMeshFaces: torch.LongTensor, epochs: int = 1000, lr: float = 0.1, device: str = 'cuda', logs_path: str = 'parameterized_leaf_logs', use_experimental_params: bool = False, clip_params: bool = True) -> torch.Tensor:
    """
    Fits a leaf to a target leaf using optimization.

    Args:
        targetMeshVertices (torch.Tensor): The vertices of the target mesh of shape (1, num_vertices, 3).
        targetMeshFaces (torch.LongTensor): The faces of the target mesh of shape (num_faces, 3).
        epochs (int, optional): The maximum number of optimization epochs. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'parameterized_leaf_logs'.
        use_experimental_params (bool, optional): Whether to use experimental parameters. Defaults to False.
        clip_params (bool, optional): Whether to clip the parameters. Defaults to True.

    Returns:
        torch.Tensor: The fitted leaf parameters of shape (1, 17).
    """
    startingLeaf = get_target_leaf_params(device=device)
    startingLeaf.requires_grad_()
    targetPointCloud, _ = kaolin.ops.mesh.sample_points(targetMeshVertices, targetMeshFaces, 10000)
    transform = torch.eye(3, device=device, requires_grad=True)
    translate = torch.zeros(1, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([transform, translate], lr=lr)
    for epoch in range(1000):
        if epoch == 200:
            optimizer = torch.optim.Adam([transform, translate, startingLeaf], lr=lr/10)
            print("------------------Optimizing parameters.------------------")
        if clip_params:
            clippedParams = clip_parameters(startingLeaf)
        else:
            clippedParams = startingLeaf
        currentPointCloud = sample_cubic_bezier_surfaces_points(parameterized_leaf(clippedParams, use_experimental_params=use_experimental_params), 100)
        movedPointCloud = torch.matmul(currentPointCloud, transform) + translate
        loss = kaolin.metrics.pointcloud.chamfer_distance(movedPointCloud, targetPointCloud)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Parameter Optimization | Epoch {epoch}: Loss {loss.item()}")
    print("Optimization finished.")
    vertices = sample_cubic_bezier_surfaces_points(parameterized_leaf(clippedParams, use_experimental_params=use_experimental_params), 50)
    vertices = (torch.matmul(vertices, transform) + translate).detach().requires_grad_(True)
    faces = cubic_bezier_surface_faces(50, device)
    optimizer = torch.optim.Adam([vertices], lr=lr/10000)
    for epoch in range(1000, epochs):
        break
        currentPointCloud, _ = kaolin.ops.mesh.sample_points(vertices, faces, 10000)
        loss = kaolin.metrics.pointcloud.chamfer_distance(currentPointCloud, targetPointCloud) + 10000 * laplace_regularizer_const(vertices.squeeze(), faces)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Mesh Optimization | Epoch {epoch}: Loss {loss.item()}")
    print("Optimization finished.")
    return vertices, faces

def parameterized_fit_main(file_path: str) -> None:
    device = 'cuda'
    targetVerts, targetFaces = load_mesh(file_path)
    points, faces = fit_paramterized_leaf(targetVerts, targetFaces, epochs=6000, lr=1, device=device, logs_path='parameterized_leaf_logs', use_experimental_params=False, clip_params=True)
    save_mesh(points, faces, "out/leaf_param_1.obj")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit to a stem obj file")

    # Add the file path argument
    parser.add_argument(
        'file_path', 
        nargs='?', 
        type=str, 
        default='models/leaf.obj', 
        help="Path to the OBJ file"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get the file path
    file_path = args.file_path
    
    # leaf_fit_main(file_path)
    parameterized_fit_main(file_path)