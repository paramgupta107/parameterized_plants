import torch
import numpy as np
import kaolin
from vector_utils import rotate_vectors
from visualizer import Visualizer
from surface_ops import sample_cubic_bezier_surfaces_points

def parameterized_leaf(parametes: torch.Tensor, device: str = 'cuda', use_experimental_params: bool = False) -> torch.Tensor:
    """
    Generate a leaf tensor product bezier patches from the given parameters.

    Args:
        parametes (torch.Tensor): The parameters of the leaf of shape (num_leaves, 17).
        device (str, optional): The device to use for computation.
        use_experimental_params (bool, optional): Whether to use experimental parameters. Defaults to False.

    Returns:
        torch.Tensor: The control points of the leaf patches of shape (num_leaves, 16, 3).
    """
    parametes = parametes.to(device)
    tilt_up = parametes[:, 0].unsqueeze(1)
    tilt_side = parametes[:, 1].unsqueeze(1)
    len_ = parametes[:, 2].unsqueeze(1)
    width = parametes[:, 3].unsqueeze(1)
    cur = parametes[:, 4].unsqueeze(1)
    dip = parametes[:, 5].unsqueeze(1)
    dip_width = parametes[:, 6].unsqueeze(1)
    cur_width = parametes[:, 7].unsqueeze(1)
    curv_offset = parametes[:, 8].unsqueeze(1)
    if use_experimental_params:
        end_width = parametes[:, 9].unsqueeze(1)
        end_curv = parametes[:, 10].unsqueeze(1)
        end_curvWidth = parametes[:, 11].unsqueeze(1)
        xx = parametes[:, 12].unsqueeze(1)
    else:
        end_width = 0
        end_curv = 0
        end_curvWidth = 0
        xx = 0
    curv_offsetstart = parametes[:, 13].unsqueeze(1)
    curv_offsetend = parametes[:, 14].unsqueeze(1)
    w_twist_start = parametes[:, 15].unsqueeze(1)
    w_twist_end = parametes[:, 16].unsqueeze(1)

    u = torch.stack([torch.sin(tilt_up) * torch.cos(tilt_side), torch.sin(tilt_up) * torch.sin(tilt_side), torch.cos(tilt_up)], dim=1)
    v = torch.stack([torch.sin(tilt_up - (np.pi / 2)) * torch.cos(tilt_side), torch.sin(tilt_up - (np.pi / 2)) * torch.sin(tilt_side), torch.cos(tilt_up - (np.pi / 2))], dim=1)
    w = torch.cross(u, v)

    w_start = rotate_vectors(w, w_twist_start, u)
    w_end = rotate_vectors(w, w_twist_end, u)

    top = u * curv_offset * len_ + v * cur
    end = u * len_

    A = torch.zeros_like(top)
    B = torch.zeros_like(top)
    C = torch.zeros_like(top)
    D = torch.zeros_like(top)

    E = top - w_start * width - u * len_ * cur_width * 0.5 + v * curv_offsetstart
    F = top - w_start * dip_width * width - u * len_ * cur_width * 0.5 - v * cur * dip + v * curv_offsetstart
    G = top + w_start * dip_width * width - u * len_ * cur_width * 0.5 - v * cur * dip + v * curv_offsetstart
    H = top + w_start * width - u * len_ * cur_width * 0.5 + v * curv_offsetstart
    I = top - w_end * width + u * len_ * cur_width * 0.5 + v * curv_offsetend
    J = top - w_end * dip_width * width + u * len_ * cur_width * 0.5 - v * cur * dip + v * curv_offsetend
    K = top + w_end * dip_width * width + u * len_ * cur_width * 0.5 - v * cur * dip + v * curv_offsetend
    L = top + w_end * width + u * len_ * cur_width * 0.5 + v * curv_offsetend
    M = end - w * width * end_width - u * 0.5 * len_ * (1 - cur_width) * end_curv + v * xx * cur
    N = end - w * width * end_width * end_curvWidth
    O = end + w * width * end_width * end_curvWidth
    P = end + w * width * end_width - u * 0.5 * len_ * (1 - cur_width) * end_curv + v * xx * cur

    return torch.stack([A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P], dim=1)

def get_target_leaf_params():
    """
    Get the target leaf parameters.

    Returns:
        torch.Tensor: The target leaf parameters of shape (1, 17).
    """
    return torch.tensor([[0.97, 1.45, 3.9, 0.66, 0.98, -0.38, 0, 0.55, 0.36, 0, 0, 0, 0, 0.03, -0.56, 0.618407346410207, -0.181592653589793]])

def get_random_deviation_leaf(magnitude: float, parameters: torch.Tensor, device: str = 'cuda'):
    """
    Get a random deviation from starting leaves.

    Args:
        magnitude (float): The magnitude of the deviation.
        parameters (torch.Tensor): The starting leaf parameters to generate deviations from of shape (num_leaves, 17).
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The deviated leaf parameters of shape (num_leavs, 17).
    """
    return (parameters + torch.randn_like(parameters) * magnitude * torch.tensor([[np.pi, np.pi, 5, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, np.pi/3, np.pi/3]])).to(device)

def generate_sample_timelapse_data(num_steps: int, max_noise_magnitude: float, logs_path: str = 'leaf_logs/sample_log.json'):
    """
    Generate sample timelapse data for the leaf optimization.

    Args:
        num_steps (int): The number of optimization steps.
        max_noise_magnitude (float): The maximum noise magnitude.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.
    """
    visualizer = Visualizer(logs_path)
    targetParams = get_target_leaf_params()
    visualizer.add_target(targetParams)
    randomParams = get_random_deviation_leaf(max_noise_magnitude, targetParams)
    for i in range(num_steps):
        u = i / (num_steps-1)
        visualizer.add_step(targetParams * u + randomParams * (1 - u), i*10)
    visualizer.save()

def clip_parameters(params: torch.Tensor, device = 'cuda'):
    """
    Clip the leaf parameters. Note that the clipping process is done with torch.no_grad().

    Args:
        params (torch.Tensor): The leaf parameters of shape (1, 17).
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Returns:
        torch.Tensor: The clipped leaf parameters of shape (1, 17)
    """
    minVals =  torch.tensor([[0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -np.pi/3, -np.pi/3]]).to(device)
    maxVals =  torch.tensor([[np.pi, np.pi, 5, 2, 1, 1, 1, 1, 1, 0.001, 0.001, 0.001, 0.001, 1, 1, np.pi/3, np.pi/3]]).to(device)
    shift = (params - minVals) / (maxVals - minVals)
    # print("params", params)
    # print("sub", params - minVals)
    # print("divisor", maxVals - minVals)
    # print("div", (params - minVals) / (maxVals - minVals))
    # print("shift", shift)
    clipped =  torch.sigmoid(shift) * (maxVals - minVals) + minVals
    # print("clipped", clipped)
    return clipped


def fit_leaf(targetLeaf: torch.Tensor, startingLeaf: torch.Tensor, epochs: int = 1000, lr: float = 0.1, device: str = 'cuda', logs_path: str = 'leaf_logs/log.json', use_experimental_params: bool = False, eps = 0.0000001, clip_params: bool = True) -> torch.Tensor:
    """
    Fits a leaf to a target leaf using optimization.

    Args:
        targetLeaf (torch.Tensor): The target leaf parameters of shape (1, 17).
        startingLeaf (torch.Tensor): The starting leaf parameters of shape (1, 17).
        epochs (int, optional): The maximum number of optimization epochs. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.
        use_experimental_params (bool, optional): Whether to use experimental parameters. Defaults to False.
        eps (float, optional): The epsilon value for when to stop optimizing. Defaults to 0.0000001.
        clip_params (bool, optional): Whether to clip the parameters. Defaults to True.

    Returns:
        torch.Tensor: The fitted leaf parameters of shape (1, 17).
    """
    targetPointCloud = sample_cubic_bezier_surfaces_points(parameterized_leaf(targetLeaf), 100)
    timelapse = Visualizer(logs_path)
    optimizer = torch.optim.Adam([startingLeaf], lr=lr)
    epoch = 0
    prevLoss = 0
    while True:
        if clip_params:
            clippedParams = clip_parameters(startingLeaf)
        else:
            clippedParams = startingLeaf
        currentPointCloud = sample_cubic_bezier_surfaces_points(parameterized_leaf(clippedParams, use_experimental_params=use_experimental_params), 100)
        loss = kaolin.metrics.pointcloud.chamfer_distance(currentPointCloud, targetPointCloud)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")
        if epoch % 10 == 0:
            timelapse.add_step(clippedParams, epoch, loss.item())
        if epoch % 100 == 0:
            if abs(prevLoss - loss.item()) < eps:
                break
            prevLoss = loss.item()
        epoch += 1
        if epoch >= epochs:
            break
    print("Optimization finished.")
    timelapse.add_target(targetLeaf)
    timelapse.save()
    return startingLeaf

def main():
    targetLeaf = get_target_leaf_params()
    for dev in [0.1, 0.4, 0.8, 1.2, 1.6, 2.0]:
        for clip in [True, False]:
            startingLeaf = get_random_deviation_leaf(dev, targetLeaf)
            if clip:
                startingLeaf = clip_parameters(startingLeaf)
            startingLeaf.requires_grad_()
            fit_leaf(targetLeaf, startingLeaf, epochs=5000, logs_path='leaf_logs/{}{}_log.json'.format(dev, '_clipped' if clip else ''), clip_params=clip)

if __name__ == "__main__":
    main()