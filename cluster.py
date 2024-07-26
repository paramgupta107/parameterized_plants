import os
import kaolin
from sklearn.cluster import SpectralClustering
import numpy as np
from mesh_ops import load_mesh

def save_point_cloud_to_ply_colored(point_cloud: np.ndarray, cluster: np.ndarray, filename: str):
    """
    Save a point cloud to a PLY file with red color.

    Args:
        point_cloud (np.ndarray): Point cloud coordinates of shape (num_points, 3).
        cluster (np.ndarray): Cluster indices of shape (num_points,).
        filename (str): Name of the output PLY file.

    Returns:
        None
    """
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (255, 192, 203),# Pink
        (0, 255, 127)   # Lime
    ]
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
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        
        # Write point cloud vertices with red color
        for idx, point in enumerate(point_cloud):
            color = colors[cluster[idx] % len(colors)]
            file.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

# Example usage:
# point_cloud = torch.rand((100, 3))  # Replace with your actual point cloud data
# save_point_cloud_to_ply(point_cloud, 'output_red.ply')

if __name__ == "__main__":
    targetVerts, targetFaces = load_mesh("models/plant.obj")
    targetPointCloud, _ = kaolin.ops.mesh.sample_points(targetVerts, targetFaces, 10000)
    points = targetPointCloud.reshape(-1, 3).cpu().numpy()
    clustering = SpectralClustering(n_clusters=15, degree=5,
        assign_labels='discretize',
        random_state=0, verbose=True).fit(points)
    print(clustering.labels_)
    save_point_cloud_to_ply_colored(points, clustering.labels_, 'out/output_red.ply')
