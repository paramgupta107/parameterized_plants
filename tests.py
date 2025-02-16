import unittest
import torch
import os
from mesh_ops import save_mesh
from mesh_ops import vertices_faces_to_patches
from surface_ops import write_traingle_patch_bv
from mesh_ops import gen_starting_net
from mesh_ops import save_point_cloud_to_ply
from mesh_ops import load_mesh
from surface_ops import write_tensor_product_bv
from surface_ops import sample_cubic_bezier_surfaces_points
from vector_utils import normalize_vectors
from vector_utils import rotate_vectors
from surface_ops import sample_cubic_bezier_curve_points
from surface_ops import sample_quadratic_bezier_curve_points
from surface_ops import sample_linear_bezier_curve_points
from mesh_ops import polylines_edges
from surface_ops import cubic_c1_curve_segments_control_points
from surface_ops import cubic_c0_curve_segments_control_points
from surface_ops import subdivide_c1_cubic_handles
from surface_ops import subdivide_c0_quadratic_control_points
from surface_ops import cubic_bezier_curve_curvature
from surface_ops import parallel_curves_surface_faces
from surface_ops import cubic_bezier_surface_faces
from surface_ops import sample_cubic_bspline_curve_points
from surface_ops import sample_quadratic_bspline_curve_points
from surface_ops import sample_linear_bspline_curve_points
from surface_ops import sample_cubic_bspline_loop_curve_points
class SaveMeshTests(unittest.TestCase):
    def test_save_mesh(self):
        # Define test input
        vertices = torch.Tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        faces = torch.Tensor([
            [0, 1, 2]
        ]).long()
        filename = "out/test.obj"

        # Call the function
        save_mesh(vertices, faces, filename)

        # Read the saved file
        with open(filename, 'r') as file:
            saved_content = file.read()

        # Define the expected output
        expected_content = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n"

        # Assert the saved content matches the expected output
        self.assertEqual(saved_content, expected_content)

        # Clean up the test file
        os.remove(filename)
    def test_save_mesh_extra_dim(self):
        # Define test input
        vertices = torch.Tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]).unsqueeze(0)
        faces = torch.Tensor([
            [0, 1, 2]
        ]).long()
        filename = "test.obj"

        # Call the function
        save_mesh(vertices, faces, filename)

        # Read the saved file
        with open(filename, 'r') as file:
            saved_content = file.read()

        # Define the expected output
        expected_content = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n"

        # Assert the saved content matches the expected output
        self.assertEqual(saved_content, expected_content)

        # Clean up the test file
        os.remove(filename)
    def test_save_mesh_edges(self):
        # Define test input
        vertices = torch.Tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        faces = torch.Tensor([
            [0, 1], [1, 2]
        ]).long()
        filename = "out/test.obj"

        # Call the function
        save_mesh(vertices, faces, filename)

        # Read the saved file
        with open(filename, 'r') as file:
            saved_content = file.read()

        # Define the expected output
        expected_content = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nl 1 2\nl 2 3\n"

        # Assert the saved content matches the expected output
        self.assertEqual(saved_content, expected_content)

        # Clean up the test file
        os.remove(filename)
class VerticesFacesToPatchesTestCase(unittest.TestCase):
    def test_vertices_faces_to_patches(self):
        vertices = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        faces = torch.tensor([[0, 1, 2], [1, 2, 0]]).long()
        expected_result = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]])

        result = vertices_faces_to_patches(vertices, faces)
        self.assertTrue(torch.all(torch.eq(result, expected_result)))


class WriteTraingleBVTestCase(unittest.TestCase):

    def test_degree_1(self):
        patches = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
        ])
        degree = 1
        filename = 'test.bv'
        write_traingle_patch_bv(patches, degree, filename)
        with open(filename, 'r') as file:
            content = file.read()
        expected_content = "3 1\n1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n3 1\n10.0 11.0 12.0\n13.0 14.0 15.0\n16.0 17.0 18.0\n"
        self.assertEqual(content, expected_content)
        if os.path.exists(filename):
            os.remove(filename)
    def test_incorrect_dim(self):
        patches = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
        ])
        degree = 2
        filename = 'test.bv'
        with self.assertRaises(ValueError):
            write_traingle_patch_bv(patches, degree, filename)
    def test_degree_2(self):
        patches = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
            [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0], [28.0, 29.0, 30.0], [31.0, 32.0, 33.0], [34.0, 35.0, 36.0]]
        ])
        degree = 2
        filename = 'test.bv'
        write_traingle_patch_bv(patches, degree, filename)
        with open(filename, 'r') as file:
            content = file.read()
        expected_content = "3 2\n1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n10.0 11.0 12.0\n13.0 14.0 15.0\n16.0 17.0 18.0\n3 2\n19.0 20.0 21.0\n22.0 23.0 24.0\n25.0 26.0 27.0\n28.0 29.0 30.0\n31.0 32.0 33.0\n34.0 35.0 36.0\n"
        self.assertEqual(content, expected_content)
        if os.path.exists(filename):
            os.remove(filename)

class GenStartingTestCase(unittest.TestCase):
    def test_gen_starting_net(self):
        for subdiv in range(1, 6):
            vertices, faces = gen_starting_net(radius=1, subdiv=subdiv, requires_grad=True, device='cuda')
            self.assertEqual(vertices.shape, (1, 2+2**(2+2*subdiv), 3))
            self.assertEqual(faces.shape, (2**(3+2*subdiv), 3))

class SavePointCloudToPLYTestCase(unittest.TestCase):
    def test_save_point_cloud_to_ply(self):
        point_cloud = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        filename = 'test.ply'
        save_point_cloud_to_ply(point_cloud, filename)
        with open(filename, 'r') as file:
            content = file.read()
        expected_content = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n"
        self.assertEqual(content, expected_content)
        if os.path.exists(filename):
            os.remove(filename)

class LoadMeshTestCase(unittest.TestCase):
    def test_load_mesh(self):
        filename = 'test.obj'
        with open(filename, 'w') as file:
            file.write("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n")
        vertices, faces = load_mesh(filename, device='cpu')
        expected_vertices = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        expected_faces = torch.tensor([[0, 1, 2]])
        print(vertices.shape, expected_vertices.shape)
        self.assertTrue(torch.all(torch.eq(vertices, expected_vertices)))
        self.assertEqual(vertices.shape, expected_vertices.shape)
        self.assertTrue(torch.all(torch.eq(faces, expected_faces)))
        self.assertEqual(faces.shape, expected_faces.shape)
        os.remove(filename)

class WriteTensorProductBVTestCase(unittest.TestCase):
    def test_degree_3(self):
        patches = torch.tensor([[[0.0,0.0,0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 2.0], [0.0, 1.0, 3.0], [0.0,2.0,0.0], [0.0, 2.0, 1.0], [0.0, 2.0, 2.0], [0.0, 2.0, 3.0], [0.0, 3.0, 0.0], [0.0, 3.0, 1.0], [0.0, 3.0, 2.0], [0.0, 3.0, 3.0]]])
        degree = 3
        filename = 'test.bv'
        write_tensor_product_bv(patches, degree, filename)
        with open(filename, 'r') as file:
            content = file.read()
        expected_content = "4 3\n0.0 0.0 0.0\n0.0 0.0 1.0\n0.0 0.0 2.0\n0.0 0.0 3.0\n0.0 1.0 0.0\n0.0 1.0 1.0\n0.0 1.0 2.0\n0.0 1.0 3.0\n0.0 2.0 0.0\n0.0 2.0 1.0\n0.0 2.0 2.0\n0.0 2.0 3.0\n0.0 3.0 0.0\n0.0 3.0 1.0\n0.0 3.0 2.0\n0.0 3.0 3.0\n"
        self.assertEqual(content, expected_content)
        if os.path.exists(filename):
            os.remove(filename)

class SampleCubicBeizerSurfacerTestCase(unittest.TestCase):
    def test_sample_points(self):
        patches = torch.rand(2, 16, 3)
        num_points_per_dim = 10
        points = sample_cubic_bezier_surfaces_points(patches, num_points_per_dim, device='cpu')
        self.assertEqual(points.shape, (2, num_points_per_dim*num_points_per_dim, 3))

class SampleCubicBezierCurvePointsTestCase(unittest.TestCase):
    def test_sample_points(self):
        control_points = torch.rand(2, 4, 3)
        num_points = 10
        points = sample_cubic_bezier_curve_points(control_points, num_points, device='cpu')
        self.assertEqual(points.shape, (2, num_points, 3))

class SampleQuadraticBezierCurvePointsTestCase(unittest.TestCase):
    def test_sample_points(self):
        control_points = torch.rand(2, 3, 3)
        num_points = 10
        points = sample_quadratic_bezier_curve_points(control_points, num_points, device='cpu')
        self.assertEqual(points.shape, (2, num_points, 3))

class SampleLinearBezierCurvePointsTestCase(unittest.TestCase):
    def test_sample_points(self):
        control_points = torch.rand(2, 2, 3)
        num_points = 10
        points = sample_linear_bezier_curve_points(control_points, num_points, device='cpu')
        self.assertEqual(points.shape, (2, num_points, 3))
class TestRotateVectors(unittest.TestCase):
    def test_rotate_vectors(self):
        vectors = torch.tensor([[0.8, 2.93, 1.53], [0.0, 1.0, 0.0], [1.0, 1.0, 5.0]])
        angles = torch.tensor([0.9, 7.0, 7.5])
        axes = torch.tensor([[1.0, 2.1, 1.2], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        expected_result = torch.tensor([[0.8921,2.67027,1.90778], [0.0, 1.0, 0.0], [5.03664,1,0.79518]])
        result = rotate_vectors(vectors, angles, axes)

        self.assertTrue(torch.allclose(result, expected_result))

class TestNormalizeVectors(unittest.TestCase): 
    def test_normalize_vectors(self): 
        vectors = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]) 
        expected_result = torch.tensor([[0.26726124, 0.53452248, 0.80178373], [0.45584231, 0.56980288, 0.68376346], [0.50257071, 0.57436653, 0.64616235]]) 
        result = normalize_vectors(vectors) 
        self.assertTrue(torch.allclose(result, expected_result))
    def test_nomralize_vecors_high_dem(self):
        vectors = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]]) 
        expected_result = torch.tensor([[[0.26726124, 0.53452248, 0.80178373], [0.45584231, 0.56980288, 0.68376346]], [[0.50257071, 0.57436653, 0.64616235], [0.50257071, 0.57436653, 0.64616235]]])
        result = normalize_vectors(vectors)
        self.assertTrue(torch.allclose(result, expected_result))

class TestPolylineEdges(unittest.TestCase):
    def test_polyline_edges(self):
        numSegs = 4
        expected_result = torch.tensor([[0, 1], [1, 2], [2, 3]])
        result = polylines_edges(numSegs, device='cpu')
        self.assertTrue(torch.allclose(result, expected_result))
        self.assertEqual(result.shape, expected_result.shape)
        self.assertTrue(result.dtype == expected_result.dtype)

class TestCubicCurveSegments(unittest.TestCase):
    def test_cubic_curve_segments_control_points(self):
        handles = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]])
        expected = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
        result = cubic_c1_curve_segments_control_points(handles, device='cpu')
        self.assertEqual(result.shape, (1, 4, 3))
        self.assertTrue(torch.allclose(result, expected))
    def test_cubic_curve_segments_control_points_extra_dim(self):
        handles = torch.tensor([[[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]], [[[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]], [[30.0, 30.0, 30.0], [50.0, 50.0, 50.0]]]])
        expected = torch.tensor([[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]], [[[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0], [40.0, 40.0, 40.0]]]])
        result = cubic_c1_curve_segments_control_points(handles, device='cpu')
        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(torch.allclose(result, expected))

class TestCubicC0CurveSegments(unittest.TestCase):
    def test_cubic_c0_curve_segments_control_points(self):
        points = torch.Tensor([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7]])
        expected = torch.Tensor([[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]], [[4., 4., 4.], [5., 5., 5.], [6., 6., 6.], [7., 7., 7.]]])
        result = cubic_c0_curve_segments_control_points(points, device = 'cpu')
        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(torch.allclose(result, expected))

class TestSubdivideC1CubicHandles(unittest.TestCase):
    def test_subdivide_c1_cubic_handles(self):
        handles = torch.tensor([[[1,1,1],[2,2,2]], [[3,3,3],[4,4,4]]])
        subdivided_handles = subdivide_c1_cubic_handles(handles, device='cpu')
        expected = torch.tensor([[[1.2500, 1.2500, 1.2500],
         [1.7500, 1.7500, 1.7500]],

        [[2.1250, 2.1250, 2.1250],
         [2.8750, 2.8750, 2.8750]],

        [[3.2500, 3.2500, 3.2500],
         [3.7500, 3.7500, 3.7500]]])
        self.assertTrue(torch.allclose(subdivided_handles, expected))
        self.assertEqual(subdivided_handles.shape, expected.shape)
    
    def test_subdivide_c1_cubic_handles_extra_dim(self):
        handles = torch.tensor([[[[1,1,1],[2,2,2]], [[3,3,3],[4,4,4]]], [[[10,10,10],[20,20,20]], [[30,30,30],[40,40,40]]]])
        subdivided_handles = subdivide_c1_cubic_handles(handles, device='cpu')
        expected = torch.tensor([[[[1.2500, 1.2500, 1.2500],
         [1.7500, 1.7500, 1.7500]],

        [[2.1250, 2.1250, 2.1250],
         [2.8750, 2.8750, 2.8750]],

        [[3.2500, 3.2500, 3.2500],
         [3.7500, 3.7500, 3.7500]]],
         [[[12.500, 12.500, 12.500],
         [17.500, 17.500, 17.500]],

        [[21.250, 21.250, 21.250],
         [28.750, 28.750, 28.750]],

        [[32.500, 32.500, 32.500],
         [37.500, 37.500, 37.500]]]])
        self.assertTrue(torch.allclose(subdivided_handles, expected))
        self.assertEqual(subdivided_handles.shape, expected.shape)

class TestSubdivideC0ControlPoints(unittest.TestCase):
    def test_subdivide_c0_quadratic_control_points(self):
        control_points = torch.tensor([[[1,1,1],[2,2,2],[3,3,3]], [[4,4,4],[5,5,5],[6,6,6]]])
        subdivided_control_points = subdivide_c0_quadratic_control_points(control_points, device='cpu')
        expected = torch.tensor([[[1.0000, 1.0000, 1.0000],
         [1.5000, 1.5000, 1.5000],
         [2.0000, 2.0000, 2.0000]],

        [[2.0000, 2.0000, 2.0000],
         [2.5000, 2.5000, 2.5000],
         [3.0000, 3.0000, 3.0000]],

        [[4.0000, 4.0000, 4.0000],
         [4.5000, 4.5000, 4.5000],
         [5.0000, 5.0000, 5.0000]],

        [[5.0000, 5.0000, 5.0000],
         [5.5000, 5.5000, 5.5000],
         [6.0000, 6.0000, 6.0000]]])
        self.assertTrue(torch.allclose(subdivided_control_points, expected))
        self.assertEqual(subdivided_control_points.shape, expected.shape)

class TestCubicBezierCurvature(unittest.TestCase):
    def test_cubic_bezier_curve_curvature(self):
        control_points = torch.tensor([[1.0,1,1],[2,2,2],[3,3,3],[4,4,4]])
        curvature = cubic_bezier_curve_curvature(control_points, 5, device='cpu')
        expected = torch.tensor(0.0)
        self.assertTrue(torch.allclose(curvature, expected))
        self.assertEqual(curvature.shape, expected.shape)
    def test_cubic_bezier_curve_curvature_extra_dim(self):
        control_points = torch.tensor([[[1.0,1,1],[2,2,2],[3,3,3],[4,4,4]], [[10.0,10,10],[20,20,20],[30,30,30],[40,40,40]]])
        curvature = cubic_bezier_curve_curvature(control_points, 5, device='cpu')
        expected = torch.tensor(0.0)
        self.assertTrue(torch.allclose(curvature, expected))
        self.assertEqual(curvature.shape, expected.shape)

class TestParallelCurvesSurfaceFaces(unittest.TestCase):
    def test_parallel_curves_surface_faces(self):
        faces = parallel_curves_surface_faces(5, 3, device='cpu')
        expected = torch.tensor([[ 0,  5,  6],
        [ 1,  6,  7],
        [ 2,  7,  8],
        [ 3,  8,  9],
        [ 0,  6,  1],
        [ 1,  7,  2],
        [ 2,  8,  3],
        [ 3,  9,  4],
        [ 5, 10, 11],
        [ 6, 11, 12],
        [ 7, 12, 13],
        [ 8, 13, 14],
        [ 5, 11,  6],
        [ 6, 12,  7],
        [ 7, 13,  8],
        [ 8, 14,  9]])
        self.assertTrue(torch.allclose(faces, expected))
        self.assertEqual(faces.shape, expected.shape)

class TestCubicBezierSurfaceFaces(unittest.TestCase):
    def test_cubic_bezier_surface_faces(self):
        faces = cubic_bezier_surface_faces(4, device='cpu')
        expected = torch.tensor([[ 0,  4,  5],
        [ 1,  5,  6],
        [ 2,  6,  7],
        [ 0,  5,  1],
        [ 1,  6,  2],
        [ 2,  7,  3],
        [ 4,  8,  9],
        [ 5,  9, 10],
        [ 6, 10, 11],
        [ 4,  9,  5],
        [ 5, 10,  6],
        [ 6, 11,  7],
        [ 8, 12, 13],
        [ 9, 13, 14],
        [10, 14, 15],
        [ 8, 13,  9],
        [ 9, 14, 10],
        [10, 15, 11]])
        self.assertTrue(torch.allclose(faces, expected))
        self.assertEqual(faces.shape, expected.shape)
class SampleCubicBsplineCurvePoints(unittest.TestCase):
    def test_sample_cubic_bspline_curve_points(self):
        control_points = torch.zeros(2, 4, 18, 3)
        num_points = 10
        points = sample_cubic_bspline_curve_points(control_points, num_points, device='cpu')
        self.assertEqual(points.shape, (2, 4, (18-3)*10, 3))

class SampleQuadraticBsplineCurvePoints(unittest.TestCase):
    def test_sample_quadratic_bspline_curve_points(self):
        control_points = torch.zeros(2, 3, 18, 3)
        num_points = 10
        points = sample_quadratic_bspline_curve_points(control_points, num_points, device='cpu')
        self.assertEqual(points.shape, (2, 3, (18-2)*10, 3))

class SampleLinearBsplineCurvePoints(unittest.TestCase):
    def test_sample_linear_bspline_curve_points(self):
        control_points = torch.zeros(2, 3, 18, 3)
        num_points = 10
        points = sample_linear_bspline_curve_points(control_points, num_points, device='cpu')
        self.assertEqual(points.shape, (2, 3, (18-1)*10, 3))

class SampleCubicBsplineLoopCurvePoints(unittest.TestCase):
    def test_sample_cubic_bspline_loop_curve_points(self):
        control_points = torch.zeros(2, 4, 18, 3)
        num_points = 10
        points = sample_cubic_bspline_loop_curve_points(control_points, num_points, device='cpu')
        self.assertEqual(points.shape, (2, 4, 18*10, 3))

if __name__ == '__main__':
    unittest.main()