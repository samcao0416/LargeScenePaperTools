import open3d as o3d
import numpy as np
import os
import cv2
import pymeshlab as ml
from copy import deepcopy

class MeshImageFilter():

    def __init__(self, mesh_path, load_mesh = True, simplify = True, D_view_mask = None, depth_diff_value = 0.5):

        self.mesh_path = mesh_path

        if load_mesh:
            self.load_mesh(simplify = simplify)
        else:
            print("[ INFO ] Mesh is not loaded. May lead to following program crash.")
            self.mesh = None

        self.D_view_mask = D_view_mask

        self.raycast_scene = None
        self.depth_diff_value = depth_diff_value
        self.raycastscene = self.RaycastScene(self)
        self.meshprocess = self.MeshProcess(self)
        self.renderimg = self.RenderImg(self)

    class RaycastScene():
        def __init__(self, outerclass):
            self.st = outerclass
            self.InitSceneWithMesh()

        
        def InitSceneWithMesh(self):
            # Convert the mesh to a tensor
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.st.mesh)

            # Setup RaycastScene
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(t_mesh)
            self.st.raycast_scene = scene

        def CastScene(self, intrinsic: np.array = None, extrinsic: np.array = None, width: int = None, height: int = None):

            if type(width) != int or type(height) != int:
                raise TypeError("Width and Height must be int.")
            
            # Set camera parameters
            intrinsic = o3d.core.Tensor(intrinsic, dtype=o3d.core.float64)
            extrinsic = o3d.core.Tensor(np.linalg.inv(extrinsic), dtype=o3d.core.float64)
            self.rays = self.st.raycast_scene.create_rays_pinhole(intrinsic, extrinsic, width, height)

            # Cast rays
            result = self.st.raycast_scene.cast_rays(self.rays)

            self.cast_result = result
            

        def triangles_per_view(self):

            if self.st.mesh is None:
                raise RuntimeError("Mesh is not loaded.")

            if self.st.raycast_scene is None:
                raise RuntimeError("Raycast Scene is not initialized, Run RaycastScene.CastScene() first.")
            
            visible_indices = self.cast_result["primitive_ids"].numpy().flatten()
            visible_indices, _ = np.unique(visible_indices, return_index = True)
            filtered_indices = visible_indices[np.where(visible_indices < len(self.st.mesh.triangles))[0]]

            self.filtered_indices = filtered_indices
            return filtered_indices

        def save_visible_triangles(self, path = None):
            
            vertices = np.asarray(deepcopy(self.st.mesh.vertices))
            colors = np.asarray(deepcopy(self.st.mesh.vertex_colors))
            triangles = np.asarray(deepcopy(self.st.mesh.triangles))[self.filtered_indices, :]

            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            new_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            new_mesh.triangles = o3d.utility.Vector3iVector(triangles)

            if path is None:
                path = os.path.join(os.path.dirname(self.st.mesh_path), "visible_triangles.ply") 
            o3d.io.write_triangle_mesh(path, new_mesh, True)
            print("[ INFO ] Visible triangles saved to {}".format(path))

        def save_visible_depth_pcd(self, path:str = None):

            depth = self.cast_result["t_hit"].numpy()
            depth = np.where(depth == np.inf, 0, depth)
            rays = self.rays.numpy()
            point = rays[...,0:3] + rays[..., 3:] * depth[..., None]
            point = point.reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point)
            if path is None:
                path = os.path.join(os.path.dirname(self.st.mesh_path), "depth_pointcloud.ply") 
            o3d.io.write_point_cloud(path, pcd)
            print("[ INFO ] Dpeth Pointcloud saved to {}".format(path))

        def save_visible_depth_img(self, path: str):
            
            depth = self.cast_result["t_hit"].numpy()
            # depth_mask = np.where(depth != np.inf, 255, depth)
            depth_mask = np.where(depth == np.inf, 0, depth)
            depth_mask = np.where(depth_mask > 0, 255, 0)
            depth = np.where(depth == np.inf, 0, depth)
            depth = np.where(depth < 10.0, depth / 10.0, 2.0 - 10 / depth) / 2.0
            depth *= 255

            bgr_depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            bgra_depth = cv2.merge((bgr_depth, depth_mask.astype(np.uint8)))
            cv2.imwrite(path, bgra_depth)

        def save_mask_depth(self, path: str, raw_depth_img: str = None):

            mask = self.cast_result["t_hit"].numpy()

            if raw_depth_img is None:
                mask = np.where(mask != np.inf, 255, mask)
                mask = np.where(mask == np.inf, 0, mask)
            else:
                raw_depth  = cv2.imread(raw_depth_img, cv2.IMREAD_GRAYSCALE)
                new_depth = np.where(mask == np.inf, 0, mask)
                new_depth = np.where(new_depth < 10.0, new_depth / 10.0, 2.0 - 10 / new_depth) / 2.0
                new_depth *= 255
                
                # for i in range(raw_depth.shape[0]):
                #     for j in range(raw_depth.shape[1]):
                #         if raw_depth[i][j] == 0 or new_depth[i][j] == 0:
                #             mask[i][j] = 0
                #         elif abs(raw_depth[i][j] - new_depth[i][j]) >= 0.01:
                #             mask[i][j] = 0
                #         else:
                #             mask[i][j] = 255
                # Create a mask where either raw_depth or new_depth is 0
                mask_zero = np.logical_or(raw_depth == 0, new_depth == 0)

                # Create a mask where the absolute difference between raw_depth and new_depth is greater than or equal to 0.01
                mask_diff = np.abs(raw_depth - new_depth) >= self.st.depth_diff_value

                # Combine the two masks
                mask = np.logical_or(mask_zero, mask_diff)

                # Convert the boolean mask to an integer mask with values 0 and 255
                mask = np.where(mask, 0, 255)

            if path[-5] == "D" and self.st.D_view_mask is not None:     # sam added for down view mask
                mask = mask * self.st.D_view_mask       
            
            cv2.imwrite(path, mask)

        def merge_mask(self, path: str, raw_depth_img:str =  None, raw_image_path: str = None):
            mask = self.cast_result["t_hit"].numpy()

            if raw_depth_img is None:
                mask = np.where(mask != np.inf, 255, mask)
                mask = np.where(mask == np.inf, 0, mask)
            else:
                raw_depth  = cv2.imread(raw_depth_img, cv2.IMREAD_GRAYSCALE)
                new_depth = np.where(mask == np.inf, 0, mask)
                new_depth = np.where(new_depth < 10.0, new_depth / 10.0, 2.0 - 10 / new_depth) / 2.0
                new_depth *= 255
                
                # for i in range(raw_depth.shape[0]):
                #     for j in range(raw_depth.shape[1]):
                #         if raw_depth[i][j] == 0 or new_depth[i][j] == 0:
                #             mask[i][j] = 0
                #         elif abs(raw_depth[i][j] - new_depth[i][j]) >= 0.01:
                #             mask[i][j] = 0
                #         else:
                #             mask[i][j] = 255
                # Create a mask where either raw_depth or new_depth is 0
                mask_zero = np.logical_or(raw_depth == 0, new_depth == 0)

                # Create a mask where the absolute difference between raw_depth and new_depth is greater than or equal to 0.01
                mask_diff = np.abs(raw_depth - new_depth) >= self.st.depth_diff_value

                # Combine the two masks
                mask = np.logical_or(mask_zero, mask_diff)

                # Convert the boolean mask to an integer mask with values 0 and 255
                mask = np.where(mask, 0, 255)

            if path[-5] == "D" and self.st.D_view_mask is not None:     # sam added for down view mask
                mask = mask * self.st.D_view_mask

            if raw_image_path is None:
                raw_image_path = os.path.join(os.path.dirname(os.path.dirname(path)), "raw", os.path.basename(path)[:-4] + '.jpg')

            raw_image = cv2.imread(raw_image_path)

            alpha_image = np.zeros((raw_image.shape[0], raw_image.shape[1], 4), dtype=np.uint8)

            alpha_image[..., :3] = raw_image
            alpha_image[..., 3] = mask

            cv2.imwrite(path, alpha_image)

        def cal_mask_percentage(self):

            mask = self.cast_result["t_hit"].numpy()

            pixels_seen = len(np.where(mask != np.inf)[0])
            pixels_unseen = len(np.where(mask == np.inf)[0])

            return pixels_seen / (pixels_seen + pixels_unseen)
        
        def get_hit_triangles_index(self):

            

            hit_triangles = self.cast_result['primitive_ids'][self.cast_result['primitive_ids'] != -1].numpy()

            hit_triangles_indexes = np.unique(hit_triangles)

            
            return  hit_triangles_indexes
        
    class MeshProcess():
        def __init__(self, outerclass):
            self.st = outerclass
        
        def crop_mesh(self, bbox = None, quad = None):

            triangle_centers = self._get_triangle_centers()
            cropped_index = []
            for index, coor in enumerate(triangle_centers):
                if bbox is not None:
                    if self._in_box(coor, bbox):
                        cropped_index.append(index)

                elif quad is not None:
                    coor_xy = coor[0:2]
                    if self._is_point_inside_quad(quad, coor_xy):
                        cropped_index.append(index)
            self.cropped_index = cropped_index

            if len(self.cropped_index) == 0:
                print("No triangles in the given bbox.")
                exit()
                raise RuntimeError("No triangles in the given bbox.")

            self.bbox = bbox
        
        def _get_triangle_centers(self):
            triangles = np.asarray(self.st.mesh.triangles)
            vertices = np.asarray(self.st.mesh.vertices)
            triangles_centers = []

            for triangle_vertex in triangles:
                vertex_a = np.array(vertices[triangle_vertex[0]])
                vertex_b = np.array(vertices[triangle_vertex[1]])
                vertex_c = np.array(vertices[triangle_vertex[2]])

                vertex = np.stack((vertex_a, vertex_b, vertex_c), axis = 0)
                center = np.mean(vertex, axis = 0)
                triangles_centers.append(center)
            
            return np.array(triangles_centers)

        def _in_box(self, coordinate, bbox):
            if coordinate[0] >= bbox[0] and coordinate[0] <= bbox[3] and \
                coordinate[1] >= bbox[1] and coordinate[1] <= bbox[4] and \
                coordinate[2] >= bbox[2] and coordinate[2] <= bbox[5]:
                return True
            else: return False

        def save_cropped_mesh(self, path = None):
            cropped_triangles = np.asarray(deepcopy(self.st.mesh.triangles))[self.cropped_index,:]
            cropped_vertices = np.asarray(deepcopy(self.st.mesh.vertices))
            cropped_colors = np.asarray(deepcopy(self.st.mesh.vertex_colors))

            cropped_mesh = o3d.geometry.TriangleMesh()
            cropped_mesh.vertices = o3d.utility.Vector3dVector(cropped_vertices)
            cropped_mesh.vertex_colors = o3d.utility.Vector3dVector(cropped_colors)
            cropped_mesh.triangles = o3d.utility.Vector3iVector(cropped_triangles)

            if path is None:
                path = os.path.join(os.path.dirname(self.st.mesh_path), "cropped_mesh.ply") # .obj

            o3d.io.write_triangle_mesh(path, cropped_mesh, True)

        def calculate_percentage(self, indices_list: list):
            # num(triangles ids in cropped_mesh triangle ids) / num(triangle ids)
            num_triangles = len(indices_list)#len(self.cropped_index)

            num_in_cropped_mesh = np.intersect1d(indices_list, self.cropped_index).size

            if num_triangles == 0:
                percent = 0
            else:
                percent = num_in_cropped_mesh / num_triangles

            return percent
        
        def get_crop_bbox(self):
            return self.bbox
        
        def _cross_product(self, p1, p2):
            return p1[0] * p2[1] - p1[1] * p2[0]
        
        def _is_left_turn(self, p, q, r):
            return self._cross_product([q[0]-p[0], q[1]-p[1]], [r[0]-q[0], r[1]-q[1]]) > 0
        
        def _is_point_inside_quad(self, quad, point):
            if len(quad) != 4:
                raise ValueError("The quadrilateral must have exactly 4 points")

            check = self._is_left_turn(quad[0], quad[1], point)
            for i in range(1, 4):
                if self._is_left_turn(quad[i], quad[(i+1)%4], point) != check:
                    return False
            return True

    def load_mesh(self, simplify = True):

        if simplify:
            import time
            start_time = time.time()
            print("[ INFO ] Simplifying mesh...")
            ms = ml.MeshSet()
            ms.load_new_mesh(self.mesh_path)
            ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=1000000)
            self.mesh_path = os.path.join(os.path.dirname(self.mesh_path), "mesh_1m.ply")
            ms.save_current_mesh(self.mesh_path)
            end_time = time.time()
            print("[ INFO ] Mesh Simplified to 1m faces in %f seconds." %(end_time - start_time))
        self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        print("[ INFO ] Mesh Loaded.")

        # if  self.mesh.has_triangle_uvs() and self.mesh.texture is not None and self.mesh.texture.image is not None:
        #     self.texture_to_vertex_color()

    def delete_mesh(self):
        self.mesh = None

    def _texture_to_vertex_color(self):
        raise NotImplementedError
    
    def get_mesh_dimensions(self):
        # in form of [x_min, x_max, y_min, y_max, z_min, z_max]
        min_bounds = self.mesh.get_min_bound()
        max_bounds = self.mesh.get_max_bound()
        bbox = [min_bounds[0], max_bounds[0], min_bounds[1], max_bounds[1], min_bounds[2], max_bounds[2]]
        return bbox
    
    def get_mesh_bounding_box(self):
        # in form of [X, Y, Z]
        min_bounds = self.mesh.get_min_bound()
        max_bounds = self.mesh.get_max_bound()

        bbox_lengths = [max_bounds[0] - min_bounds[0], max_bounds[1] - min_bounds[1], max_bounds[2] - min_bounds[2]]
        return bbox_lengths
    
    class RenderImg:
        def __init__(self, outerclass):

            self.st = outerclass

        def set_vis(self, intrinsics, width, height):
            self.cam = o3d.camera.PinholeCameraParameters()
            self.cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2])

            init_pose = np.eye(4)
            self.cam.extrinsic = np.linalg.inv(init_pose)
            vis = o3d.visualization.Visualizer()
            vis.create_window(width = width, height = height, visible = False)
            vis.add_geometry(self.st.mesh)
            self.vis = vis
            

        def render_img(self, extrinsics, path):
            
            self.cam.extrinsic = np.linalg.inv(extrinsics)
        
            self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam, allow_arbitrary=True)
            self.vis.update_geometry(self.st.mesh)
            self.vis.poll_events()
            self.vis.update_renderer()
            image = (np.array(self.vis.capture_screen_float_buffer(True))* 255.0).astype(np.uint8)
            image = image[:, :, [2, 1, 0]]  # RGB to BGR

            cv2.imwrite(path, image)

        def destroy_vis(self):
            self.vis.destroy_window()

"""
Reachable Vriable list:
Class MeshImageFilter:
    mesh_path: list
    mesh: o3d.geometry.TriangleMesh
    raycast_scene: o3d.t.geometry.RaycastingScene

    Class RaycastScene:
        st: MeshImageFilder
        cast_result: dict, see -> http://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html?highlight=raycast
        filtered_indices: list
        rays: tensor, see -> http://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html?highlight=raycast

"""

"""
Function List:
Class MeshImageFilter:
    __init__(self, mesh_path, load_mesh = True)
    load_mesh(self)
    delete_mesh(self)

    Class RaycastScene:
        __init__(self, outerclass) -> InitSceneWithMesh(self)
        InitSceneWithMesh(self): Use a mesh to initialize a RaycastScene
        CastScene(self, intrinsic, extrinsic, width, height)
        triangles_per_view(self): calculate how many triangles can be seen in the given view
        save_visible_triangles(self, path = None)
        save_visible_depth_pcd(self, path = None)
        save_visible_depth_img(self, path)
        save_mask_depth(self, path)

    Class MeshProcess:
        __init__(self, outerclass)
        crop_mesh(self, bbox)
        save_cropped_mesh(self, path = None)
        _get_triangle_centers(self)
        _inbox(self, coordinate, bbox)
        calculate_percentage(self)
"""