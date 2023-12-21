import numpy as np
import open3d as o3d
import cv2
import sys
import glob
# 创建半径为10米的球
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
sphere = sphere.subdivide_midpoint(number_of_iterations=4)

# 创建旋转矩阵
rotate_front = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, 0))
rotate_left = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 4, 0))
rotate_right = o3d.geometry.get_rotation_matrix_from_xyz((0, -np.pi / 4, 0))
rotate_down = o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 4, 0, 0))
rotate_up = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 4, 0, 0))

rotations = [rotate_front, rotate_left, rotate_right, rotate_up, rotate_down]

# 创建场景并加入三角面片
scene = o3d.t.geometry.RaycastingScene()
mesh_for_raycast = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

scene.add_triangles(mesh_for_raycast)

# 创建相机参数
width_px = 1000
height_px = 1000
fov = 90
# calculate focal length
focal_length = (width_px / 2) / np.tan(np.deg2rad(fov / 2))
# matrix
intrinsic_matrix = np.array([[focal_length, 0, width_px / 2],
                             [0, focal_length, height_px / 2],
                             [0, 0, 1]])
print(intrinsic_matrix)
intrinsic_matrix = o3d.core.Tensor(intrinsic_matrix, dtype=o3d.core.float64)
# 分别对应四个方向旋转
points = []
for rotation in rotations:

    # 创建旋转后的相机的外参矩阵
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation
    # to open3d
    extrinsic_matrix = o3d.core.Tensor(extrinsic_matrix, dtype=o3d.core.float64)

    # 创建pinhole的光线
    rays = scene.create_rays_pinhole(intrinsic_matrix, extrinsic_matrix, width_px, height_px)
    # 光线追踪并获取结果
    ans = scene.cast_rays(rays)
    # print some rays
    # convert to numpy
    ans['t_hit'] = ans['t_hit'].numpy()
    rays = rays.numpy()

    # get the intersection points from ans['t_hit'] and direction from rays. rays shape is (1000,1000,6) ans['t_hit'] is (1000,1000)
    point = rays[..., 3:] * ans['t_hit'][..., None]
    # save the point cloud, point shape is (1000,1000,3)
    # reshape to (1000000,3)
    # point = point.reshape(-1, 3)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point)
    # o3d.io.write_point_cloud(f'point_cloud_{rotation}.ply', pcd)
    points.append(point)

#3072 3072 956.85568340139776 956.79150117283348  1552.7335463532427 0.02877005132633028 -0.016963323388509872 0.0010226479526464832 -0.00028283043660053391
intrinsiccalib = np.array([[956.85568340139776, 0, 1558.7467810169978], [0, 956.79150117283348, 1552.7335463532427], [0, 0, 1]])
distortion = np.array([0.02877005132633028, -0.016963323388509872, 0.0010226479526464832, -0.00028283043660053391])

#get path from argv
path = sys.argv[1]
# get all the images
filenames = glob.glob(path + '/*.jpg')
prefix = ['_F','_L','_R','_U','_D']
rvecs = np.zeros((3, 1))
tvecs = np.zeros((3, 1))
for filename in filenames:
    # 读取图片
    img = cv2.imread(filename)
    # 读取图片的大小
    h, w = img.shape[:2]
    #identiy matrix
    for i in range(len(prefix)):
        # reshape the point to (1000000,3)
        pointsreshape = points[i].reshape(-1, 3)
        #convert to (1000000,1,3)
        pointsreshape = pointsreshape.reshape(-1, 1, 3)
        imagePoints, _ = cv2.fisheye.projectPoints(pointsreshape, rvecs, tvecs, intrinsiccalib, distortion)
        print(imagePoints.shape)
        imagePoints = imagePoints.reshape(1000, 1000, 2)
        # new image is 1000 * 1000 image get the color from the original image
        # new_img = np.zeros((1000, 1000, 3))
        # for j in range(1000):
        #     for k in range(1000):
        #         new_img[j, k] = img[int(imagePoints[j, k, 1]), int(imagePoints[j, k, 0])]
        new_img = img[np.round(imagePoints[..., 1]).astype(int), np.round(imagePoints[..., 0]).astype(int)]
        # save the new image
        cv2.imwrite(filename[:-4] + prefix[i] + '.jpg', new_img)
